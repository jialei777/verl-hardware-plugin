# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatches for Google TPU (``tpu_dist``) distributed collectives and runtime helpers.

Why this is needed:
1. A TPU tensor can reach a process group that only has a ``cpu:gloo`` backend. ``all_reduce`` then
   stages it through CPU; Gloo has no ``ReduceOp.AVG``, so AVG becomes SUM + divide on that path.
   With torch-tpu >= 0.2.0, ``ReduceOp.AVG`` on ``tpu_dist`` is native and ``all_gather_object`` on
   ``tpu_dist`` groups works (``aten::resize_`` is implemented), so both pass through unpatched.
2. ``verl.utils.distributed.set_numa_affinity`` queries ``pynvml`` (NVIDIA NVML) when
   ``libnuma.so`` is present, and ``verl.utils.torch_functional.logprobs_from_logits``
   falls back to row-wise Python loops (``logprobs_from_logits_v2``) instead of vectorized
   ``logprobs_from_logits_naive`` (run in fp32, as on GPU) on TPU.

All patches are applied lazily when ``PlatformTPU()`` is constructed (i.e. only when the
``tpu`` platform is selected for the process).
"""

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

import verl.utils.distributed as verl_dist
import verl.utils.torch_functional as verl_tf

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False
_runtime_helpers_applied = False


def _tpu_available() -> bool:
    if not hasattr(torch, "tpu"):
        try:
            import torch_tpu  # noqa: F401
        except Exception:
            return False
    from verl_hardware_plugin.accelerators.tpu import device_api

    return device_api.is_available()


def _is_gloo_only_group_with_tpu_tensor(tensor: Any, group: Any) -> bool:
    if group is None or torch.compiler.is_compiling():
        return False
    device = getattr(tensor, "device", None)
    if getattr(device, "type", None) != "tpu":
        return False
    try:
        backend = str(dist.get_backend(group))
    except Exception:
        return False
    return "gloo" in backend and "tpu_dist" not in backend


def _noop_numa_affinity() -> None:
    return None


def _apply_verl_runtime_helpers() -> None:
    global _runtime_helpers_applied
    if _runtime_helpers_applied:
        return

    try:
        if hasattr(verl_dist, "set_numa_affinity"):
            try:
                verl_dist.set_numa_affinity.__code__ = _noop_numa_affinity.__code__
            except Exception:
                pass
            verl_dist.set_numa_affinity = _noop_numa_affinity
    except Exception as e:
        logger.debug("Skipping set_numa_affinity TPU patch: %s", e)

    try:
        orig_logprobs = getattr(verl_tf, "logprobs_from_logits", None)
        naive_logprobs = getattr(verl_tf, "logprobs_from_logits_naive", None)
        if callable(orig_logprobs) and callable(naive_logprobs) and not getattr(orig_logprobs, "_tpu_patched", False):

            def _patched_logprobs_from_logits(logits, labels, inplace_backward=True):
                device = getattr(logits, "device", None)
                if getattr(device, "type", None) == "tpu":
                    # fp32 log-softmax, as flash-attn's cross entropy computes it on GPU.
                    return naive_logprobs(logits.float(), labels)
                return orig_logprobs(logits, labels, inplace_backward=inplace_backward)

            _patched_logprobs_from_logits._tpu_patched = True  # type: ignore[attr-defined]
            verl_tf.logprobs_from_logits = _patched_logprobs_from_logits
    except Exception as e:
        logger.debug("Skipping logprobs_from_logits TPU patch: %s", e)

    _runtime_helpers_applied = True


def apply() -> None:
    global _applied
    _apply_verl_runtime_helpers()
    if _applied:
        return
    if not _tpu_available():
        return

    original_all_reduce = dist.all_reduce

    def _patched_all_reduce(tensor, op=None, group=None, async_op=False):
        # Forward `op` only when the caller passed one. Under torch.compile, dynamo rewrites the original
        # `dist.all_reduce` into a functional collective, but it only converts a ReduceOp to its string form when
        # the traced function is the current `dist.all_reduce` attribute (i.e. this wrapper). A forwarded
        # `ReduceOp.SUM` default would reach `_functional_collectives.all_reduce` unconverted and fail with
        # "'RedOpType' object has no attribute 'lower'" (e.g. vLLM's compiled tensor-parallel all-reduce).
        if not async_op and _is_gloo_only_group_with_tpu_tensor(tensor, group):
            cpu_tensor = tensor.detach().cpu()
            if op is None:
                result = original_all_reduce(cpu_tensor, group=group, async_op=False)
            elif op == dist.ReduceOp.AVG:
                world_size = dist.get_world_size(group=group)
                result = original_all_reduce(cpu_tensor, op=dist.ReduceOp.SUM, group=group, async_op=False)
                cpu_tensor.div_(world_size)
            else:
                result = original_all_reduce(cpu_tensor, op=op, group=group, async_op=False)
            tensor.copy_(cpu_tensor)
            return result
        if op is None:
            return original_all_reduce(tensor, group=group, async_op=async_op)
        # torch-tpu >= 0.2.0 implements ReduceOp.AVG on tpu_dist natively (SUM + divide inside the HLO).
        return original_all_reduce(tensor, op=op, group=group, async_op=async_op)

    dist.all_reduce = _patched_all_reduce

    _applied = True
    logger.info("[verl_hardware_plugin] Patched torch.distributed.all_reduce for TPU tensors on Gloo-only groups")

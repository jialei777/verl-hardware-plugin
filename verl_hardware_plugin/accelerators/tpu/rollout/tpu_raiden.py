# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM worker extension that receives the trainer's weights over Raiden (tpu-sync) on TPU.

``TPUvLLMHttpServer`` passes this class to vLLM as ``worker_extension_cls`` when
``actor_rollout_ref.rollout.checkpoint_engine.backend=raiden``. The driver
(``update_raiden_weights`` in ``verl_hardware_plugin.accelerators.tpu.engines.raiden_checkpoint_engine``) calls its
methods on every vLLM worker through ``collective_rpc``:

1. ``init_raiden_sync_on_worker``: allocate TP-sharded receive buffers for the tensors the trainer
   published and register them with the Raiden controller (first sync only).
2. ``install_raiden_weights``: after the transfer, copy the received buffers to TPU HBM and fuse or
   transpose them into vLLM's parameters (every sync).
3. ``get_model_weights_stats``: weight norms for the optional parity check.

This module imports vLLM (through the upstream worker extension), so only vLLM workers import it.
"""

import logging
import math
import os
import time
from typing import Any, Optional

import ray
import torch

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from verl_hardware_plugin.accelerators.tpu.engines import raiden_checkpoint_engine as raiden
from verl_hardware_plugin.accelerators.tpu.engines.ray_weight_registry import get_ray_weight_registry
from verl_hardware_plugin.accelerators.tpu.engines.tpu_checkpoint_engine import (
    _get_parent_module,
    _resolve_model_key,
    _to_target_layout,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# How long a rollout worker waits for the controller address and the trainer's tensor shapes.
_REGISTRY_POLL_ATTEMPTS = 30
_REGISTRY_POLL_INTERVAL_S = 0.5
# Row-parallel layers: vLLM shards their input dimension (dim 1) across TP ranks; every other 2-D weight
# whose output dimension divides evenly is sharded on dim 0.
_ROW_PARALLEL_SUFFIXES = (".o_proj.weight", ".down_proj.weight")
_KV_PROJ_SUFFIXES = (".self_attn.k_proj.weight", ".self_attn.v_proj.weight")
# vLLM fuses these trainer tensors into one parameter: (fused suffix, source suffixes in concat order).
_FUSED_TARGETS = (
    (
        ".self_attn.qkv_proj.weight",
        (".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"),
    ),
    (
        ".self_attn.qkv_proj.bias",
        (".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"),
    ),
    (".mlp.gate_up_proj.weight", (".mlp.gate_proj.weight", ".mlp.up_proj.weight")),
    (".mlp.gate_up_proj.bias", (".mlp.gate_proj.bias", ".mlp.up_proj.bias")),
)


def _sampler_sharding(name: str, global_shape: list[int], tp_size: int) -> tuple[list[str], list[int]]:
    """Returns ``(sharding_spec, local_shape)`` of a trainer tensor on one vLLM TP rank."""
    spec_axes = [""] * len(global_shape)
    local_shape = list(global_shape)
    if len(global_shape) == 2:
        if any(name.endswith(s) for s in _ROW_PARALLEL_SUFFIXES):
            # Row parallel: partition input dimension (dim 1) across TP ranks
            spec_axes = ["", "tp"]
            local_shape[1] = global_shape[1] // tp_size
        elif global_shape[0] % tp_size == 0:
            # Column parallel or vocab parallel: partition output dimension (dim 0) across TP ranks
            spec_axes = ["tp", ""]
            local_shape[0] = global_shape[0] // tp_size
    return spec_axes, local_shape


class vLLMRaidenWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM Worker Extension for Google Cloud TPU with Raiden P2P weight synchronization."""

    def _get_vllm_model(self) -> Any:
        """Extract the underlying nn.Module from the vLLM V1 worker."""
        worker: Any = getattr(self, "worker", self)
        return worker.model_runner.model

    def _wait_for_raiden_registry(self, rank_val: int) -> tuple[str, dict[str, list[int]]]:
        """Returns the controller address and the trainer's global tensor shapes from ``RayWeightRegistry``."""
        registry = get_ray_weight_registry()
        for _ in range(_REGISTRY_POLL_ATTEMPTS):
            try:
                addr, shapes = ray.get([registry.get_controller_address.remote(), registry.get_global_shapes.remote()])
                if addr and shapes:
                    return addr, shapes
            except Exception as e:
                logger.debug(f"Raiden Sampler Rank {rank_val}: registry lookup failed: {e}")
            time.sleep(_REGISTRY_POLL_INTERVAL_S)
        raise RuntimeError(
            f"Raiden Sampler Rank {rank_val}: No RaidenController address or "
            f"global shapes found in RayWeightRegistry after timeout"
        )

    def _resolve_gqa_kv_head_idx(self, worker: Any, tp_size: int, rank_val: int) -> tuple[Optional[int], Optional[int]]:
        """Return ``(num_kv_heads, kv_head_idx)`` when GQA replicates KV heads across TP ranks."""
        model_config = getattr(getattr(worker, "vllm_config", None), "model_config", None)
        get_kv_heads = getattr(model_config, "get_total_num_kv_heads", None)
        if not callable(get_kv_heads):
            return None, None
        try:
            num_kv_heads = int(get_kv_heads())
        except Exception:
            return None, None
        if num_kv_heads <= 0 or num_kv_heads >= tp_size or tp_size % num_kv_heads != 0:
            return None, None
        tp_rank = rank_val
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            tp_rank = int(get_tensor_model_parallel_rank())
        except Exception:
            pass
        return num_kv_heads, tp_rank // (tp_size // num_kv_heads)

    def init_raiden_sync_on_worker(
        self, parallelism: int = raiden.RAIDEN_DEFAULT_PARALLELISM, job_name: str = "sampler"
    ) -> bool:
        """Initialize Raiden WeightSynchronizer listener and register with central RaidenController.

        Only the first call does work: the receive buffers, the synchronizer and the registration are
        reused by every later sync. ``job_name`` distinguishes multiple rollout replicas on the controller.
        """
        if getattr(self, "_raiden_ws", None) is not None:
            return True

        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            logger.warning("Raiden Sampler: could not locate vllm_model to bind parameters.")
            return False

        from tpu_sync.rpc import raiden_controller, raiden_service_pb2

        bind_ip = ray.util.get_node_ip_address().strip("[]")
        rank_val = getattr(self, "rank", 0)
        # Let the OS pick the control listener port so multiple rollout replicas on a host never collide.
        listener_port = 0

        # 1. Fetch RaidenController address and un-fused global shapes from RayWeightRegistry
        controller_addr, global_shapes_map = self._wait_for_raiden_registry(rank_val)

        # 2. Determine tensor parallel size from vLLM V1 config
        worker: Any = getattr(self, "worker", self)
        parallel_config = getattr(getattr(worker, "vllm_config", None), "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", None)
        if not tp_size or tp_size <= 0:
            raise RuntimeError(
                f"Raiden Sampler Rank {rank_val}: Unable to determine tensor_parallel_size (tp_size) "
                f"from worker.vllm_config.parallel_config!"
            )

        num_kv_heads, kv_head_idx = self._resolve_gqa_kv_head_idx(worker, tp_size, rank_val)

        # 3. Allocate local TPU staging buffers matching Trainer un-fused layout and sharding specs
        staging_tensors = {}
        variable_protos = []
        valid_params = []
        replicas: dict[str, int] = {}

        for idx, (name, g_shape) in enumerate(sorted(global_shapes_map.items(), key=lambda x: x[0])):
            g_shape = list(g_shape)
            global_shard_indices: list[int] = []
            if (
                len(g_shape) == 2
                and kv_head_idx is not None
                and num_kv_heads is not None
                and name.endswith(_KV_PROJ_SUFFIXES)
            ):
                spec_axes = ["tp", ""]
                local_shape = [g_shape[0] // num_kv_heads, g_shape[1]]
                sharding_mesh = [num_kv_heads, 1]
                global_shard_indices = [kv_head_idx]
            else:
                spec_axes, local_shape = _sampler_sharding(name, g_shape, tp_size)
                sharding_mesh = [tp_size if axis == "tp" else 1 for axis in spec_axes]

            replicas[name] = tp_size // int(math.prod(sharding_mesh))

            # Allocate a local staging buffer with matching layout (the trainer sends bf16)
            t = torch.empty(local_shape, dtype=torch.bfloat16, device=torch.device(raiden.RAIDEN_DEVICE))
            staging_tensors[name] = t
            valid_params.append((name, t))

            proto_kwargs: dict[str, Any] = {
                "name": name,
                "shape": g_shape,
                "mesh_shape": sharding_mesh,
                "layout": list(range(len(local_shape) - 1, -1, -1)),
                "item_size": t.element_size(),
                "layer_idx": idx,
                "sharding_spec": spec_axes,
            }
            if global_shard_indices:
                proto_kwargs["global_shard_indices"] = global_shard_indices
            variable_protos.append(raiden_service_pb2.VariableMetadataProto(**proto_kwargs))

        self._raiden_staging = staging_tensors
        self._sorted_vllm_params = valid_params
        self._raiden_replicas = replicas

        raiden.tpu_synchronize()

        logger.info(
            f"Raiden Sampler Rank {rank_val}: binding {len(valid_params)} staging tensors to WeightSynchronizer "
            f"(sample: name={valid_params[0][0]}, shape={valid_params[0][1].shape}, "
            f"total numel={sum(t.numel() for _, t in valid_params)})"
        )

        self._raiden_ws = raiden.create_torch_weight_synchronizer(
            [[t] for _, t in valid_params],
            local_port=0,
            parallelism=parallelism,
            listener_port=listener_port,
            bind_ip=bind_ip,
        )

        # Note: Whether a tensor can skip the CPU (de)tiling pass is decided by Raiden's planner per transfer
        # from the source and destination slices together, and delivered to this listener with the transfer.
        # Setting it here from the local shape alone would make the two sides disagree whenever a trainer
        # FSDP shard is not 8-row aligned (e.g. Qwen3 embedding at 32+ FSDP ranks).

        try:
            ctrl_client = raiden_controller.RaidenControllerClientFacade(controller_addr)
            unit_id = raiden_controller.RaidenId(job_name, str(rank_val), "weights")
            ctrl_client.register_work_unit(
                unit_id,
                [f"{bind_ip}:{self._raiden_ws.local_port}"],
                f"{bind_ip}:{self._raiden_ws.listener_port}",
                mesh_shape=[1, tp_size],
                variables=variable_protos,
                mesh_axes=["fsdp", "tp"],
            )
        except Exception as e:
            logger.error(f"Raiden Sampler Rank {rank_val} failed to register with RaidenController: {e}")
            raise
        logger.info(
            f"Raiden Sampler Rank {rank_val} bound {len(valid_params)} dynamic staging tensors and registered "
            f"directly with RaidenController ({controller_addr}): mesh_shape=[1, {tp_size}], "
            f"data_port={self._raiden_ws.local_port}, listener_port={self._raiden_ws.listener_port}"
        )

        return True

    @torch.no_grad()
    def install_raiden_weights(self, exact_parity: bool = False) -> dict[str, Any]:
        """Install received weights from host staging buffer into TPU HBM via zero-copy H2D DMA
        and fuse/transpose them directly into vLLM model parameters.
        """
        if getattr(self, "_raiden_ws", None) is None:
            logger.warning("Raiden Sampler: install_raiden_weights called before _raiden_ws was initialized.")
            return {}

        t_start = time.perf_counter()
        t_h2d_start = time.perf_counter()
        self._raiden_ws.h2d()
        t_h2d = time.perf_counter() - t_h2d_start

        # Unpack, fuse, and transpose staging tensors into vLLM model parameters
        vllm_model = self._get_vllm_model()
        model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else vllm_model.model.state_dict()
        module_dict = dict(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else {}

        parity = raiden.RaidenParityCheck("exact") if exact_parity else None

        def resolve_model_key(k: str) -> Optional[str]:
            resolved = _resolve_model_key(k, model_sd)
            return resolved if resolved in model_sd else None

        def copy_into(resolved_key: str, src: torch.Tensor) -> None:
            target_param = model_sd[resolved_key]
            target_local = target_param.to_local() if hasattr(target_param, "to_local") else target_param
            parent_mod = _get_parent_module(resolved_key, module_dict)
            is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))
            adapted = _to_target_layout(src, target_local, is_flipped)
            if parity is not None:
                parity.check_exact(resolved_key, target_local, adapted)
            target_local.copy_(adapted)

        consumed_staging = set()

        # 1. Handle fused projections (QKV and Gate-Up)
        for target_suffix, src_suffixes in _FUSED_TARGETS:
            primary_src = src_suffixes[0]
            for name in list(self._raiden_staging.keys()):
                if primary_src in name:
                    layer_src_keys = [name.replace(primary_src, s) for s in src_suffixes]
                    if all(sk in self._raiden_staging for sk in layer_src_keys):
                        resolved_key = resolve_model_key(name.replace(primary_src, target_suffix))
                        if resolved_key is not None:
                            fused = torch.cat([self._raiden_staging[sk] for sk in layer_src_keys], dim=0)
                            copy_into(resolved_key, fused)
                            consumed_staging.update(layer_src_keys)

        # 2. Handle all remaining non-fused parameters (o_proj, down_proj, layernorms, embeddings)
        unresolved: list[str] = []
        for name, src_t in self._raiden_staging.items():
            if name in consumed_staging:
                continue
            resolved_key = resolve_model_key(name)
            if resolved_key is not None:
                copy_into(resolved_key, src_t)
            else:
                unresolved.append(name)

        # 3. Handle tied word embeddings. Only when the trainer did not send lm_head itself: for untied
        #    models (e.g. Qwen3-8B / 32B) the received lm_head must not be overwritten by embed_tokens.
        received_lm_head = any(k == "lm_head.weight" or k.endswith(".lm_head.weight") for k in self._raiden_staging)
        if received_lm_head:
            pass
        elif (
            hasattr(vllm_model, "lm_head")
            and hasattr(vllm_model, "model")
            and hasattr(vllm_model.model, "embed_tokens")
        ):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.model.embed_tokens.weight)
        elif hasattr(vllm_model, "lm_head") and hasattr(vllm_model, "embed_tokens"):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.embed_tokens.weight)

        t_sync_start = time.perf_counter()
        raiden.tpu_synchronize()
        t_sync = time.perf_counter() - t_sync_start
        t_total = time.perf_counter() - t_start

        logger.info(
            f"[RAIDEN TELEMETRY | Sampler Worker] Sampler Rank {getattr(self, 'rank', 0)}: install_raiden_weights "
            f"completed in {t_total:.4f}s (H2D={t_h2d:.4f}s, TPUSyncBarrier={t_sync:.4f}s)"
        )
        result: dict[str, Any] = {"total": t_total, "h2d": t_h2d, "sync": t_sync}
        if parity is not None:
            result["exact_parity"] = parity.exact_result(getattr(self, "rank", 0), unresolved)
        return result

    def get_model_weights_stats(self, include_shards: bool = False) -> dict:
        """Computes deterministic parameter count, L1 norm, and L2 norm of this worker's received weights."""
        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            return {"error": "No model found on worker"}

        if not getattr(self, "_sorted_vllm_params", None):
            raise RuntimeError(
                f"get_model_weights_stats is only supported for Raiden after initialization, "
                f"but self._sorted_vllm_params is not initialized on worker rank {getattr(self, 'rank', 'unknown')}!"
            )

        res = raiden.compute_tensor_stats(self._sorted_vllm_params)
        res["rank"] = getattr(self, "rank", 0)
        replicas = getattr(self, "_raiden_replicas", {})
        for name, stats in res["per_tensor"].items():
            if name in replicas:
                stats["replicas"] = replicas[name]
        return res

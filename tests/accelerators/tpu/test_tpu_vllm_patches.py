# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""The FP32 sampler / log-prob patches adapt to the installed vllm-torchtpu."""

import sys
import types
from unittest import mock

import torch

from verl_hardware_plugin.accelerators.tpu.rollout import tpu_vllm_patches


def _install_fake_vllm(runner_cls):
    runner_mod = types.ModuleType("vllm_torchtpu.runner.tpu_runner")
    runner_mod.TPUModelRunner = runner_cls
    outputs_mod = types.ModuleType("vllm.v1.outputs")
    outputs_mod.LogprobsTensors = object
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.runner": types.ModuleType("vllm_torchtpu.runner"),
        "vllm_torchtpu.runner.tpu_runner": runner_mod,
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.outputs": outputs_mod,
    }
    return (
        mock.patch.dict(sys.modules, modules),
        mock.patch.object(tpu_vllm_patches.importlib.util, "find_spec", return_value=object()),
    )


def test_sampler_patch_wraps_bf16_sampler_with_extra_parameters():
    """Newer signatures append parameters (topk_candidates); the BF16 noise is still redrawn in FP32."""
    seen = {}

    class Runner:
        def sample_from_logits(
            self, logits, temperatures, u, top_k, top_p, all_greedy=False, topk_candidates=0
        ):  # draws noise in the dtype it is given
            seen.update(u_dtype=u.dtype, topk_candidates=topk_candidates)
            return logits

    p1, p2 = _install_fake_vllm(Runner)
    with p1, p2:
        tpu_vllm_patches.patch_tpu_sampler()
        assert Runner._verl_fp32_sampler_patched is True
        Runner().sample_from_logits(
            torch.zeros(2, 4), torch.ones(2), torch.zeros(2, 4, dtype=torch.bfloat16), None, None, False, 7
        )
    assert seen == {"u_dtype": torch.float32, "topk_candidates": 7}


def test_patches_are_skipped_when_vllm_torchtpu_already_uses_fp32():
    """vllm-torchtpu #1747 / #2024 compute in FP32 themselves: keep their implementations."""

    class Runner:
        def sample_from_logits(self, logits, temperatures, u, top_k, top_p, all_greedy=False):
            u = u.to(torch.float32)
            return logits

        def gather_logprobs(self, logits, sampled_tokens):
            return logits.to(torch.float32)

    original_sampler, original_logprobs = Runner.sample_from_logits, Runner.gather_logprobs
    p1, p2 = _install_fake_vllm(Runner)
    with p1, p2:
        tpu_vllm_patches.patch_tpu_sampler()
        tpu_vllm_patches.patch_tpu_logprobs()
    assert Runner.sample_from_logits is original_sampler
    assert Runner.gather_logprobs is original_logprobs
    assert Runner._verl_fp32_sampler_patched and Runner._verl_fp32_logprobs_patched

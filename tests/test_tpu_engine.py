# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the TPU TorchTitan engine utilities on CPU."""

import os
from unittest import mock

import torch


def test_bucket_length_and_env_override():
    from verl_hardware_plugin.engines.tpu_utils import bucket_length, get_tpu_seq_bucket_size

    assert get_tpu_seq_bucket_size() == 256
    assert bucket_length(1) == 256
    assert bucket_length(256) == 256
    assert bucket_length(257) == 512
    assert bucket_length(70, bucket_size=64) == 128

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "64"}):
        assert get_tpu_seq_bucket_size() == 64
        assert bucket_length(65) == 128


def test_unwrap_metadata():
    from verl_hardware_plugin.engines.tpu_utils import unwrap_metadata

    assert unwrap_metadata([torch.tensor(3.5)]) == 3.5
    assert unwrap_metadata((True, False)) is True
    assert unwrap_metadata("flex") == "flex"


def test_pad_packed_inputs_for_tpu_builds_4d_document_causal_mask():
    from tensordict import TensorDict

    from verl_hardware_plugin.engines.tpu_utils import pad_packed_inputs_for_tpu

    # Two packed documents of lengths 3 and 2 -> orig_seq_len = 5
    input_ids = torch.nested.nested_tensor(
        [torch.tensor([10, 11, 12]), torch.tensor([20, 21])],
        layout=torch.jagged,
    )
    position_ids = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 2]), torch.tensor([0, 1])],
        layout=torch.jagged,
    )
    micro_batch = TensorDict({}, batch_size=[])

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "8"}):
        _, _, _, attention_masks, orig_seq_len = pad_packed_inputs_for_tpu(
            input_ids=input_ids,
            position_ids=position_ids,
            micro_batch=micro_batch,
            device=torch.device("cpu"),
        )

    assert orig_seq_len == 5
    assert attention_masks.shape == (1, 1, 8, 8)
    assert attention_masks.dtype == torch.bool
    # Document 0 (tokens 0..2) attends causally within [0..2] and not to document 1 (tokens 3..4)
    assert attention_masks[0, 0, 2, 0].item() is True
    assert attention_masks[0, 0, 0, 2].item() is False
    assert attention_masks[0, 0, 3, 2].item() is False
    assert attention_masks[0, 0, 4, 3].item() is True
    # Padded tail (tokens 5..7) has self-attention only (no cross-token attention)
    assert attention_masks[0, 0, 6, 6].item() is True
    assert attention_masks[0, 0, 6, 5].item() is False


def test_replace_varlen_attention_with_tpu_attention():
    from verl_hardware_plugin.engines.tpu_utils import (
        TPUVarlenAttention,
        replace_varlen_attention_with_tpu_attention,
    )

    class _DummyAttnBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = torch.nn.Identity()

    model = torch.nn.Sequential(_DummyAttnBlock(), _DummyAttnBlock())
    assert replace_varlen_attention_with_tpu_attention([model]) == 2
    assert isinstance(model[0].inner_attention, TPUVarlenAttention)
    assert isinstance(model[1].inner_attention, TPUVarlenAttention)
    # Idempotent when called a second time
    assert replace_varlen_attention_with_tpu_attention([model]) == 0

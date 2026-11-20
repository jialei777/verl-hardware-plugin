# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for ``gdn_slot_schedule`` and its ``TorchTitanTPUEngineWithLMHead`` input/output wiring."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl_hardware_plugin.accelerators.tpu.engines import gdn_slot_schedule, torchtitan_tpu


def _make_packed_mb(
    seqs_pos: list[list[int]],
    *,
    vocab: int = 32,
    padded_seq_len: int | None = None,
    calculate_entropy: bool = False,
    seed: int = 0,
) -> TensorDict:
    gen = torch.Generator().manual_seed(seed)
    id_seqs = [torch.randint(1, vocab, (len(p),), generator=gen, dtype=torch.int64) for p in seqs_pos]
    pos_seqs = [torch.tensor(p, dtype=torch.int64) for p in seqs_pos]
    mb = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(id_seqs, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(pos_seqs, layout=torch.jagged),
        },
        batch_size=[len(seqs_pos)],
    )
    tu.assign_non_tensor(
        mb,
        use_remove_padding=True,
        calculate_entropy=calculate_entropy,
        temperature=1.0,
    )
    if padded_seq_len is not None:
        tu.assign_non_tensor_data(mb, "tpu_padded_seq_len", padded_seq_len)
    return mb


def test_has_internal_document_resets_checks_exact_sequence_starts():
    # Two standard single-document sequences starting at 0.
    pos = torch.tensor([0, 1, 2, 0, 1, 2, 3])
    cu = torch.tensor([0, 3, 7])
    assert gdn_slot_schedule.has_internal_document_resets(pos, cu) is False

    # Sub-document reset inside the first sequence.
    pos_multi = torch.tensor([0, 1, 0, 1, 0, 1, 2])
    assert gdn_slot_schedule.has_internal_document_resets(pos_multi, cu) is True

    # Continuation sequence starting at position > 0 that later resets to 0 inside the same sequence:
    # (pos == 0).sum() == 1 <= n_seqs == 1, yet index 2 is an interior document reset.
    pos_cont_reset = torch.tensor([5, 6, 0, 1])
    cu_single = torch.tensor([0, 4])
    assert gdn_slot_schedule.has_internal_document_resets(pos_cont_reset, cu_single) is True

    # Continuation sequence without interior reset.
    pos_cont_clean = torch.tensor([5, 6, 7, 8])
    assert gdn_slot_schedule.has_internal_document_resets(pos_cont_clean, cu_single) is False


def test_classify_and_sort_micro_batches_agree_on_categories():
    # padded_len=16, auto_slot_len=8 -> 2 row slots of length 8.
    mb_seg = _make_packed_mb([[0, 1, 0, 1], [0, 1, 2]], padded_seq_len=16)  # internal reset -> SEGMENTED (4+3=7)
    mb_single = _make_packed_mb([list(range(12))], padded_seq_len=16)  # 1 seq > 8, <= 16 -> SINGLE_DOC (12)
    mb_row_small = _make_packed_mb([list(range(5)), list(range(4))], padded_seq_len=16)  # 2 seqs <= 8 -> ROW_SLOT (9)
    mb_row_large = _make_packed_mb([list(range(8)), list(range(6))], padded_seq_len=16)  # 2 seqs <= 8 -> ROW_SLOT (14)

    for mb, expected_cat in (
        (mb_seg, gdn_slot_schedule.SLOT_CAT_SEGMENTED),
        (mb_single, gdn_slot_schedule.SLOT_CAT_SINGLE_DOC),
        (mb_row_small, gdn_slot_schedule.SLOT_CAT_ROW_SLOT),
        (mb_row_large, gdn_slot_schedule.SLOT_CAT_ROW_SLOT),
    ):
        _, cat, _ = gdn_slot_schedule.classify_gdn_micro_batch(
            mb,
            auto_slot_len=8,
            padded_len=16,
            forward_only=False,
            allow_row_slot=True,
        )
        assert cat == expected_cat

    sorted_mbs, sorted_idx = gdn_slot_schedule.sort_micro_batches_for_slots(
        [mb_seg, mb_single, mb_row_small, mb_row_large],
        ["seg", "single", "row_small", "row_large"],
        auto_slot_len=8,
        static_len=16,
        bucket_size=8,
        forward_only=False,
        allow_row_slot=True,
    )
    assert sorted_idx == ["row_large", "row_small", "single", "seg"]
    assert len(sorted_mbs) == 4


def test_classify_excludes_differentiated_entropy_and_3d_mrope_from_row_slots():
    mb_ent = _make_packed_mb([[0, 1, 2], [0, 1, 2, 3]], padded_seq_len=16, calculate_entropy=True)
    # Training pass (forward_only=False) with entropy requested -> row-slot disallowed.
    mask_train, cat_train, _ = gdn_slot_schedule.classify_gdn_micro_batch(
        mb_ent, auto_slot_len=8, padded_len=16, forward_only=False, allow_row_slot=True
    )
    assert mask_train == 0
    assert cat_train == gdn_slot_schedule.SLOT_CAT_SEGMENTED

    # Forward-only pass (old log-probs) computes chunked entropy without logits -> row-slot allowed.
    mask_fwd, cat_fwd, _ = gdn_slot_schedule.classify_gdn_micro_batch(
        mb_ent, auto_slot_len=8, padded_len=16, forward_only=True, allow_row_slot=True
    )
    assert mask_fwd == 1
    assert cat_fwd == gdn_slot_schedule.SLOT_CAT_ROW_SLOT

    # 3-D MRoPE positions must not be flattened into 1-D slots.
    mb_mrope = _make_packed_mb([[0, 1, 2]], padded_seq_len=8)
    mb_mrope["position_ids"] = torch.nested.as_nested_tensor(
        [torch.zeros((3, 3), dtype=torch.int64)], layout=torch.jagged
    )
    mask_mrope, cat_mrope, _ = gdn_slot_schedule.classify_gdn_micro_batch(
        mb_mrope, auto_slot_len=4, padded_len=8, forward_only=True, allow_row_slot=True
    )
    assert mask_mrope == 0
    assert cat_mrope == gdn_slot_schedule.SLOT_CAT_SEGMENTED


def test_plan_micro_batch_gdn_slots_band_all_reduce_and_refreshes_stale_key():
    mb_row = _make_packed_mb([[0, 1, 2, 3]], padded_seq_len=16, calculate_entropy=True)
    # Forward-only: eligible for both row-slot (8) and single-doc (16) -> picks row-slot 8.
    slots_fwd = gdn_slot_schedule.plan_micro_batch_gdn_slots(
        [mb_row], auto_slot_len=8, forward_only=True, allow_row_slot=True
    )
    assert slots_fwd == [8]
    assert tu.get_non_tensor_data(data=mb_row, key=gdn_slot_schedule.MB_SLOT_LEN_KEY, default=-1) == 8

    # Reusing the same TensorDict on a training pass where a peer rank is segmented (BAND -> 0) must overwrite 8 -> 0.
    def _peer_is_segmented(tensor: torch.Tensor, op: object = None) -> None:
        assert op == dist.ReduceOp.BAND
        tensor.zero_()

    with (
        mock.patch.object(dist, "is_initialized", return_value=True),
        mock.patch.object(dist, "get_world_size", return_value=2),
        mock.patch.object(dist, "all_reduce", side_effect=_peer_is_segmented) as ar,
    ):
        slots_train = gdn_slot_schedule.plan_micro_batch_gdn_slots(
            [mb_row], auto_slot_len=8, forward_only=False, allow_row_slot=True
        )
    ar.assert_called_once()
    assert slots_train == [0]
    assert tu.get_non_tensor_data(data=mb_row, key=gdn_slot_schedule.MB_SLOT_LEN_KEY, default=-1) == 0


def test_build_cp_full_positions_modes_and_validation():
    mb = _make_packed_mb([[0, 1, 2], [0, 1, 2, 3]], padded_seq_len=12)

    # Segmented multi-doc: monotonic tail [4, 5, 6, 7, 8] after last_pos=3.
    seg_pos = gdn_slot_schedule.build_cp_full_positions(mb, 12, unsegmented=False)
    assert seg_pos.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Single-doc unsegmented: zero-padded tail.
    mb_one = _make_packed_mb([[0, 1, 2, 3, 4]], padded_seq_len=8)
    unseg_pos = gdn_slot_schedule.build_cp_full_positions(mb_one, 8, unsegmented=True)
    assert unseg_pos.tolist() == [0, 1, 2, 3, 4, 0, 0, 0]

    # Row-slot permuted: each 6-token row starts with its sequence and pads with zeros.
    cu = mb["input_ids"].offsets()
    ids_pad = torch.zeros((1, 12), dtype=torch.int64)
    pos_pad = torch.zeros((1, 12), dtype=torch.int64)
    lbl_pad = torch.zeros((1, 12), dtype=torch.int64)
    _, _, _, slot_src_cpu, _ = gdn_slot_schedule.permute_packed_for_row_slots(cu, 12, 6, ids_pad, pos_pad, lbl_pad)
    slot_pos = gdn_slot_schedule.build_cp_full_positions(mb, 12, unsegmented=True, slot_src_cpu=slot_src_cpu)
    assert slot_pos.tolist() == [0, 1, 2, 0, 0, 0, 0, 1, 2, 3, 0, 0]

    # Multimodal and 3-D MRoPE raise ValueError instead of silently corrupting positions.
    mb_mm = _make_packed_mb([[0, 1, 2]], padded_seq_len=4)
    mb_mm["multi_modal_inputs"] = [{"pixel_values": torch.zeros(1)}]
    with pytest.raises(ValueError, match="cp_full_positions"):
        gdn_slot_schedule.build_cp_full_positions(mb_mm, 4, unsegmented=False)

    mb_mrope = _make_packed_mb([[0, 1, 2]], padded_seq_len=4)
    mb_mrope["position_ids"] = torch.nested.as_nested_tensor(
        [torch.zeros((3, 3), dtype=torch.int64)], layout=torch.jagged
    )
    with pytest.raises(ValueError, match="cp_full_positions"):
        gdn_slot_schedule.build_cp_full_positions(mb_mrope, 4, unsegmented=False)


def test_enable_fused_gdn_projections_and_model_kwarg_validation():
    @dataclass
    class _GDNCfg:
        fuse_input_projections: bool = False

    class _SlottedLayer:
        __slots__ = ("attention",)

        def __init__(self, attn: object) -> None:
            self.attention = attn

    gdn_a, gdn_b = _GDNCfg(), _GDNCfg()
    root_cfg = SimpleNamespace(layers=[_SlottedLayer(gdn_a), {"linear_attn": gdn_b}, SimpleNamespace(other=1)])
    assert gdn_slot_schedule.enable_fused_gdn_projections(root_cfg) == 2
    assert gdn_a.fuse_input_projections is True
    assert gdn_b.fuse_input_projections is True

    with pytest.raises(RuntimeError, match="torchtitan#216"):
        gdn_slot_schedule.enable_fused_gdn_projections(SimpleNamespace(layers=[SimpleNamespace(other=1)]))

    class _LegacyModel(torch.nn.Module):
        def forward(self, tokens: torch.Tensor, *, positions: torch.Tensor | None = None) -> torch.Tensor:
            return tokens

    class _UpdatedModel(torch.nn.Module):
        def forward(
            self,
            tokens: torch.Tensor,
            *,
            positions: torch.Tensor | None = None,
            cp_full_positions: torch.Tensor | None = None,
            gated_delta_document_resets: bool | None = None,
        ) -> torch.Tensor:
            return tokens

    with pytest.raises(RuntimeError, match="cp_full_positions"):
        gdn_slot_schedule.validate_model_cp_slot_support(
            [_LegacyModel()], need_cp_full_positions=True, need_document_resets_override=False
        )
    with pytest.raises(RuntimeError, match="gated_delta_document_resets"):
        gdn_slot_schedule.validate_model_cp_slot_support(
            [_LegacyModel()], need_cp_full_positions=False, need_document_resets_override=True
        )
    assert gdn_slot_schedule.validate_model_cp_slot_support(
        [_UpdatedModel()], need_cp_full_positions=True, need_document_resets_override=True
    ) == (True, True)


@pytest.mark.parametrize("cp_enabled", [False, True])
@pytest.mark.parametrize("calculate_entropy", [False, True])
def test_engine_prepare_inputs_and_outputs_row_slot_roundtrip_matches_baseline(
    cp_enabled: bool, calculate_entropy: bool
):
    """Row-slot permutation + optional 2-way CP sharding produces identical jagged log_probs and entropy."""
    torch.manual_seed(7)
    vocab, hidden_dim, bucket_size, slot_len = 24, 8, 16, 8
    seqs_pos = [[0, 1, 2, 3, 4], [0, 1, 2, 3]]  # 9 tokens -> padded_seq_len=16, 2 slots of 8

    # Token-wise deterministic embedding table so a position-independent token model gives the exact same hidden
    # representation per token regardless of where the token sits in the padded/permuted/CP-sharded buffer.
    token_embed = torch.randn(vocab, hidden_dim)
    lm_head_weight_t = torch.randn(hidden_dim, vocab)

    def _make_stub_engine(*, cp: bool, cp_full_pos: bool) -> torchtitan_tpu.TorchTitanTPUEngineWithLMHead:
        eng = object.__new__(torchtitan_tpu.TorchTitanTPUEngineWithLMHead)
        eng.parallel_dims = SimpleNamespace(
            tp_enabled=False,
            cp_enabled=cp,
            get_mesh=lambda name: "cp_mesh",
        )
        eng.model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=vocab))
        eng.engine_config = SimpleNamespace(
            max_token_len_per_gpu=bucket_size,
            entropy_checkpointing=False,
            entropy_from_logits_with_chunking=False,
            engine_kwargs={"tpu": {"seq_bucket_size": bucket_size}},
        )
        eng.trainer = SimpleNamespace(
            config=SimpleNamespace(
                parallelism=SimpleNamespace(
                    context_parallel_load_balancer=None,
                    context_parallel_ptrr_mask_key=None,
                )
            )
        )
        eng._gdn_row_slot_len = 0
        eng._cp_full_positions = cp_full_pos
        eng._tpu_kernels = frozenset()
        eng.compute_entropy_from_logits = verl_F.entropy_from_logits
        return eng

    def _exact_logprobs(hidden_2d: torch.Tensor, labels_1d: torch.Tensor, weight_t: torch.Tensor, **kwargs):
        temperature = float(kwargs.get("temperature", 1.0))
        logits = (hidden_2d @ weight_t).float() / temperature
        log_probs_all = torch.log_softmax(logits, dim=-1)
        safe_labels = labels_1d.clamp(0, weight_t.shape[1] - 1)
        return log_probs_all.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    # 1. Baseline: no row slots, no CP.
    mb_base = _make_packed_mb(seqs_pos, vocab=vocab, padded_seq_len=16, calculate_entropy=calculate_entropy, seed=11)
    base_engine = _make_stub_engine(cp=False, cp_full_pos=False)
    with (
        mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")),
        mock.patch.object(torchtitan_tpu, "fused_linear_logprobs_tpu", side_effect=_exact_logprobs),
        torch.no_grad(),
    ):
        ids_b, extra_in_b, extra_kw_b, out_args_b = base_engine.prepare_model_inputs(mb_base)
        assert "cp_full_positions" not in extra_in_b
        assert extra_kw_b == {}
        hidden_b = token_embed[ids_b.clamp(0, vocab - 1)]
        out_base = base_engine.prepare_model_outputs((hidden_b, lm_head_weight_t), out_args_b, mb_base)

    # 2. Row-slot scheduled path (with optional 2-way contiguous CP sharding + cp_full_positions).
    mb_slot = _make_packed_mb(seqs_pos, vocab=vocab, padded_seq_len=16, calculate_entropy=calculate_entropy, seed=11)
    slots = gdn_slot_schedule.plan_micro_batch_gdn_slots(
        [mb_slot], auto_slot_len=slot_len, forward_only=True, allow_row_slot=True
    )
    assert slots == [slot_len]

    slot_engine = _make_stub_engine(cp=cp_enabled, cp_full_pos=cp_enabled)
    rank_outputs: list[dict[str, torch.Tensor]] = []
    cp_world = 2 if cp_enabled else 1
    peer_Hidden_logprobs: dict[str, list[torch.Tensor]] = {"lp": [], "ent": []}

    # First run prepare_model_inputs + local forward on each simulated CP rank.
    per_rank_state = []
    for cp_rank in range(cp_world):
        if cp_enabled:

            def _cp_shard(batch, *_args, _r=cp_rank):
                half = batch["input"].shape[-1] // 2
                sl = slice(_r * half, (_r + 1) * half)
                return {
                    "input": batch["input"][..., sl],
                    "positions": batch["positions"][..., sl],
                    "labels": batch["labels"][..., sl],
                    "attention_masks": f"mask_rank_{_r}",
                }

            slot_engine._cp_attention_owner = lambda _fn=_cp_shard: SimpleNamespace(cp_shard=_fn)

        with (
            mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")),
            torch.no_grad(),
        ):
            ids_r, extra_in_r, extra_kw_r, out_args_r = slot_engine.prepare_model_inputs(mb_slot)

        if calculate_entropy and cp_rank == 0:
            # With grad enabled, differentiated entropy must safely bypass row-slot permutation even if MB_SLOT_LEN_KEY
            # was left on the micro-batch.
            with mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")):
                _, _, extra_kw_grad, out_args_grad = slot_engine.prepare_model_inputs(mb_slot)
            assert "gated_delta_document_resets" not in extra_kw_grad
            assert gdn_slot_schedule.POST_GATHER_CONTIG_SRC_KEY not in out_args_grad

        assert extra_kw_r.get("gated_delta_document_resets") is False
        if cp_enabled:
            assert extra_kw_r.get("attention_masks") == f"mask_rank_{cp_rank}"
            assert extra_in_r["cp_full_positions"].tolist() == [0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0]
        else:
            assert "cp_full_positions" not in extra_in_r

        hidden_r = token_embed[ids_r.clamp(0, vocab - 1)]
        lbl_1d = out_args_r["labels"].squeeze(0)
        peer_Hidden_logprobs["lp"].append(_exact_logprobs(hidden_r.squeeze(0), lbl_1d, lm_head_weight_t))
        if calculate_entropy:
            peer_Hidden_logprobs["ent"].append(
                torchtitan_tpu.chunked_entropy_from_hidden_tpu(hidden_r.squeeze(0), lm_head_weight_t, 1.0)
            )
        per_rank_state.append((hidden_r, out_args_r))

    for _cp_rank, (hidden_r, out_args_r) in enumerate(per_rank_state):
        if cp_enabled:
            call_idx = {"n": 0}

            def _gather_cp(local_tensor: torch.Tensor, padded_len: int, _state=call_idx) -> torch.Tensor:
                key = "ent" if (calculate_entropy and _state["n"] == 0) else "lp"
                _state["n"] += 1
                full = torch.cat(peer_Hidden_logprobs[key], dim=0)
                assert full.numel() == padded_len
                return full

            slot_engine._gather_context_parallel = _gather_cp

        with (
            mock.patch.object(torchtitan_tpu, "fused_linear_logprobs_tpu", side_effect=_exact_logprobs),
            torch.no_grad(),
        ):
            out_r = slot_engine.prepare_model_outputs((hidden_r, lm_head_weight_t), out_args_r, mb_slot)
        rank_outputs.append(out_r)

    for out_r in rank_outputs:
        torch.testing.assert_close(out_r["log_probs"].values(), out_base["log_probs"].values(), atol=1e-6, rtol=1e-6)
        if calculate_entropy:
            torch.testing.assert_close(out_r["entropy"].values(), out_base["entropy"].values(), atol=1e-6, rtol=1e-6)

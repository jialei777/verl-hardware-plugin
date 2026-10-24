# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Optional parity check of the ``tpu-sync`` weight sync (``engine_kwargs.tpu-sync.verify_parity``, off by default).

After every sync the trainer posts the norms of the weights it sent (``post_trainer_stats``) and the driver compares
them with the norms of the weights every rollout replica received (``verify_rollout_parity``). It is a bring-up and
debugging aid; the sync itself does not depend on it.
"""

import asyncio
import logging
import os
from typing import Any, Optional

from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_rendezvous import get_tpu_sync_rendezvous

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_tensor_stats(items: list) -> dict:
    """Compute deterministic L1/L2 norms and parameter counts across tensors.

    Reductions run on the device holding the tensors, to avoid copying the weights to the host.
    ``items`` holds ``(name, tensor)`` pairs (named tensors get a ``per_tensor`` entry) or bare tensors.
    """
    total_numel = 0
    total_l1 = 0.0
    total_l2_sq = 0.0
    per_tensor = {}

    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            name, p = item
        else:
            name = None
            p = item
        p_local = p.to_local() if hasattr(p, "to_local") else p

        # Reduce in float32 for precision.
        p_float = p_local.float()
        t_l1 = float(p_float.abs().sum().item())
        t_l2_sq = float(p_float.pow(2).sum().item())
        numel = p_local.numel()

        if name is not None:
            per_tensor[name] = {
                "l1": t_l1,
                "l2": float(t_l2_sq**0.5),
                "l2_sq": t_l2_sq,
                "numel": numel,
                "shape": list(p_local.shape),
                "dtype": str(p_local.dtype),
            }
        total_numel += numel
        total_l1 += t_l1
        total_l2_sq += t_l2_sq

    return {
        "total_numel": total_numel,
        "num_tensors": len(items),
        "l1_norm": total_l1,
        "l2_norm": float(total_l2_sq**0.5),
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


def merge_rank_stats(rank_stats: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    """Sums the per-tensor partial stats (``l1``, ``l2_sq``, ``numel``) that every trainer rank posted.

    When the trainer sends local shards, each rank posts the stats of its own shards (and rank 0 also those of the
    full tensors), so the sum over ranks describes the whole model once, in the form ``compute_tensor_stats``
    returns for full tensors.
    """
    per_tensor: dict[str, dict[str, Any]] = {}
    for stats in rank_stats.values():
        for name, t in stats.get("per_tensor", {}).items():
            acc = per_tensor.setdefault(name, {"l1": 0.0, "l2_sq": 0.0, "numel": 0})
            acc["l1"] += t["l1"]
            acc["l2_sq"] += t["l2_sq"]
            acc["numel"] += t["numel"]
    for acc in per_tensor.values():
        acc["l2"] = acc["l2_sq"] ** 0.5
    total_l2_sq = sum(t["l2_sq"] for t in per_tensor.values())
    return {
        "total_numel": sum(t["numel"] for t in per_tensor.values()),
        "num_tensors": len(per_tensor),
        "l1_norm": sum(t["l1"] for t in per_tensor.values()),
        "l2_norm": total_l2_sq**0.5,
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


def post_trainer_stats(
    rendezvous: Any,
    step: int,
    rank: int,
    is_master: bool,
    named_tensors: list,
    shard_info: dict[str, tuple[int, int]],
) -> None:
    """Posts the trainer side of the parity check for ``step`` to the job's rendezvous.

    With local shards (``shard_info`` non-empty) every rank posts the stats of its own shards, and the full tensors
    (the same on every rank) are counted by rank 0 only, so the sum over ranks counts each element once. Otherwise
    every rank holds the full model and rank 0's stats describe it.
    """
    try:
        if shard_info:
            items = [(n, t) for n, t in named_tensors if n in shard_info or is_master]
            stats = compute_tensor_stats(items)
            stats["rank"] = rank
            rendezvous.set_rank_stats.remote(step, rank, stats)
        elif is_master:
            stats = compute_tensor_stats(named_tensors)
            stats["rank"] = rank
            rendezvous.set_stats.remote(step, stats)
        else:
            return
        logger.info(
            f"[RAIDEN PARITY] Successfully stored trainer rank {rank} stats for step {step}: "
            f"L1={stats['l1_norm']:.4f}, numel={stats['total_numel']}"
        )
    except Exception as e:
        logger.warning(f"Failed to record trainer rank {rank} stats for the parity check: {e}")


async def verify_rollout_parity(manager: Any, global_steps: Optional[int], replica_labels: list[str]) -> None:
    """Compare distributed norms between the trainer and the Sampler TP workers.

    The trainer side is rank 0's stats, or the sum of every rank's stats when the ranks sent local shards.
    ``replica_labels`` names each of ``manager.replicas`` in the result lines (empty strings with one replica).

    TODO(tpu): Refactor parity verification to use rank-local scalar partitioning.
    Instead of collecting per-tensor dictionaries across all Sampler workers and checking
    replicated vs sharded heuristics over 200+ tensors, each worker can reduce its model
    locally into scalar metrics (numel, l1, l2_sq) before RPC return:
      - Rank 0 accumulates both sharded tensors and replicated 1D tensors (e.g. RMSNorms).
      - Ranks 1..N-1 accumulate only sharded tensors.
    The orchestrator can then perform verification in O(ranks) pure scalar arithmetic
    rather than O(tensors * ranks) loop aggregation.
    """
    step_key = global_steps if global_steps is not None else 0
    if step_key <= 0:
        return

    try:
        rendezvous = get_tpu_sync_rendezvous()
        # With local shards every trainer rank posts its own stats ("ranks"); wait until all of them have.
        num_trainer_ranks = manager.actor_wg.world_size
        trainer_entry = None
        for _ in range(25):
            entry = await rendezvous.get_stats.remote(step_key)
            if entry is not None and ("ranks" not in entry or len(entry["ranks"]) >= num_trainer_ranks):
                trainer_entry = entry
                break
            await asyncio.sleep(0.5)

        sampler_entries = await asyncio.gather(
            *[
                replica.server_handle.collective_rpc.remote(method="get_model_weights_stats")
                for replica in manager.replicas
            ]
        )

        if not trainer_entry or not sampler_entries:
            logger.warning(f"[RAIDEN PARITY] Incomplete stats data for step {step_key}")
            return

        if "ranks" in trainer_entry:
            # Local shards: every trainer rank posted the stats of its own shards; their sum describes the model.
            trainer_entry = {"master": merge_rank_stats(trainer_entry["ranks"])}
        # The norms of a replica's workers are summed before the comparison with the trainer, which describes
        # one copy of the model. Every replica is a full copy, so each one is compared on its own, and the
        # result lines name the replica when there are several.
        for label, entry in zip(replica_labels, sampler_entries, strict=True):
            _check_parity_norms(trainer_entry, entry, step_key, label)
    except Exception as e:
        logger.warning(f"Error during parity verification for step {step_key}: {e}")


def _check_parity_norms(trainer_entry: Any, sampler_entry: Any, step_key: int, label: str = "") -> None:
    """Compares the trainer's norms with the summed norms of one replica's workers (``sampler_entry``, the
    rollout server's ``collective_rpc`` result); ``label`` is appended to the step in the log lines."""
    # The server returns one result per TP worker (a list), or a single dict.
    results = sampler_entry if isinstance(sampler_entry, list | tuple) else [sampler_entry]
    sampler_workers = [w for w in results if isinstance(w, dict)]
    if not sampler_workers:
        logger.warning(
            f"[RAIDEN PARITY] No sampler worker stats for step {step_key}{label}: the rollout server's "
            "collective_rpc must return the worker results for the parity check."
        )
        return

    trainer_master = trainer_entry.get("master", trainer_entry)
    trainer_per_tensor = trainer_master.get("per_tensor", {})
    # Gets master list of all model tensor names (e.g., "model.layers.0.self_attn.qkv_proj.weight").
    all_param_names = list(sampler_workers[0].get("per_tensor", {}).keys()) if sampler_workers else []

    total_trainer_numel = trainer_master.get("total_numel", 0)
    total_trainer_l1 = trainer_master.get("l1_norm", 0.0)
    global_trainer_l2 = trainer_master.get("l2_norm", 0.0)

    total_sampler_l1, total_sampler_l2_sq, total_sampler_numel = 0.0, 0.0, 0
    mismatches = []
    # Number of TP ranks holding each tensor's slice (vLLMTPUSyncWorkerExtension._raiden_replicas): 1 for a
    # plain split, tp_size // num_kv_heads for GQA k/v with replicated KV heads, tp_size for unsharded tensors.
    replicas = sampler_workers[0].get("replicas") or {}

    for name in all_param_names:
        s_numels = [w.get("per_tensor", {}).get(name, {}).get("numel", 0) for w in sampler_workers]
        s_l1s = [w.get("per_tensor", {}).get(name, {}).get("l1", 0.0) for w in sampler_workers]
        s_l2_sqs = [
            w.get("per_tensor", {})
            .get(name, {})
            .get("l2_sq", w.get("per_tensor", {}).get(name, {}).get("l2", 0.0) ** 2)
            for w in sampler_workers
        ]

        if name in replicas:
            # Every slice is held by `r` ranks: summing over ranks counts it r times.
            r = replicas[name]
            s_agg_numel = sum(s_numels) // r
            s_agg_l1 = sum(s_l1s) / r
            s_agg_l2 = (sum(s_l2_sqs) / r) ** 0.5
        elif (
            len(set(s_numels)) == 1
            and (name.endswith("layernorm.weight") or "norm" in name)
            and len(s_numels[0:1]) > 0
            and s_numels[0] < 10000
        ):
            s_agg_numel = s_numels[0]
            s_agg_l1 = s_l1s[0]
            s_agg_l2 = s_l2_sqs[0] ** 0.5
        else:
            s_agg_numel = sum(s_numels)
            s_agg_l1 = sum(s_l1s)
            s_agg_l2 = sum(s_l2_sqs) ** 0.5

        t_data = trainer_per_tensor.get(name, {})
        t_agg_numel = t_data.get("numel", 0)
        t_agg_l1 = t_data.get("l1", 0.0)

        total_sampler_numel += s_agg_numel
        total_sampler_l1 += s_agg_l1
        total_sampler_l2_sq += s_agg_l2**2

        delta_numel = abs(s_agg_numel - t_agg_numel)
        delta_l1 = abs(s_agg_l1 - t_agg_l1)
        rel_tol = 1e-3 * max(abs(t_agg_l1), 1.0)

        if delta_numel != 0 or delta_l1 > rel_tol:
            mismatches.append(f"  * {name}: Trainer(L1={t_agg_l1:.4f}) vs Sampler(L1={s_agg_l1:.4f})")

    global_sampler_l2 = total_sampler_l2_sq**0.5
    total_l1_delta = abs(total_sampler_l1 - total_trainer_l1)
    total_l2_delta = abs(global_sampler_l2 - global_trainer_l2)
    total_numel_delta = abs(total_sampler_numel - total_trainer_numel)
    total_rel_tol = 1e-3 * max(abs(total_trainer_l1), 1.0)

    if not mismatches and total_numel_delta == 0 and total_l1_delta <= total_rel_tol:
        logger.info(
            f"[RAIDEN PARITY VERIFIED | Step {step_key}{label}] 100% DISTRIBUTED NORM PARITY CONFIRMED!\n"
            f"  * Global L1 Norm: {total_sampler_l1:.6f} "
            f"(Trainer={total_trainer_l1:.6f}, delta={total_l1_delta:.6f})\n"
            f"  * Global L2 Norm: {global_sampler_l2:.6f} "
            f"(Trainer={global_trainer_l2:.6f}, delta={total_l2_delta:.6f})\n"
            f"  * Total Parameters: {total_sampler_numel} across {len(all_param_names)} tensors"
        )
    else:
        mismatches_summary = "\n".join(mismatches[:10])
        logger.error(
            f"[RAIDEN PARITY MISMATCH | Step {step_key}{label}] Norms do NOT match!\n"
            f"  * Trainer: numel={total_trainer_numel}, L1={total_trainer_l1:.6f}, L2={global_trainer_l2:.6f}\n"
            f"  * Sampler: numel={total_sampler_numel}, L1={total_sampler_l1:.6f}, L2={global_sampler_l2:.6f}\n"
            f"  * Mismatched Tensors ({len(mismatches)} / {len(all_param_names)}):\n"
            f"{mismatches_summary}"
        )

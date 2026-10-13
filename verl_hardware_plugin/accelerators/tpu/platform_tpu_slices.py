# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU slice discovery and trainer / rollout placement planning.

A GKE Ray cluster with several TPU slices advertises one custom resource per slice,
``tpu-group-<k>`` (1.0 on every host of slice ``k``), next to ``TPU`` (chips per host).
Single-host slices started without that resource are addressed as ``node:<ip>``.

This module decides which slices the trainer and the rollout (sampler) use and on which
hosts every rollout replica runs. Two user-facing ways of asking for several rollout
replicas (data parallelism across replicas) are supported and lead to the same plan:

* ``actor_rollout_ref.rollout.data_parallel_size=D`` with
  ``tensor_model_parallel_size=T``: ``D`` replicas of ``T`` chips each, placed on the rollout
  slices; ``rollout.nnodes * rollout.n_gpus_per_node`` must equal ``D * T``.
* ``actor_rollout_ref.rollout.tpu_slices=[1, 2]`` (slice indices, ``tpu-group-*`` names or
  ``node:<ip>``) with ``tensor_model_parallel_size=T``: ``rollout.nnodes`` /
  ``n_gpus_per_node`` are filled in from the slices and ``D = chips / T`` is derived.

Trainer slices default to the first slice not used by the rollout and can be set with
``trainer.tpu_slices`` or ``VERL_TPU_TRAINER_SLICES``; ``VERL_TPU_ROLLOUT_SLICES`` is the
environment equivalent of ``rollout.tpu_slices``.

A replica always occupies whole hosts of a single slice (libtpu rejects multi-chip ICI
sessions smaller than a host, and a replica spanning two slices has no ICI between them),
so replicas are packed slice by slice: with ``tpu-group-1`` and ``tpu-group-2`` of two hosts
each and ``T`` = chips per host, replicas 0/1 take the hosts of ``tpu-group-1`` and 2/3 those
of ``tpu-group-2``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TPU_SLICE_RESOURCE_PREFIX = "tpu-group-"
TRAINER_SLICES_ENV = "VERL_TPU_TRAINER_SLICES"
ROLLOUT_SLICES_ENV = "VERL_TPU_ROLLOUT_SLICES"

# Resource pools whose placement is planned like a rollout replica: ``rollout_pool_<idx>``.
# Reward / teacher pools (``rollout_pool_reward_<idx>``) keep the slice round-robin.
_ROLLOUT_POOL_RE = re.compile(r"^rollout_pool_(\d+)")
_ROLLOUT_LIKE_RE = re.compile(r"(?:rollout_pool(?:_reward|_teacher)?_)(\d+)")


@dataclass
class TPUSlice:
    """One TPU slice of the Ray cluster.

    Attributes:
        name: Ray resource that selects the slice, ``tpu-group-<k>`` or ``node:<ip>``.
        hosts: Node IPs of the slice, sorted.
        chips_per_host: ``TPU`` resource of each host.
    """

    name: str
    hosts: list[str]
    chips_per_host: int

    @property
    def chips(self) -> int:
        return len(self.hosts) * self.chips_per_host

    @property
    def index(self) -> Optional[int]:
        """``k`` of ``tpu-group-<k>``; None for ``node:<ip>`` slices."""
        suffix = self.name[len(TPU_SLICE_RESOURCE_PREFIX) :] if self.name.startswith(TPU_SLICE_RESOURCE_PREFIX) else ""
        return int(suffix) if suffix.isdigit() else None


def _slice_sort_key(name: str) -> tuple:
    suffix = name.rsplit("-", 1)[-1]
    return (0, int(suffix)) if name.startswith(TPU_SLICE_RESOURCE_PREFIX) and suffix.isdigit() else (1, name)


def discover_tpu_slices(nodes: Optional[list[dict]] = None) -> list[TPUSlice]:
    """List the TPU slices of the Ray cluster, ``tpu-group-*`` first in index order.

    Args:
        nodes: ``ray.nodes()`` entries; queried from Ray when omitted (empty if Ray is not up).
    """
    if nodes is None:
        try:
            import ray

            nodes = ray.nodes() if ray.is_initialized() else []
        except Exception as e:  # pragma: no cover - defensive, Ray import/state errors
            logger.warning(f"Could not query Ray nodes for TPU slices: {e}")
            nodes = []

    hosts_by_slice: dict[str, list[tuple[tuple, str, int]]] = {}
    for node in nodes:
        resources = node.get("Resources", {}) or {}
        if not node.get("Alive", False) or "TPU" not in resources:
            continue
        ip = node.get("NodeManagerAddress") or ""
        chips = int(resources.get("TPU", 0))
        groups = [r for r in resources if r.startswith(TPU_SLICE_RESOURCE_PREFIX)]
        if not ip and not groups:
            continue
        # Single-host slices (numOfHosts=1) omit tpu-group-* resources; address them by node:<ip>.
        names = groups or [f"node:{ip}"]
        # KubeRay labels each host with its index in the slice topology; keep that order (host 0 first)
        # and fall back to the IP when the label is absent.
        worker_id = str((node.get("Labels") or {}).get("ray.io/tpu-worker-id", ""))
        order = (0, int(worker_id), ip) if worker_id.isdigit() else (1, 0, ip)
        for name in names:
            hosts_by_slice.setdefault(name, []).append((order, ip, chips))

    slices = []
    for name in sorted(hosts_by_slice, key=_slice_sort_key):
        members = sorted(hosts_by_slice[name])
        chips_per_host = {c for _, _, c in members}
        if len(chips_per_host) != 1:
            raise ValueError(f"TPU slice {name} has hosts with different chip counts: {members}")
        slices.append(TPUSlice(name=name, hosts=[ip for _, ip, _ in members], chips_per_host=chips_per_host.pop()))
    return slices


def parse_slice_spec(spec: Any) -> Optional[list[str]]:
    """Normalize a slice list given as list / comma-separated string of indices or resource names.

    ``[1, 2]``, ``"1,2"``, ``["tpu-group-1", "tpu-group-2"]`` and ``"node:10.0.0.5"`` are all
    accepted; indices become ``tpu-group-<k>``. Returns None for an empty spec.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        items: list[Any] = [s.strip() for s in spec.split(",") if s.strip()]
    elif isinstance(spec, int):
        items = [spec]
    else:
        items = list(spec)
    names = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if text.isdigit():
            names.append(f"{TPU_SLICE_RESOURCE_PREFIX}{int(text)}")
        else:
            names.append(text)
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate TPU slice in {spec!r}")
    return names or None


def resolve_slices(spec: Any, slices: list[TPUSlice], what: str) -> Optional[list[TPUSlice]]:
    """Map a slice spec onto discovered slices, failing loudly on unknown names."""
    names = parse_slice_spec(spec)
    if names is None:
        return None
    by_name = {s.name: s for s in slices}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(
            f"{what} refers to TPU slice(s) {missing} that are not in the Ray cluster. "
            f"Available: {[s.name for s in slices]}"
        )
    return [by_name[n] for n in names]


@dataclass
class TPUSlicePlan:
    """Which slices the trainer and the rollout use."""

    slices: list[TPUSlice]
    trainer: list[TPUSlice]
    rollout: list[TPUSlice]
    explicit: bool = False
    """Whether the user chose the slices (config or env) rather than the first/rest default."""

    rollout_hosts: list[tuple[TPUSlice, str]] = field(init=False)

    def __post_init__(self):
        self.rollout_hosts = [(s, ip) for s in self.rollout for ip in s.hosts]

    @property
    def rollout_chips(self) -> int:
        return sum(s.chips for s in self.rollout)

    def replica_hosts(self, replica_idx: int, nnodes: int, chips_per_node: int) -> tuple[TPUSlice, list[str]]:
        """Hosts of rollout replica ``replica_idx`` (``nnodes`` whole hosts of one slice).

        Replicas are packed slice by slice in slice order; a replica never straddles slices.
        """
        offset = 0
        for s in self.rollout:
            if s.chips_per_host != chips_per_node:
                raise ValueError(
                    f"Rollout replicas use {chips_per_node} chips per host but TPU slice {s.name} has "
                    f"{s.chips_per_host} chips per host. On TPU a replica occupies whole hosts: set "
                    f"actor_rollout_ref.rollout.n_gpus_per_node={s.chips_per_host} and a "
                    f"tensor_model_parallel_size that is a multiple of {s.chips_per_host}."
                )
            per_slice = len(s.hosts) // nnodes
            if offset <= replica_idx < offset + per_slice:
                start = (replica_idx - offset) * nnodes
                return s, s.hosts[start : start + nnodes]
            offset += per_slice
        raise ValueError(
            f"Rollout replica {replica_idx} ({nnodes} host(s) x {chips_per_node} chips) does not fit on the "
            f"rollout TPU slices {[s.name for s in self.rollout]} "
            f"({[(s.name, len(s.hosts)) for s in self.rollout]} hosts); they hold {offset} such replica(s). "
            "Reduce actor_rollout_ref.rollout.data_parallel_size / nnodes or give the rollout more slices "
            "(actor_rollout_ref.rollout.tpu_slices)."
        )


_PLAN: Optional[TPUSlicePlan] = None


def set_tpu_slice_plan(plan: Optional[TPUSlicePlan]) -> None:
    """Install the slice plan for this process (the trainer driver creates every resource pool)."""
    global _PLAN
    _PLAN = plan


def get_tpu_slice_plan() -> Optional[TPUSlicePlan]:
    return _PLAN


def build_tpu_slice_plan(
    trainer_spec: Any = None,
    rollout_spec: Any = None,
    slices: Optional[list[TPUSlice]] = None,
) -> Optional[TPUSlicePlan]:
    """Resolve trainer / rollout slices from the specs, the environment, or the first/rest default.

    Returns None when the cluster has no TPU slices (e.g. Ray not initialized).
    """
    if slices is None:
        slices = discover_tpu_slices()
    if not slices:
        if trainer_spec or rollout_spec:
            raise ValueError("TPU slices were requested but no TPU nodes are visible in the Ray cluster.")
        return None

    if trainer_spec in (None, "", []):
        trainer_spec = os.environ.get(TRAINER_SLICES_ENV) or None
    if rollout_spec in (None, "", []):
        rollout_spec = os.environ.get(ROLLOUT_SLICES_ENV) or None

    trainer = resolve_slices(trainer_spec, slices, "trainer.tpu_slices")
    rollout = resolve_slices(rollout_spec, slices, "actor_rollout_ref.rollout.tpu_slices")
    explicit = trainer is not None or rollout is not None

    if trainer is None and rollout is None:
        trainer, rollout = [slices[0]], slices[1:]
    elif trainer is None:
        assert rollout is not None
        rest = [s for s in slices if s not in rollout]
        if not rest:
            raise ValueError(
                f"actor_rollout_ref.rollout.tpu_slices={[s.name for s in rollout]} leaves no TPU slice for the "
                f"trainer (cluster slices: {[s.name for s in slices]})."
            )
        trainer = rest[:1]
    elif rollout is None:
        rollout = [s for s in slices if s not in trainer]

    assert rollout is not None
    overlap = [s.name for s in trainer if s in rollout]
    if overlap:
        raise ValueError(f"TPU slice(s) {overlap} are assigned to both the trainer and the rollout.")
    if len(trainer) > 1:
        logger.warning(
            f"trainer.tpu_slices lists {[s.name for s in trainer]}; the trainer runs on a single slice "
            f"({trainer[0].name}) because slices have no ICI between them."
        )
    return TPUSlicePlan(slices=slices, trainer=trainer, rollout=rollout, explicit=explicit)


def resolve_tpu_rollout_layout(config) -> Optional[dict]:
    """Fill rollout nnodes / n_gpus_per_node from ``tpu_slices`` and validate the replica count.

    Called by the trainer before it reads ``actor_rollout_ref.rollout.nnodes``. ``config`` is the
    full (OmegaConf) trainer config and is updated in place. Installs the slice plan used by
    ``PlatformTPU`` when the resource pools are created. Returns a summary dict (or None when the
    cluster has no TPU slices and nothing was requested).
    """
    from omegaconf import DictConfig, open_dict

    rollout = config.actor_rollout_ref.rollout
    trainer_cfg = config.trainer
    rollout_spec = rollout.get("tpu_slices", None)
    trainer_spec = trainer_cfg.get("tpu_slices", None) if hasattr(trainer_cfg, "get") else None

    plan = build_tpu_slice_plan(trainer_spec=trainer_spec, rollout_spec=rollout_spec)
    set_tpu_slice_plan(plan)
    if plan is None:
        return None

    tp = int(rollout.tensor_model_parallel_size)
    pp = int(rollout.get("pipeline_model_parallel_size", 1) or 1)
    dp = int(rollout.get("data_parallel_size", 1) or 1)
    replica_chips = tp * pp

    nnodes = int(rollout.nnodes or 0)
    n_gpus_per_node = int(rollout.n_gpus_per_node or 0)
    if plan.explicit and plan.rollout:
        chips_per_host = {s.chips_per_host for s in plan.rollout}
        if len(chips_per_host) != 1:
            raise ValueError(
                f"Rollout TPU slices {[s.name for s in plan.rollout]} have different chips per host "
                f"({sorted(chips_per_host)}); use slices of the same host shape."
            )
        slice_nnodes = sum(len(s.hosts) for s in plan.rollout)
        slice_cph = chips_per_host.pop()
        if (nnodes and nnodes != slice_nnodes) or (n_gpus_per_node and n_gpus_per_node != slice_cph):
            logger.warning(
                f"actor_rollout_ref.rollout.nnodes={nnodes} n_gpus_per_node={n_gpus_per_node} replaced by the "
                f"rollout TPU slices {[s.name for s in plan.rollout]}: nnodes={slice_nnodes}, "
                f"n_gpus_per_node={slice_cph}"
            )
        nnodes, n_gpus_per_node = slice_nnodes, slice_cph
        if isinstance(rollout, DictConfig):
            with open_dict(rollout):
                rollout.nnodes = nnodes
                rollout.n_gpus_per_node = n_gpus_per_node
        else:
            rollout.nnodes = nnodes
            rollout.n_gpus_per_node = n_gpus_per_node

    total_chips = nnodes * n_gpus_per_node
    if total_chips <= 0:
        return {"plan": plan, "num_replicas": 0}

    if dp > 1 and dp * replica_chips != total_chips:
        raise ValueError(
            f"actor_rollout_ref.rollout.data_parallel_size={dp} x tensor_model_parallel_size={tp}"
            f"{f' x pipeline_model_parallel_size={pp}' if pp > 1 else ''} needs {dp * replica_chips} rollout "
            f"chips, but the rollout has nnodes={nnodes} x n_gpus_per_node={n_gpus_per_node} = {total_chips}. "
            "On TPU data_parallel_size is the number of vLLM replicas: either give the rollout "
            f"{dp * replica_chips} chips (nnodes / n_gpus_per_node or actor_rollout_ref.rollout.tpu_slices) "
            f"or set data_parallel_size={total_chips // replica_chips if total_chips % replica_chips == 0 else 1}."
        )
    if total_chips % replica_chips:
        raise ValueError(
            f"Rollout chips nnodes={nnodes} x n_gpus_per_node={n_gpus_per_node} = {total_chips} is not a multiple "
            f"of the replica size tensor_model_parallel_size={tp}{f' x pp={pp}' if pp > 1 else ''}."
        )
    if replica_chips % n_gpus_per_node:
        raise ValueError(
            f"tensor_model_parallel_size={tp}{f' x pp={pp}' if pp > 1 else ''} chips per replica is not a multiple "
            f"of n_gpus_per_node={n_gpus_per_node}: on TPU a replica occupies whole hosts, replicas sharing a "
            "host are not supported."
        )
    num_replicas = total_chips // replica_chips
    if plan.rollout:
        # Make sure the replicas fit on the planned slices now, with a readable error instead of a hang.
        for idx in range(num_replicas):
            plan.replica_hosts(idx, replica_chips // n_gpus_per_node, n_gpus_per_node)

    summary = {
        "plan": plan,
        "num_replicas": num_replicas,
        "tensor_parallel_size": tp,
        "nnodes": nnodes,
        "n_gpus_per_node": n_gpus_per_node,
    }
    logger.info(
        "[TPU slices] trainer=%s rollout=%s -> %d rollout replica(s) x TP=%d (nnodes=%d, n_gpus_per_node=%d)",
        [s.name for s in plan.trainer],
        [s.name for s in plan.rollout],
        num_replicas,
        tp,
        nnodes,
        n_gpus_per_node,
    )
    print(
        f"[TPU slices] trainer={[s.name for s in plan.trainer]} rollout={[s.name for s in plan.rollout]} -> "
        f"{num_replicas} rollout replica(s) x TP={tp} (nnodes={nnodes}, n_gpus_per_node={n_gpus_per_node})",
        flush=True,
    )
    return summary


def plan_resource_pool_placement(
    name_prefix: str, process_on_nodes: Optional[list[int]] = None
) -> tuple[Optional[str], Optional[list[str]]]:
    """Slice resource and host pins for a resource pool named ``name_prefix``.

    Returns ``(accelerator_type, hosts)``: ``accelerator_type`` is the slice resource added to
    every placement-group bundle (``tpu-group-<k>`` or ``node:<ip>``), ``hosts`` the node IP of
    each placement group (one per entry of ``process_on_nodes``) for rollout replicas, or None
    when the pool is not pinned to hosts.
    """
    plan = get_tpu_slice_plan()
    if plan is None:
        try:
            plan = build_tpu_slice_plan()
        except ValueError as e:
            logger.warning(f"TPU slice plan unavailable for {name_prefix}: {e}")
            plan = None
        if plan is None:
            return None, None

    prefix_lower = (name_prefix or "").lower()
    is_rollout_like = any(k in prefix_lower for k in ("rollout", "reward", "teacher"))
    if not is_rollout_like or not plan.rollout:
        return plan.trainer[0].name if plan.trainer else plan.slices[0].name, None

    replica_match = _ROLLOUT_POOL_RE.match(prefix_lower)
    if replica_match and process_on_nodes:
        replica_idx = int(replica_match.group(1))
        nnodes = len(process_on_nodes)
        chips_per_node = int(process_on_nodes[0])
        tpu_slice, hosts = plan.replica_hosts(replica_idx, nnodes, chips_per_node)
        logger.info(f"[TPU slices] {name_prefix}: slice {tpu_slice.name}, hosts {hosts}")
        return tpu_slice.name, hosts

    # Reward / teacher pools and pools of unknown shape: round-robin over the rollout slices.
    match = _ROLLOUT_LIKE_RE.search(prefix_lower)
    idx = int(match.group(1)) if match else 0
    return plan.rollout[idx % len(plan.rollout)].name, None


def tpu_replica_rollout_config(rollout_config):
    """Rollout config of one TPU replica: ``data_parallel_size`` folded into the replica count.

    Works on the OmegaConf config the trainer holds and on an already converted ``RolloutConfig``.
    """
    import copy
    import dataclasses

    from omegaconf import DictConfig, open_dict

    if isinstance(rollout_config, DictConfig):
        replica_config = copy.deepcopy(rollout_config)
        with open_dict(replica_config):
            replica_config.data_parallel_size = 1
        return replica_config
    if dataclasses.is_dataclass(rollout_config):
        return dataclasses.replace(rollout_config, data_parallel_size=1)
    replica_config = copy.deepcopy(rollout_config)
    replica_config.data_parallel_size = 1
    return replica_config


def slice_chips_for(accelerator_type: Optional[str]) -> Optional[int]:
    """Total chips of the slice selected by ``accelerator_type`` according to the current plan."""
    plan = get_tpu_slice_plan()
    if plan is None or accelerator_type is None:
        return None
    for s in plan.slices:
        if s.name == accelerator_type:
            return s.chips
    return None

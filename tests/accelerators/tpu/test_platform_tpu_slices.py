# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for TPU slice discovery and rollout replica placement (no Ray cluster needed)."""

import dataclasses

import pytest
from omegaconf import OmegaConf

from verl_hardware_plugin.accelerators.tpu import platform_tpu_slices as slices_mod
from verl_hardware_plugin.accelerators.tpu.platform_tpu_slices import (
    TPUSlice,
    build_tpu_slice_plan,
    discover_tpu_slices,
    parse_slice_spec,
    plan_resource_pool_placement,
    resolve_tpu_rollout_layout,
    set_tpu_slice_plan,
    tpu_replica_rollout_config,
)


def _node(ip, slice_name=None, chips=4, worker_id=None, alive=True):
    resources = {"CPU": 48.0, "TPU": float(chips), f"node:{ip}": 1.0}
    labels = {}
    if slice_name is not None:
        resources[slice_name] = 1.0
        labels["ray.io/tpu-slice-name"] = slice_name
        labels["ray.io/tpu-pod-type"] = "v6e-8"
    if worker_id is not None:
        labels["ray.io/tpu-worker-id"] = str(worker_id)
    return {"NodeManagerAddress": ip, "Resources": resources, "Labels": labels, "Alive": alive}


# 2 x v6e-8 (2 hosts x 4 chips): worker ids deliberately disagree with IP order on tpu-group-1.
TWO_V6E8 = [
    {"NodeManagerAddress": "10.0.9.1", "Resources": {"CPU": 12.0, "node:10.0.9.1": 1.0}, "Labels": {}, "Alive": True},
    _node("10.0.0.41", "tpu-group-0", worker_id=0),
    _node("10.0.0.9", "tpu-group-0", worker_id=1),
    _node("10.0.0.70", "tpu-group-1", worker_id=1),
    _node("10.0.0.102", "tpu-group-1", worker_id=0),
    _node("10.0.0.200", "tpu-group-1", worker_id=0, alive=False),
]

# 1 x 8-chip trainer slice + 2 x single-host 4-chip rollout slices (the v5p NAP heterogeneous layout).
HETERO = [
    _node("10.1.0.1", "tpu-group-0", worker_id=0),
    _node("10.1.0.2", "tpu-group-0", worker_id=1),
    _node("10.1.1.1", "tpu-group-1", worker_id=0),
    _node("10.1.2.1", "tpu-group-2", worker_id=0),
]

# Two single-host slices without tpu-group-* resources (KubeRay numOfHosts=1): addressed as node:<ip>.
SINGLE_HOST = [_node("10.2.0.5"), _node("10.2.0.3")]


@pytest.fixture(autouse=True)
def _reset_plan(monkeypatch):
    set_tpu_slice_plan(None)
    monkeypatch.delenv(slices_mod.TRAINER_SLICES_ENV, raising=False)
    monkeypatch.delenv(slices_mod.ROLLOUT_SLICES_ENV, raising=False)
    yield
    set_tpu_slice_plan(None)


def test_discover_orders_slices_and_hosts_by_worker_id():
    slices = discover_tpu_slices(TWO_V6E8)
    assert [s.name for s in slices] == ["tpu-group-0", "tpu-group-1"]
    assert slices[0].hosts == ["10.0.0.41", "10.0.0.9"]
    # host 0 of the slice first, even though "10.0.0.102" > "10.0.0.70" neither as string nor as IP order
    assert slices[1].hosts == ["10.0.0.102", "10.0.0.70"]
    assert slices[1].chips == 8 and slices[1].chips_per_host == 4 and slices[1].index == 1


def test_discover_single_host_slices_use_node_resource():
    slices = discover_tpu_slices(SINGLE_HOST)
    assert [s.name for s in slices] == ["node:10.2.0.3", "node:10.2.0.5"]
    assert all(s.index is None and s.chips == 4 for s in slices)


@pytest.mark.parametrize(
    "spec, expected",
    [
        (None, None),
        ([], None),
        ([1, 2], ["tpu-group-1", "tpu-group-2"]),
        ("1,2", ["tpu-group-1", "tpu-group-2"]),
        (" 1 , tpu-group-2 ", ["tpu-group-1", "tpu-group-2"]),
        (["node:10.2.0.5"], ["node:10.2.0.5"]),
        (0, ["tpu-group-0"]),
    ],
)
def test_parse_slice_spec(spec, expected):
    assert parse_slice_spec(spec) == expected


def test_parse_slice_spec_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_slice_spec([1, "tpu-group-1"])


def test_default_plan_is_first_slice_trainer_rest_rollout():
    plan = build_tpu_slice_plan(slices=discover_tpu_slices(TWO_V6E8))
    assert [s.name for s in plan.trainer] == ["tpu-group-0"]
    assert [s.name for s in plan.rollout] == ["tpu-group-1"]
    assert not plan.explicit


def test_rollout_slices_pick_trainer_from_the_rest():
    plan = build_tpu_slice_plan(rollout_spec=[0], slices=discover_tpu_slices(TWO_V6E8))
    assert [s.name for s in plan.rollout] == ["tpu-group-0"]
    assert [s.name for s in plan.trainer] == ["tpu-group-1"]
    assert plan.explicit


def test_env_specs_are_used(monkeypatch):
    monkeypatch.setenv(slices_mod.ROLLOUT_SLICES_ENV, "1,2")
    monkeypatch.setenv(slices_mod.TRAINER_SLICES_ENV, "0")
    plan = build_tpu_slice_plan(slices=discover_tpu_slices(HETERO))
    assert [s.name for s in plan.rollout] == ["tpu-group-1", "tpu-group-2"]
    assert [s.name for s in plan.trainer] == ["tpu-group-0"]


def test_plan_rejects_overlap_unknown_and_no_trainer_slice():
    slices = discover_tpu_slices(TWO_V6E8)
    with pytest.raises(ValueError, match="both the trainer and the rollout"):
        build_tpu_slice_plan(trainer_spec=[1], rollout_spec=[1], slices=slices)
    with pytest.raises(ValueError, match="not in the Ray cluster"):
        build_tpu_slice_plan(rollout_spec=[7], slices=slices)
    with pytest.raises(ValueError, match="leaves no TPU slice for the trainer"):
        build_tpu_slice_plan(rollout_spec=[0, 1], slices=slices)
    assert build_tpu_slice_plan(slices=[]) is None
    with pytest.raises(ValueError, match="no TPU nodes"):
        build_tpu_slice_plan(rollout_spec=[1], slices=[])


def test_replica_hosts_pack_whole_hosts_slice_by_slice():
    plan = build_tpu_slice_plan(rollout_spec=[1, 2], slices=discover_tpu_slices(HETERO))
    assert plan.rollout_chips == 8
    assert plan.replica_hosts(0, 1, 4) == (plan.rollout[0], ["10.1.1.1"])
    assert plan.replica_hosts(1, 1, 4) == (plan.rollout[1], ["10.1.2.1"])
    with pytest.raises(ValueError, match="does not fit"):
        plan.replica_hosts(2, 1, 4)
    # a 2-host replica cannot straddle the two single-host slices
    with pytest.raises(ValueError, match="does not fit"):
        plan.replica_hosts(0, 2, 4)

    plan = build_tpu_slice_plan(slices=discover_tpu_slices(TWO_V6E8))
    # DP=2 x TP=4: one replica per host of tpu-group-1, host 0 first
    assert plan.replica_hosts(0, 1, 4)[1] == ["10.0.0.102"]
    assert plan.replica_hosts(1, 1, 4)[1] == ["10.0.0.70"]
    # TP=8: the whole slice
    assert plan.replica_hosts(0, 2, 4)[1] == ["10.0.0.102", "10.0.0.70"]
    with pytest.raises(ValueError, match="chips per host"):
        plan.replica_hosts(0, 1, 2)


def test_plan_resource_pool_placement_pins_rollout_pools():
    set_tpu_slice_plan(build_tpu_slice_plan(slices=discover_tpu_slices(TWO_V6E8)))
    assert plan_resource_pool_placement("global_pool", [4, 4]) == ("tpu-group-0", None)
    assert plan_resource_pool_placement("rollout_pool_0", [4]) == ("tpu-group-1", ["10.0.0.102"])
    assert plan_resource_pool_placement("rollout_pool_1", [4]) == ("tpu-group-1", ["10.0.0.70"])
    assert plan_resource_pool_placement("rollout_pool_0", [4, 4]) == ("tpu-group-1", ["10.0.0.102", "10.0.0.70"])
    # reward / teacher pools keep the slice round-robin, without host pins
    assert plan_resource_pool_placement("rollout_pool_reward_0", [4]) == ("tpu-group-1", None)
    with pytest.raises(ValueError, match="does not fit"):
        plan_resource_pool_placement("rollout_pool_2", [4])


def test_plan_resource_pool_placement_single_slice_cluster():
    set_tpu_slice_plan(build_tpu_slice_plan(slices=discover_tpu_slices(SINGLE_HOST[:1])))
    # one slice: trainer and rollout share it, as before
    assert plan_resource_pool_placement("global_pool", [4]) == ("node:10.2.0.5", None)
    assert plan_resource_pool_placement("rollout_pool_0", [4]) == ("node:10.2.0.5", None)


def _config(**rollout):
    base = {
        "trainer": {"nnodes": 2, "n_gpus_per_node": 4},
        "actor_rollout_ref": {
            "rollout": {
                "nnodes": 0,
                "n_gpus_per_node": 4,
                "tensor_model_parallel_size": 8,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "tpu_slices": None,
            }
        },
    }
    base["actor_rollout_ref"]["rollout"].update(rollout)
    return OmegaConf.create(base)


def test_resolve_layout_from_slices_derives_nnodes_and_dp(monkeypatch):
    monkeypatch.setattr(slices_mod, "discover_tpu_slices", lambda nodes=None: discover_tpu_slices(TWO_V6E8))
    config = _config(tpu_slices=[1], tensor_model_parallel_size=4)
    summary = resolve_tpu_rollout_layout(config)
    assert config.actor_rollout_ref.rollout.nnodes == 2
    assert config.actor_rollout_ref.rollout.n_gpus_per_node == 4
    assert summary["num_replicas"] == 2
    assert [s.name for s in summary["plan"].rollout] == ["tpu-group-1"]


def test_resolve_layout_dp_times_tp_must_match_chips(monkeypatch):
    monkeypatch.setattr(slices_mod, "discover_tpu_slices", lambda nodes=None: discover_tpu_slices(TWO_V6E8))
    config = _config(nnodes=2, tensor_model_parallel_size=4, data_parallel_size=2)
    assert resolve_tpu_rollout_layout(config)["num_replicas"] == 2

    config = _config(nnodes=2, tensor_model_parallel_size=4, data_parallel_size=4)
    with pytest.raises(ValueError, match="needs 16 rollout chips"):
        resolve_tpu_rollout_layout(config)

    config = _config(tpu_slices=[1], tensor_model_parallel_size=8, data_parallel_size=2)
    with pytest.raises(ValueError, match="needs 16 rollout chips"):
        resolve_tpu_rollout_layout(config)


def test_resolve_layout_rejects_replicas_sharing_a_host(monkeypatch):
    monkeypatch.setattr(slices_mod, "discover_tpu_slices", lambda nodes=None: discover_tpu_slices(TWO_V6E8))
    config = _config(nnodes=2, tensor_model_parallel_size=2)
    with pytest.raises(ValueError, match="whole hosts"):
        resolve_tpu_rollout_layout(config)


def test_resolve_layout_without_tpu_cluster_is_a_noop(monkeypatch):
    monkeypatch.setattr(slices_mod, "discover_tpu_slices", lambda nodes=None: [])
    config = _config(nnodes=2, tensor_model_parallel_size=4, data_parallel_size=2)
    assert resolve_tpu_rollout_layout(config) is None
    assert config.actor_rollout_ref.rollout.nnodes == 2


def test_tpu_replica_rollout_config_folds_dp():
    cfg = OmegaConf.create({"data_parallel_size": 4, "tensor_model_parallel_size": 4})
    replica_cfg = tpu_replica_rollout_config(cfg)
    assert replica_cfg.data_parallel_size == 1 and cfg.data_parallel_size == 4

    @dataclasses.dataclass(frozen=True)
    class Cfg:
        data_parallel_size: int = 4
        tensor_model_parallel_size: int = 4

    assert tpu_replica_rollout_config(Cfg()).data_parallel_size == 1


def test_slice_objects_are_comparable():
    a = TPUSlice("tpu-group-1", ["10.0.0.1"], 4)
    assert a == TPUSlice("tpu-group-1", ["10.0.0.1"], 4)

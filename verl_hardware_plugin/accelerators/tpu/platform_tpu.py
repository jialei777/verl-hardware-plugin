# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Google TPU platform implementation.

Registers the ``tpu`` platform and provides the device abstraction, the Ray resource
options, and the worker-environment plumbing a TPU process needs to join a slice. The
training engine, the generation engine and the weight-transfer path live elsewhere.

``torch_tpu`` mirrors part of the NVIDIA device API surface, so ``TPUDeviceModuleProxy``
adapts it to the memory-management calls verl expects. The platform class itself derives
from ``PlatformBase``: every device query, Ray resource option and environment variable is
defined here rather than inherited from another vendor's platform.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

import ray
import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import extend_torchtitan_engine_config
from verl_hardware_plugin.accelerators.tpu.patches import reduce_avg_allreduce_patch
from verl_hardware_plugin.accelerators.tpu.platform_tpu_slices import (
    plan_resource_pool_placement,
    resolve_tpu_rollout_layout,
    slice_chips_for,
    tpu_replica_rollout_config,
)

extend_torchtitan_engine_config()

logger = logging.getLogger(__name__)


def _ensure_torch_tpu() -> bool:
    """Try to import torch_tpu, which registers the ``tpu_dist`` distributed backend.

    Returns True if the TPU runtime is usable after the attempt.
    """
    if hasattr(torch, "tpu"):
        return True
    try:
        import torch_tpu  # noqa: F401

        return hasattr(torch, "tpu")
    except Exception as e:
        logger.debug("The current machine has no torch.tpu, because: %s", e)
    return False


_ensure_torch_tpu()  # Attempt at module load time so availability checks are faster later

# Base port for the TPU distributed slice builder mesh. Each local rank takes ``base + local_rank``.
TPU_PROCESS_BASE_PORT = 8471

# TPU chip HBM capacities in bytes
HBM_BYTES_TPU_V5P = 95 * 1024 * 1024 * 1024  # 95 GB
HBM_BYTES_TPU_V6E = 32 * 1024 * 1024 * 1024  # 32 GB
HBM_BYTES_TPU_V7X = 192 * 1024 * 1024 * 1024  # 192 GB

TPU_HBM_BYTES_MAP = {
    "v5p": HBM_BYTES_TPU_V5P,
    "v6e": HBM_BYTES_TPU_V6E,
    "tpu7x": HBM_BYTES_TPU_V7X,
    "v7x": HBM_BYTES_TPU_V7X,
}

# Fallback HBM capacity when the chip generation cannot be determined.
HBM_BYTES_TPU_DEFAULT = HBM_BYTES_TPU_V6E

# TPU slice topology (an ``x,y,z`` chip mesh) keyed by the number of chips in the slice. The vLLM
# rollout (``rollout/tpu_vllm_patches.py``) resolves its mesh here too, so the trainer and the
# rollout agree on the physical layout of a given slice size.
DEFAULT_TPU_TOPOLOGY_MAP = {
    1: "1,1,1",
    2: "1,2,1",
    4: "2,2,1",
    8: "2,4,1",
    16: "4,4,1",
    32: "4,8,1",
    64: "8,8,1",
    128: "8,16,1",
    256: "16,16,1",
}

# TPU v5p 3D mesh topology mappings (X, Y, Z) by pod type or total chips (1 logical device per chip in Megacore mode).
# Each host is 2x2x1 (4 chips = 8 TensorCores); multi-host slices stack along Z then X/Y (2x2x2 = 8 chips,
# 2x2x4 = 16 chips, 2x4x4 = 32 chips, 4x4x4 = 64 chips, 4x4x8 = 128 chips, 4x8x8 = 256 chips).
TPU_V5P_TOPOLOGY_MAP = {
    "v5p-512": "4,8,8",
    "v5p-256": "4,4,8",
    "v5p-128": "4,4,4",
    "v5p-64": "2,4,4",
    "v5p-32": "2,2,4",
    "v5p-16": "2,2,2",
    "v5p-8": "2,2,1",
    256: "4,8,8",
    128: "4,4,8",
    64: "4,4,4",
    32: "2,4,4",
    16: "2,2,4",
    8: "2,2,2",
    4: "2,2,1",
    2: "1,2,1",
    1: "1,1,1",
}

# TPU 7x (Ironwood) 4D mesh topology mappings (X, Y, Z, CoresPerChip=2), keyed by device count.
# A host has 4 chips (2x2x1) and 8 devices; the multi-host entries follow the GKE slice shapes
# (2x2x2 = 2 hosts, 2x2x4 = 4, 2x4x4 = 8, 4x4x4 = 16).
TPU_V7X_TOPOLOGY_MAP = {
    128: "4,4,4,2",
    64: "2,4,4,2",
    32: "2,2,4,2",
    16: "2,2,2,2",
    8: "2,2,1,2",
    4: "1,2,1,2",
    2: "1,1,1,2",
    1: "1,1,1,1",
}

# Pod types are consulted first: a pod type names the physical slice exactly, whereas the chip
# count is only the number of chips this job was given and can be a subset of the slice.
TPU_POD_TYPE_TOPOLOGY_MAP = {
    "v6e-4": "2,2,1",
    "v6e-8": "2,4,1",
    "v6e-16": "4,4,1",
    "v6e-32": "4,8,1",
    "v5p-8": "2,2,1",
    "v5p-16": "2,2,2",
    "v5p-32": "2,2,4",
    "v5p-64": "2,4,4",
    "v5p-128": "4,4,4",
    "v5p-256": "4,4,8",
    "v5p-512": "4,8,8",
}

# Slices at or below this many chips live on a single host, so the chips are addressed within
# the host bounds rather than across them.
TPU_SINGLE_HOST_MAX_CHIPS = 4


def get_tpu_topology_map(pod_type: str = "") -> dict:
    """Returns the topology map for the configured TPU generation (4D for TPU 7x, 3D for v5p/v6e)."""
    tpu_type = (pod_type or os.environ.get("TPU_ACCELERATOR_TYPE", "v6e")).lower()
    if any(k in tpu_type for k in ("tpu7x", "v7x")):
        return TPU_V7X_TOPOLOGY_MAP
    if "v5p" in tpu_type:
        return TPU_V5P_TOPOLOGY_MAP
    return DEFAULT_TPU_TOPOLOGY_MAP


def resolve_tpu_topology_bounds(
    total_chips: int,
    num_nodes: int,
    pod_type: str = "",
) -> tuple[str, str, str, str]:
    """Resolve ``(topology, host_bounds, chips_per_host_bounds, chips_per_host)`` for a TPU slice.

    Resolution order for the topology is: the ``TORCH_TPU_TOPOLOGY`` escape hatch, then the pod
    type reported by Ray, then the chip count.

    Raises:
        ValueError: if the topology cannot be determined. Guessing here is worse than failing:
            a wrong mesh does not error, it silently trains on part of the slice.
    """
    topo_map = get_tpu_topology_map(pod_type)
    unit_bounds = topo_map.get(1, "1,1,1")
    pod_key = pod_type.lower() if pod_type else ""
    topology = (
        os.environ.get("TORCH_TPU_TOPOLOGY")
        or TPU_POD_TYPE_TOPOLOGY_MAP.get(pod_key)
        or topo_map.get(pod_key)
        or topo_map.get(total_chips)
    )
    if not topology:
        raise ValueError(
            f"Cannot determine the TPU topology for total_chips={total_chips} "
            f"(pod_type={pod_type!r}). Set TORCH_TPU_TOPOLOGY='x,y,z' to override. "
            f"Known chip counts: {sorted(k for k in topo_map if isinstance(k, int))}."
        )

    inferred_chips_per_host = max(1, total_chips // max(1, num_nodes))
    chips_per_host = str(
        os.environ.get("VERL_TPU_CHIPS_PER_HOST")
        or os.environ.get("VLLM_TPU_CHIPS_PER_HOST")
        or inferred_chips_per_host
    )

    if total_chips <= TPU_SINGLE_HOST_MAX_CHIPS and unit_bounds == "1,1,1":
        host_bounds = unit_bounds
        chips_per_host_bounds = topology if num_nodes == 1 else unit_bounds
    else:
        host_bounds = topology
        chips_per_host_bounds = unit_bounds

    return topology, host_bounds, chips_per_host_bounds, chips_per_host


def get_tpu_chip_hbm_bytes() -> int:
    """Detect the TPU chip generation from Ray node labels or env vars and return its HBM capacity."""
    tpu_type = ""

    # Query Ray cluster node labels for TPU resource type
    try:
        if ray.is_initialized():
            tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
            if tpu_nodes:
                labels = tpu_nodes[0].get("Labels", {})
                tpu_type = (
                    labels.get("ray.io/accelerator-type")
                    or labels.get("ray.io/tpu-pod-type")
                    or labels.get("cloud.google.com/gke-tpu-accelerator")
                    or ""
                ).lower()
    except Exception as e:
        logger.warning(f"Unable to query Ray node labels for TPU chip type: {e}")

    # Fallback to environment variables
    if not tpu_type:
        tpu_type = (
            os.environ.get("TPU_ACCELERATOR_TYPE")
            or os.environ.get("ACCELERATOR_TYPE")
            or os.environ.get("TPU_TYPE")
            or ""
        ).lower()

    for chip_gen, hbm_bytes in TPU_HBM_BYTES_MAP.items():
        if chip_gen in tpu_type:
            return hbm_bytes

    logger.warning(f"Unable to determine TPU chip HBM bytes for tpu_type='{tpu_type}'. Returning -1.")
    return -1


class DummyTpuDeviceModule:
    """Fallback device module for CPU-only nodes and driver processes.

    Provides no-op implementations for torch.tpu APIs on processes where torch_tpu is not imported
    or no TPU devices are attached.
    """

    def is_available(self) -> bool:
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 0

    def synchronize(self) -> None:
        pass

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def get_device_name(self, device: Any = None) -> str:
        return os.environ.get("TPU_ACCELERATOR_TYPE", "v6e")


class TPUDeviceModuleProxy:
    """Proxy wrapper for torch.tpu to emulate the PyTorch NVIDIA memory management APIs.

    Provides default fallback implementations for memory tracking methods
    (e.g. memory_reserved, memory_allocated, get_device_properties) that are called throughout verl's
    codebase but not natively provided by torch_tpu.
    """

    def __init__(self, original_module):
        self.__dict__["_original_module"] = original_module

    def __getattr__(self, name):
        if name == "set_device":
            return self.set_device

        if hasattr(self._original_module, name):
            return getattr(self._original_module, name)

        if name == "memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "reset_peak_memory_stats":
            return lambda *args, **kwargs: None
        elif name == "get_device_name":
            return lambda *args, **kwargs: os.environ.get("TPU_ACCELERATOR_TYPE", "v6e")
        elif name == "get_device_properties":

            class DummyDeviceProperties:
                def __init__(self, total_memory=HBM_BYTES_TPU_DEFAULT):
                    self.total_memory = total_memory
                    self.name = "Google TPU"
                    self.major = 1
                    self.minor = 0

            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else HBM_BYTES_TPU_DEFAULT
            return lambda *args, **kwargs: DummyDeviceProperties(total_memory=total_mem)
        elif name == "mem_get_info":
            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else HBM_BYTES_TPU_DEFAULT
            return lambda *args, **kwargs: (total_mem, total_mem)

        raise AttributeError(f"'TPUDeviceModuleProxy' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._original_module, name, value)

    def is_available(self) -> bool:
        if hasattr(self._original_module, "is_available"):
            try:
                return self._original_module.is_available()
            except Exception as e:
                logger.warning(f"torch.tpu.is_available() check failed: {e}")
                return False
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        if hasattr(self._original_module, "current_device"):
            try:
                return self._original_module.current_device()
            except Exception as e:
                logger.warning(f"torch.tpu.current_device() failed: {e}")
                return 0
        return 0

    def device_count(self) -> int:
        if hasattr(self._original_module, "device_count"):
            try:
                return self._original_module.device_count()
            except Exception as e:
                logger.warning(f"torch.tpu.device_count() failed: {e}")
                return 0
        return 0

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if hasattr(self._original_module, "synchronize"):
            try:
                self._original_module.synchronize()
            except Exception as e:
                logger.warning(f"torch.tpu.synchronize() failed: {e}")

    def manual_seed_all(self, seed: int) -> None:
        # torch_tpu does not always expose manual_seed_all; the global torch seed is the
        # right fallback because a TPU process owns exactly one chip.
        fn = getattr(self._original_module, "manual_seed_all", None)
        if fn is not None:
            fn(seed)
        else:
            torch.manual_seed(seed)

    def empty_cache(self) -> None:
        if hasattr(self._original_module, "_clear_cache"):
            try:
                self._original_module._clear_cache()
            except Exception as e:
                logger.warning(f"Failed to clear TPU cache: {e}")


def _ensure_slicebuilder_insecure_grpc(raw_args: str) -> str:
    """Ensure LIBTPU_INIT_ARGS enables insecure SliceBuilder gRPC to avoid GKE ALTS handshake hangs."""
    tokens = raw_args.split()
    if not any(t.startswith("--slicebuilder_use_insecure_grpc") for t in tokens):
        tokens.append("--slicebuilder_use_insecure_grpc=true")
    if not any(t.startswith("--undefok=") and "slicebuilder_use_insecure_grpc" in t for t in tokens):
        tokens.append("--undefok=slicebuilder_use_insecure_grpc")
    return " ".join(tokens)


def patch_ray_worker() -> None:
    """Ray ``worker_process_setup_hook`` for GKE TPU pods.

    Runs once in every Ray worker process before any task. It does two things:

    1. Pins ``VERL_PLATFORM=tpu`` so a worker that did not inherit the driver's environment still
       resolves the TPU platform.
    2. Makes Raylet's accelerator-id lookup non-fatal. Each GKE pod is given only its own slice of
       the host's TPU chips, so a host-level index lookup can run off the end of the visible list
       and raise ``IndexError``. Returning an empty list is correct here: verl assigns chips itself
       via ``TPU_VISIBLE_CHIPS``.
    """
    os.environ["VERL_PLATFORM"] = "tpu"

    try:
        original_func = ray._private.worker.Worker.get_accelerator_ids_for_accelerator_resource

        def patched_func(self, resource_name, resource_regex):
            try:
                return original_func(self, resource_name, resource_regex)
            except IndexError as e:
                logger.debug("Intercepted Ray accelerator lookup IndexError for resource %r: %s", resource_name, e)
                return []

        ray._private.worker.Worker.get_accelerator_ids_for_accelerator_resource = patched_func
    except Exception as e:
        logger.warning(f"Failed to apply Ray worker accelerator patch: {e}")


def configure_torchrun_tpu_env() -> None:
    """Populate PJRT multi-process environment variables when launched via ``torchrun``.

    In Ray single-controller runs (``sft_trainer_ray`` / ``main_ppo``), ``get_worker_env_vars``
    populates ``TORCH_TPU_SLICEBUILDER_ADDRESSES`` and the topology bounds on each worker actor.
    In ``torchrun`` multi-controller runs (``verl.trainer.sft_trainer``), ``PlatformTPU()`` is
    instantiated inside ``auto_set_device`` / ``initialize_global_process_group`` before
    ``torch.distributed.init_process_group(backend="cpu:gloo,tpu:tpu_dist")`` runs. Setting
    missing PJRT variables from ``LOCAL_RANK`` / ``RANK`` / ``WORLD_SIZE`` lets ``torchrun``
    work out of the box without a wrapper script.
    """
    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    if "TORCH_TPU_SLICEBUILDER_ADDRESSES" in os.environ:
        return
    try:
        if ray.is_initialized():
            return
    except Exception:
        pass

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ.get("RANK", local_rank))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    except ValueError:
        return

    if world_size < 1 or local_world_size < 1:
        return

    num_nodes = max(1, world_size // local_world_size)
    raw_hosts = os.environ.get("TPU_WORKER_HOSTNAMES")
    if raw_hosts:
        hostnames = [h.strip() for h in raw_hosts.split(",") if h.strip()]
    elif num_nodes == 1:
        hostnames = [os.environ.get("MASTER_ADDR", "127.0.0.1")]
    else:
        return

    if len(hostnames) != num_nodes:
        return

    sb_addresses = [
        f"{hostnames[r // local_world_size]}:{TPU_PROCESS_BASE_PORT + (r % local_world_size)}"
        for r in range(world_size)
    ]

    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("TORCH_TPU_SLICEBUILDER_ADDRESSES", ",".join(sb_addresses))
    os.environ.setdefault("TPU_PROCESS_ADDRESSES", ",".join(sb_addresses))
    os.environ.setdefault("TPU_PROCESS_PORT", str(TPU_PROCESS_BASE_PORT + local_rank))
    os.environ.setdefault("CLOUD_TPU_TASK_ID", str(rank // local_world_size))
    os.environ.setdefault("TPU_WORKER_HOSTNAMES", ",".join(hostnames))
    os.environ.setdefault("TPU_VISIBLE_CHIPS", str(local_rank))

    pod_type = (
        os.environ.get("TPU_ACCELERATOR_TYPE") or os.environ.get("ACCELERATOR_TYPE") or os.environ.get("TPU_TYPE") or ""
    )
    try:
        topology, host_bounds, chips_per_host_bounds, chips_per_host = resolve_tpu_topology_bounds(
            total_chips=world_size,
            num_nodes=num_nodes,
            pod_type=pod_type,
        )
        os.environ.setdefault("TORCH_TPU_TOPOLOGY", topology)
        os.environ.setdefault("TPU_HOST_BOUNDS", host_bounds)
        os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", chips_per_host_bounds)
        os.environ.setdefault("CHIPS_PER_HOST", chips_per_host)
    except ValueError:
        pass


@PlatformRegistry.register(platform="tpu")
class PlatformTPU(PlatformBase):
    """Platform backend for Google TPUs.

    Answers device queries, builds the Ray resource options for TPU actors, and supplies
    the PJRT environment a worker needs. The parts of ``PlatformBase`` modelled on a CUDA
    device are answered explicitly here rather than inherited.
    """

    def __init__(self):
        super().__init__()
        reduce_avg_allreduce_patch.apply()
        configure_torchrun_tpu_env()
        original_tpu = getattr(torch, "tpu", DummyTpuDeviceModule())
        self._device_module = TPUDeviceModuleProxy(original_tpu)
        self._claimed_slices: set[str] = set()

    @property
    def vendor_name(self) -> str:
        return "google"

    @property
    def device_name(self) -> str:
        return "tpu"

    @property
    def device_module(self):
        return self._device_module

    def current_device(self) -> int:
        return self.device_module.current_device()

    def device_count(self) -> int:
        return self.device_module.device_count()

    def set_device(self, device_index: int) -> None:
        self.device_module.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        self.device_module.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        self.device_module.manual_seed_all(seed)

    def empty_cache(self) -> None:
        # Delegate when torch.tpu provides it. A TPU runtime that manages HBM itself has no
        # CUDA-style caching allocator to drain, in which case there is nothing to do.
        empty = getattr(self.device_module, "empty_cache", None)
        if empty is not None:
            empty()

    def set_allocator_settings(self, settings: str) -> None:
        # TPU exposes no configurable caching allocator equivalent to CUDA's.
        pass

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # TPU does not use the CUDA compute capability model.
        return (None, None)

    def is_available(self) -> bool:
        if hasattr(torch, "tpu"):
            try:
                return torch.tpu.is_available()
            except Exception as e:
                logger.warning(f"torch.tpu.is_available() check failed: {e}")
        return False

    def is_platform_available(self, use_smi_check=False) -> bool:
        if os.environ.get("VERL_PLATFORM") == "tpu":
            return True
        if "TPU_NAME" in os.environ or "TPU_VISIBLE_DEVICES" in os.environ:
            return True
        # The env vars come first because a CPU-only Ray driver process has no chip attached
        # yet still belongs to a TPU job. Falling back to the runtime lets a plain TPU VM,
        # where nothing sets TPU_NAME, detect itself instead of defaulting to CUDA.
        return self.is_available()

    def ray_resource_name(self) -> str:
        return "TPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        tpu_chips = int(num_gpus)
        return {"resources": {"TPU": tpu_chips}} if tpu_chips >= 1 else {}

    def communication_backend_name(self) -> str:
        return "tpu_dist"

    def visible_devices_envvar(self) -> str:
        # Deliberately CUDA_VISIBLE_DEVICES, not TPU_VISIBLE_CHIPS. verl assigns
        # os.environ[this key] when launching vLLM servers; pointing it at
        # TPU_VISIBLE_CHIPS would overwrite the chip index that get_worker_env_vars()
        # writes and ray_local_rank_override() reads. Changing this breaks rank mapping.
        return "CUDA_VISIBLE_DEVICES"

    def is_ipc_supported(self) -> bool:
        # TPU has no CUDA-style IPC handle. verl derives ``use_shm = not is_support_ipc()``,
        # so returning False routes weight transfer through shared memory instead.
        return False

    def cudart(self) -> Any:
        # There is no CUDA runtime on a TPU host; PlatformBase documents None for "not applicable".
        return None

    # No TPU profiler or NVTX equivalent. PlatformBase asks platforms without profiling
    # support to make these no-ops: yield immediately, and start/stop do nothing.
    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def ray_noset_envvars(self) -> list[str]:
        # CUDA_VISIBLE_DEVICES is listed because visible_devices_envvar() deliberately returns
        # it (see above), so Ray must be told not to manage that variable either.
        return [
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        ]

    def ray_local_rank_override(self) -> Optional[str]:
        # Ray does not enumerate TPU chips, so the index comes from TPU_VISIBLE_CHIPS, which
        # get_worker_env_vars() sets when the worker is created.
        return os.environ.get("TPU_VISIBLE_CHIPS", "0")

    def supports_colocated_worker_groups(self) -> bool:
        # A TPU chip belongs to one process and cannot be shared between colocated WorkerGroups.
        return False

    def rollout_env_vars(self) -> dict[str, str]:
        """Return env vars to inject into the rollout engine.

        The rollout server actor is created with an explicit ``runtime_env``, which replaces
        inheritance from the job, so compiler/runtime flags set on the driver must be named
        to survive. Also merges ``VERL_TPU_EXTRA_<VAR>`` overrides and sets
        ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`` so vLLM forwards them to its worker actors.
        """
        env_vars: dict[str, str] = {}
        copied: list[str] = []
        for var in ("XLA_FLAGS", "LIBTPU_INIT_ARGS"):
            base = os.environ.get(var, "")
            extra = os.environ.get(f"VERL_TPU_EXTRA_{var}", "")
            combined = " ".join(part for part in (base, extra) if part).strip()
            if combined:
                env_vars[var] = combined
                copied.append(var)
        if "LIBTPU_INIT_ARGS" in env_vars:
            env_vars["LIBTPU_INIT_ARGS"] = _ensure_slicebuilder_insecure_grpc(env_vars["LIBTPU_INIT_ARGS"])
        if os.environ.get("TPU_ACCELERATOR_TYPE"):
            env_vars["TPU_ACCELERATOR_TYPE"] = os.environ["TPU_ACCELERATOR_TYPE"]
            copied.append("TPU_ACCELERATOR_TYPE")
        if os.environ.get("VERL_USE_EXTERNAL_MODULES"):
            env_vars["VERL_USE_EXTERNAL_MODULES"] = os.environ["VERL_USE_EXTERNAL_MODULES"]
            copied.append("VERL_USE_EXTERNAL_MODULES")
        if copied:
            existing = os.environ.get("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "")
            extra_copy = [k.strip() for k in existing.split(",") if k.strip()] + copied
            env_vars["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(dict.fromkeys(extra_copy))
        return env_vars

    def get_ray_init_kwargs(self) -> dict[str, Any]:
        """Install the GKE TPU worker setup hook and pin the platform inside Ray workers.

        The hook is passed as a module path rather than the function. Ray exports a function under
        a generated key and keeps only the function's name in the job's runtime env. vLLM's
        EngineCore calls ``ray.init`` again with the runtime env it captured from the server actor,
        and Ray rejects that name as conflicting with the key. A module path round-trips unchanged.
        """
        env_vars = {
            "VERL_PLATFORM": "tpu",
            "TPU_ACCELERATOR_TYPE": os.environ.get("TPU_ACCELERATOR_TYPE", "v6e"),
            "LIBTPU_INIT_ARGS": _ensure_slicebuilder_insecure_grpc(os.environ.get("LIBTPU_INIT_ARGS", "")),
        }
        for key in (
            "VERL_USE_EXTERNAL_MODULES",
            "WANDB_API_KEY",
            "WANDB_BASE_URL",
            "WANDB_ENTITY",
            "WANDB_PROJECT",
            "WANDB_MODE",
            "VERL_FILE_LOGGER_ROOT",
            "VERL_FILE_LOGGER_PATH",
            "TENSORBOARD_DIR",
        ):
            if os.environ.get(key):
                env_vars[key] = os.environ[key]
        return {
            "runtime_env": {
                "worker_process_setup_hook": f"{patch_ray_worker.__module__}.{patch_ray_worker.__qualname__}",
                "env_vars": env_vars,
            }
        }

    def auto_assign_accelerator_type(self, name_prefix: str, accelerator_type: Optional[str]) -> Optional[str]:
        """Pin a resource pool to its own TPU slice on multi-slice clusters.

        Called from ``verl.single_controller.ray.base.RayResourcePool.get_placement_groups``
        whenever ``device_name == "tpu"``; ``PlatformBase`` does not declare it, so the TPU
        platform has to provide it.

        A KubeRay TPU cluster advertises one ``tpu-group-<n>`` custom resource per slice. Without
        an affinity label a pool can straddle two slices, which the TPU mesh cannot span. Each pool
        gets the first slice that no earlier pool claimed, so the trainer and rollout pools land on
        different slices; once every slice is claimed, further pools share the last one.
        """
        if accelerator_type is not None:
            return accelerator_type

        planned_slice, _ = self.plan_resource_pool_placement(name_prefix)
        if planned_slice is not None:
            return planned_slice

        try:
            if ray.is_initialized():
                tpu_slices = set()
                for node in ray.nodes():
                    if node.get("Alive"):
                        for res in node.get("Resources", {}):
                            if res.startswith("tpu-group-"):
                                tpu_slices.add(res)
                slices = sorted(tpu_slices)
                for tpu_slice in slices:
                    if tpu_slice not in self._claimed_slices:
                        self._claimed_slices.add(tpu_slice)
                        return tpu_slice
                if slices:
                    return slices[-1]
        except Exception as e:
            logger.debug("Could not auto-assign a TPU slice for %r: %s", name_prefix, e)

        return accelerator_type

    def resolve_rollout_layout(self, config: Any) -> Optional[dict]:
        """Resolve TPU trainer/rollout slice layout and populate rollout nnodes/n_gpus_per_node."""
        return resolve_tpu_rollout_layout(config)

    def replica_rollout_config(self, rollout_config: Any) -> Any:
        """Return a per-replica copy of ``rollout_config`` with ``data_parallel_size=1``."""
        return tpu_replica_rollout_config(rollout_config)

    def plan_resource_pool_placement(
        self, name_prefix: str, process_on_nodes: Optional[list[int]] = None
    ) -> tuple[Optional[str], Optional[list[str]]]:
        """Slice resource and per-placement-group host pins for a resource pool.

        Trainer pools get the trainer slice; ``rollout_pool_<idx>`` pools get the slice and the hosts
        of rollout replica ``idx`` from the slice plan (see ``platform_tpu_slices``), so that several
        replicas on one multi-host slice occupy distinct hosts instead of racing for the same one.
        """
        try:
            return plan_resource_pool_placement(name_prefix, process_on_nodes)
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"TPU slice assignment failed for {name_prefix}: {e}")
            return None, None

    def pin_placement_group_scheme(self, pg_scheme: list[list[dict]], hosts: Optional[list[str]]) -> None:
        """Pin each placement group of ``pg_scheme`` to the host planned for it."""
        if not hosts:
            return
        if len(hosts) != len(pg_scheme):
            logger.warning(
                f"TPU host pins {hosts} do not match the {len(pg_scheme)} placement group(s); leaving them unpinned"
            )
            return
        for bundles, ip in zip(pg_scheme, hosts, strict=True):
            for bundle in bundles:
                bundle[f"node:{ip}"] = 1e-4

    def configure_placement_group_bundle(
        self,
        bundle: dict,
        use_gpu: bool,
        device_name: str,
        name_prefix: str,
        accelerator_type: Optional[str] = None,
    ) -> None:
        """Shape a placement-group bundle for GKE TPU.

        Called unconditionally from ``verl.single_controller.ray.base.RayResourcePool``
        whenever ``device_name == "tpu"``; ``PlatformBase`` does not declare it, so the TPU
        platform has to provide it.

        The slice affinity is requested as a fractional amount so it acts as a label rather
        than a real reservation.
        """
        if use_gpu:
            bundle[device_name] = 1
        if accelerator_type is not None:
            bundle[accelerator_type] = 1e-4

    def get_tpu_env_vars(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        pgs: list,
        accelerator_type: Optional[str] = None,
    ) -> dict[str, str]:
        """Generate TPU-specific distributed environment variables for PJRT mesh initialization.

        ``pgs`` are the placement groups of the worker's resource pool. A pool is pinned to one
        slice (see ``auto_assign_accelerator_type``), so its bundles, in placement-group and
        bundle order, are exactly the processes of the PJRT mesh. This runs in the process that
        creates the workers (the trainer's TaskRunner), so nothing here may depend on the IP of
        the calling process -- it can be the head node or any TPU host.
        """
        node_ip_map = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes() if node.get("Alive", False)}
        bundle_ips = []

        for pg in pgs:
            specs = ray._private.state.state.placement_group_table(pg.id)
            if specs.get("state") != "CREATED":
                continue
            bundles_map = specs.get("bundles_to_node_id", {})
            for b_idx in sorted(bundles_map.keys()):
                node_id = bundles_map[b_idx]
                if node_id in node_ip_map:
                    bundle_ips.append(node_ip_map[node_id])

        base_port = TPU_PROCESS_BASE_PORT
        sb_addresses = [f"{ip}:{base_port + (b_idx % local_world_size)}" for b_idx, ip in enumerate(bundle_ips)]

        # Extract unique worker hostnames preserving rank order
        unique_hostnames = list(dict.fromkeys(bundle_ips))

        env_vars = {
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "CLOUD_TPU_TASK_ID": str(rank // local_world_size),
            "TPU_WORKER_HOSTNAMES": ",".join(unique_hostnames),
            "TPU_VISIBLE_CHIPS": str(local_rank),
            "LIBTPU_INIT_ARGS": _ensure_slicebuilder_insecure_grpc(os.environ.get("LIBTPU_INIT_ARGS", "")),
        }

        # Derive the mesh from the pod type Ray reports, falling back to the chip count.
        # If this worker group uses only part of a slice (e.g. one TP=4 replica per host of a 2-host
        # slice), resolve the mesh from world_size rather than the full slice's pod_type label.
        tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
        tpu_type = tpu_nodes[0].get("Labels", {}).get("ray.io/tpu-pod-type", "") if tpu_nodes else ""
        slice_chips = slice_chips_for(accelerator_type)
        effective_pod_type = "" if (slice_chips is not None and world_size != slice_chips) else tpu_type

        topology, host_bounds, chips_per_host_bounds, chips_per_host = resolve_tpu_topology_bounds(
            total_chips=world_size,
            num_nodes=len(unique_hostnames),
            pod_type=effective_pod_type,
        )

        env_vars.update(
            {
                "TORCH_TPU_TOPOLOGY": topology,
                "TPU_HOST_BOUNDS": host_bounds,
                "TPU_CHIPS_PER_HOST_BOUNDS": chips_per_host_bounds,
                "CHIPS_PER_HOST": chips_per_host,
            }
        )

        return env_vars

    def get_worker_env_vars(
        self,
        resource_pool,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        device_name: str,
    ) -> dict[str, str]:
        """Return platform-specific TPU environment variables for worker nodes."""
        env_vars = {
            "TPU_ACCELERATOR_TYPE": os.environ.get("TPU_ACCELERATOR_TYPE", "v6e"),
        }
        if "VERL_PLATFORM" in os.environ:
            env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]
        for var in self.ray_noset_envvars():
            env_vars[var] = "1"
        pgs = resource_pool.get_placement_groups(device_name=device_name)
        tpu_env = self.get_tpu_env_vars(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            name_prefix=name_prefix,
            pgs=pgs,
            accelerator_type=getattr(resource_pool, "accelerator_type", None),
        )
        env_vars.update(tpu_env)
        return env_vars

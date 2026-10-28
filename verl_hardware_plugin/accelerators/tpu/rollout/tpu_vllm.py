# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM rollout replica and HTTP server for Google TPU.

Upstream verl builds the vLLM rollout through ``RolloutReplicaRegistry``. On TPU the plugin
registers a ``vllm`` loader that returns :class:`TPUvLLMReplica`. It keeps upstream's HTTP server
and changes only what TPU needs:

* One server actor per replica, on the first worker's node (``launch_tpu_vllm_servers``). vLLM
  then spans the replica's hosts through its Ray executor, instead of verl's one server per node.
* ``distributed_executor_backend=external_launcher`` and ``enable_sleep_mode=False``: the
  multi-host executor is selected by ``patch_vllm_for_tpu``, and vllm-torchtpu has no sleep
  mode.
* With ``checkpoint_engine.backend=tpu-sync``, vLLM's workers load the tpu-sync worker extension
  (``tpu_sync_worker_extension.py``), which receives the trainer's weights over the network.

The same logic previously lived behind ``get_resource_name() == "TPU"`` branches in verl's
``vllm_async_server.py`` and ``replica.py``.
"""

import asyncio
import inspect
import logging
import time
from typing import Any

import ray

from verl.plugin.platform import get_platform
from verl.utils.device import get_device_name
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from verl_hardware_plugin.accelerators.tpu.profilers.tpu_vllm_timeline import vLLMServerTimelineMixin
from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import patch_vllm_for_tpu
from verl_hardware_plugin.accelerators.tpu.weight_sync.weight_sync_backends import TPU_SYNC_BACKEND, canonical_backend

logger = logging.getLogger(__name__)

# Matches upstream ``vllm_async_server._GATE_BARRIER_TIMEOUT_S``.
_GATE_BARRIER_TIMEOUT_S = 60.0

# vLLM imports the worker extension by name in its worker processes.
TPU_SYNC_WORKER_EXTENSION_CLS = (
    "verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension.vLLMTPUSyncWorkerExtension"
)


async def launch_tpu_vllm_servers(replica: vLLMReplica) -> None:
    """Launch the vLLM rollout server actor for a TPU replica."""
    if replica.config.data_parallel_size > 1:
        raise NotImplementedError(
            "actor_rollout_ref.rollout.data_parallel_size="
            f"{replica.config.data_parallel_size} is not supported on TPU. The TPU "
            "distributed runtime requires each compiled program to span the whole "
            "TPU mesh, but a single DP group only spans "
            f"tensor_model_parallel_size={replica.config.tensor_model_parallel_size} "
            "chips. Set actor_rollout_ref.rollout.data_parallel_size=1 instead: verl "
            "then creates one rollout replica per tensor_model_parallel_size chips, "
            "which is equivalent to engine-internal data parallelism."
        )

    # Upstream starts one server per node and gives it the node's devices. On TPU one server, on the
    # first worker's node, drives all of the replica's hosts and holds no chips: patch_vllm_for_tpu
    # gives each of vLLM's own workers a chip.
    node_id = await replica.workers[0].__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())

    prefix = replica._get_server_name_prefix()
    if replica.is_reward_model:
        name = f"{prefix}server_reward_{replica.replica_rank}_0{replica.name_suffix}"
    elif replica.is_teacher_model:
        name = f"{prefix}server_teacher_{replica.replica_rank}_0{replica.name_suffix}"
    else:
        name = f"{prefix}server_{replica.replica_rank}_0{replica.name_suffix}"

    pgs = replica.resource_pool.get_placement_groups(device_name=get_device_name())
    env_vars = {
        **{var: "1" for var in get_platform().ray_noset_envvars()},
        **get_platform().rollout_env_vars(),
        # One engine spans all of the replica's hosts through Ray. patch_vllm_for_tpu builds on
        # vllm-torchtpu's Ray multi-host backend with the V1 executor, not the V2 one.
        "TPU_MULTIHOST_BACKEND": "ray",
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "0",
        # The initialize_ray_cluster patch in patch_vllm_for_tpu attaches vLLM to these placement groups.
        "VERL_TPU_PG_IDS": ",".join(pg.id.hex() for pg in pgs),
    }

    server = replica.server_class.options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        ),
        runtime_env={"env_vars": env_vars},
        name=name,
        max_concurrency=replica.config.ray_actor_max_concurrency,
    ).remote(
        config=replica.config,
        model_config=replica.model_config,
        rollout_mode=replica.rollout_mode,
        workers=replica.workers,
        replica_rank=replica.replica_rank,
        node_rank=0,
        gpus_per_node=replica.gpus_per_replica_node,
        nnodes=replica.nnodes,
        cuda_visible_devices="",
    )
    replica.servers.append(server)

    master_address, master_port, dp_rpc_port = await replica.servers[0].get_master_address.remote()
    await replica.servers[0].launch_server.remote(
        master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
    )

    server_address, server_port = await replica.servers[0].get_server_address.remote()
    replica._server_handle = replica.servers[0]
    replica._server_address = (
        f"[{server_address}]:{server_port}"
        if is_valid_ipv6_address(server_address)
        else f"{server_address}:{server_port}"
    )


class _TPUAdmissionGateMixin:
    """Unblocks ``abort_all_requests`` once admitted requests enter ``AsyncLLM``, not on their first token.

    Upstream ``vLLMHttpServer`` decrements ``_admitting`` only when ``engine.generate()`` yields a token. When more
    than ``max_num_seqs`` requests are queued against one replica, the overflow has already completed
    ``AsyncLLM.add_request`` (registered in ``output_processor`` and ``engine_core``) and sits in ``WAITING`` without
    yielding, stalling ``abort_all_requests`` for ``_GATE_BARRIER_TIMEOUT_S`` (60 s) before ``pause_generation`` aborts
    them anyway. Tracking ``_pending_engine_entry`` across ``_park_until_admitted`` -> ``engine.add_request`` bounds
    the gate barrier to engine registration (~ms) while keeping ``pause_generation`` race-free.
    """

    _engine_add_request_hooked = False
    _resume_event: asyncio.Event
    engine: Any

    def _pending_entries(self) -> set[str]:
        pending = getattr(self, "_pending_engine_entry", None)
        if pending is None:
            pending = set()
            self._pending_engine_entry = pending
        return pending

    async def launch_server(self, *args, **kwargs):
        result = await super().launch_server(*args, **kwargs)  # type: ignore[misc]
        engine = getattr(self, "engine", None)
        add_request = getattr(engine, "add_request", None)
        if getattr(self, "node_rank", 0) == 0 and callable(add_request):
            pending = self._pending_entries()

            async def _tracked_add_request(*r_args, **r_kwargs):
                req_id = r_args[0] if r_args else r_kwargs.get("request_id")
                try:
                    return await add_request(*r_args, **r_kwargs)
                finally:
                    if isinstance(req_id, str):
                        pending.discard(req_id)

            engine.add_request = _tracked_add_request
            self._engine_add_request_hooked = True
        return result

    async def _park_until_admitted(self, request_id: str):
        rejected = await super()._park_until_admitted(request_id)  # type: ignore[misc]
        if rejected is None and self._engine_add_request_hooked:
            self._pending_entries().add(request_id)
        return rejected

    async def generate(self, prompt_ids, sampling_params, request_id, *args, **kwargs):
        try:
            return await super().generate(prompt_ids, sampling_params, request_id, *args, **kwargs)  # type: ignore[misc]
        finally:
            if self._engine_add_request_hooked:
                self._pending_entries().discard(request_id)

    async def abort_all_requests(
        self, reset_prefix_cache: bool = True, reject_request: bool = False, abort_only: bool = False
    ) -> dict[str, Any]:
        if abort_only or getattr(self, "node_rank", 0) != 0:
            return await super().abort_all_requests(  # type: ignore[misc]
                reset_prefix_cache=reset_prefix_cache, reject_request=reject_request, abort_only=abort_only
            )
        try:
            self._submission_paused = True
            self._rejecting = reject_request
            self._resume_event.clear()
            pending = self._pending_entries()
            deadline = time.monotonic() + _GATE_BARRIER_TIMEOUT_S
            while pending if self._engine_add_request_hooked else getattr(self, "_admitting", 0) > 0:
                if time.monotonic() > deadline:
                    in_flight = len(pending) if self._engine_add_request_hooked else getattr(self, "_admitting", 0)
                    logger.warning(
                        "Submission gate barrier timed out with %d request(s) entering engine, proceeding",
                        in_flight,
                    )
                    break
                await asyncio.sleep(0.005)
            request_ids = list(self.engine.output_processor.request_states.keys())
            await _pause_generation_abort(self.engine, clear_cache=reset_prefix_cache)
            logger.info("Aborted %d requests: %s", len(request_ids), request_ids)
            return {"aborted_count": len(request_ids), "request_ids": request_ids}
        except Exception:
            logger.exception("Error aborting requests")
            raise


async def _pause_generation_abort(engine: Any, *, clear_cache: bool) -> None:
    """Pause ``engine`` and abort in-flight requests across vLLM 0.31 and 0.32+.

    vLLM <= 0.31 takes ``wait_for_inflight_requests=False``; vLLM >= 0.32 replaced that
    boolean with ``mode="abort"``. Inspect the class method so instance-level wrappers
    (such as ``vLLMServerTimelineMixin``) do not hide the parameter list.
    """
    raw = inspect.unwrap(getattr(type(engine), "pause_generation", engine.pause_generation))
    try:
        has_mode = "mode" in inspect.signature(raw).parameters
    except (TypeError, ValueError):
        has_mode = False
    if has_mode:
        await engine.pause_generation(mode="abort", clear_cache=clear_cache)
    else:
        await engine.pause_generation(wait_for_inflight_requests=False, clear_cache=clear_cache)


class TPUvLLMHttpServer(vLLMServerTimelineMixin, _TPUAdmissionGateMixin, vLLMHttpServer):
    """``vLLMHttpServer`` with the TPU engine arguments."""

    def __init__(self, *args, **kwargs):
        # This actor builds the vLLM engine and starts its EngineCore process, so patch vLLM before
        # anything else. The patches carry themselves into the EngineCore and vLLM's workers.
        patch_vllm_for_tpu()
        super().__init__(*args, **kwargs)

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        super()._preprocess_engine_kwargs(engine_kwargs)
        # engine_kwargs is merged last into the vLLM CLI args, so these win over the defaults.
        # patch_vllm_for_tpu switches the engine to vLLM's Ray executor at config time.
        engine_kwargs["distributed_executor_backend"] = "external_launcher"
        engine_kwargs["enable_sleep_mode"] = False

    def _get_worker_extension_cls(self) -> str:
        # The tpu-sync checkpoint engine transfers weights straight into vLLM's workers, which need the
        # methods of the tpu-sync worker extension. Other backends keep upstream's extension.
        checkpoint_engine = getattr(getattr(self, "config", None), "checkpoint_engine", None)
        if canonical_backend(getattr(checkpoint_engine, "backend", None)) == TPU_SYNC_BACKEND:
            return TPU_SYNC_WORKER_EXTENSION_CLS
        return super()._get_worker_extension_cls()


class TPUvLLMReplica(vLLMReplica):
    """vLLM rollout replica for TPU: one server actor drives all of the replica's hosts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TPUvLLMHttpServer)

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )
        await launch_tpu_vllm_servers(self)

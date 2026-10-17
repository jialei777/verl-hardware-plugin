# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU helper utilities for the TorchTitan training engine.

Provides sequence-length bucketing, host-side metadata unwrapping, packed-input padding,
``torch.compile(backend="tpu")`` Dynamo configuration, ``TPUSplashAttention``, the torchtitan TPU
recipe that shards with ``SimpleFSDP``, cross-rank micro-batch shape alignment, and ``DEFER_AND_FUSE``
eager-mode contexts for ``torch_tpu`` training.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch._dynamo
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from verl.utils import tensordict_utils as tu
from verl.workers.config import TorchtitanEngineConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Attribute name used to attach the full bucket-padded device tensor to a CPU
# nested tensor returned from ``prepare_model_outputs``, so loss functions can
# consume the fixed-shape device tensor directly without XLA recompilation.
TPU_PADDED_VALUES_ATTR = "_tpu_padded_values"

_DEFAULT_TPU_SEQ_BUCKET_SIZE = 256

# Mapping from HuggingFace model_type to torchtitan model name.
# Torchtitan models not mapped here:
#   - flux: diffusion model, not applicable to verl's RL/SFT workflows
#   - llama3_ft: fault-tolerant variant of llama3, same HF models (mapped via "llama")
_HF_MODEL_TYPE_TO_TORCHTITAN_NAME = {
    "qwen2": "qwen3",
    "qwen3": "qwen3",
    "qwen2_moe": "qwen3",
    "qwen3_moe": "qwen3",
    "llama": "llama3",
    "llama4": "llama4",
    "deepseek_v3": "deepseek_v3",
    "gpt_oss": "gpt_oss",
}

# torchtitan TPU recipes whose ``parallelize_fn`` shards with SimpleFSDP when ``parallelism.use_simple_fsdp`` is set.
_TPU_SIMPLE_FSDP_RECIPES = {
    "qwen3": ("torchtitan.experiments.tpu.qwen3.infra.parallelize", "parallelize_qwen3"),
    "llama3": ("torchtitan.experiments.tpu.llama3.infra.parallelize", "parallelize_llama"),
}


def create_noop_dataloader_config() -> Any:
    """Create a no-op TorchTitan ``BaseDataLoader.Config`` when verl manages its own data loading."""
    from torchtitan.components.data.loader import BaseDataLoader

    class NoOpDataLoader(BaseDataLoader):
        """A no-op dataloader for use when verl manages its own data loading.

        Satisfies the BaseDataLoader interface required by torchtitan's Trainer
        but does nothing. Its __iter__ yields nothing, and state_dict /
        load_state_dict are no-ops.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(BaseDataLoader.Config):
            pass

        def __init__(self, **kwargs: Any):
            pass

        def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
            return iter([])

        def state_dict(self) -> dict[str, Any]:
            return {}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            pass

    return NoOpDataLoader.Config()


def derive_torchtitan_name_and_flavor(hf_config: Any) -> tuple[str, str]:
    """Derive torchtitan model name and flavor from a HuggingFace config.

    The name is mapped from ``hf_config.model_type``. The flavor is found by
    matching architecture parameters (dim, n_layers, vocab_size) against the
    known flavors registered in the torchtitan model package.

    Args:
        hf_config: A HuggingFace AutoConfig object.

    Returns:
        A ``(name, flavor)`` tuple.

    Raises:
        ValueError: If model_type is unsupported or no matching flavor is found.
    """
    model_type = getattr(hf_config, "model_type", None)
    if model_type is None:
        raise ValueError("HuggingFace config does not have 'model_type' field")

    name = _HF_MODEL_TYPE_TO_TORCHTITAN_NAME.get(model_type)
    if name is None:
        raise ValueError(
            f"Cannot derive torchtitan model name from HF model_type '{model_type}'. "
            f"Supported types: {list(_HF_MODEL_TYPE_TO_TORCHTITAN_NAME.keys())}."
        )

    # Import the model package and use model_registry to build each flavor's config.
    # model_registry has sensible defaults for all optional params (attn_backend, etc.).
    model_module = importlib.import_module(f"torchtitan.models.{name}")
    model_registry = model_module.model_registry

    # The configs dict name isn't derivable from the model name
    # (e.g. gpt_oss -> gptoss_configs), so we find it by convention.
    flavor_names = None
    for attr, obj in vars(model_module).items():
        if attr.endswith("_configs") and isinstance(obj, dict):
            flavor_names = list(obj.keys())
            break

    if flavor_names is None:
        raise ValueError(
            f"Could not find model configs dict in torchtitan.models.{name}. "
            f"Expected a dict attribute ending with '_configs'."
        )

    hidden_size = hf_config.hidden_size
    num_layers = hf_config.num_hidden_layers
    vocab_size = hf_config.vocab_size

    for flavor_name in flavor_names:
        cfg = model_registry(flavor_name).model
        n_layers = getattr(cfg, "n_layers", None) or len(getattr(cfg, "layers", []))
        if (
            getattr(cfg, "dim", None) == hidden_size
            and n_layers == num_layers
            and getattr(cfg, "vocab_size", None) == vocab_size
        ):
            logger.info(
                "Auto-derived torchtitan name=%r, flavor=%r from HF model_type=%r",
                name,
                flavor_name,
                model_type,
            )
            return name, flavor_name

    raise ValueError(
        f"No matching torchtitan flavor found for model_type='{model_type}' "
        f"(hidden_size={hidden_size}, num_hidden_layers={num_layers}, "
        f"vocab_size={vocab_size}). "
        f"Available flavors for '{name}': {flavor_names}."
    )


def enable_fsdp_gradient_division(model: nn.Module, dp_size: int) -> None:
    """
    Re-enable FSDP's automatic gradient division.

    TorchTitan calls disable_fsdp_gradient_division() which sets gradient_divide_factor=1.0.
    This re-enables it by setting the factor to the specified dp_size, so gradients are
    averaged across FSDP ranks. This is needed for verl's loss scaling (loss * dp_size)
    to work correctly.

    Args:
        model: The model (or model part) to enable gradient division on.
        dp_size: The data parallel size to use as the gradient divide factor.
    """

    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(float(dp_size))


@contextmanager
def spmd_no_typecheck_context():
    """Disable ``spmd_types`` type-checking when ``spmd_types`` is installed and active."""
    try:
        import spmd_types as spmd
    except ImportError:
        yield
        return
    with spmd.no_typecheck():
        yield


class ContextParallelGather(torch.autograd.Function):
    """All-gather along dim 0 whose backward keeps only the local slice.

    Every CP rank computes the same loss on the gathered tensor, so the upstream gradient is
    identical across CP ranks and its local slice is already the full gradient of the local shard.
    Summing it across ranks (the default all-gather backward) would scale gradients by the CP size.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: torch.distributed.ProcessGroup) -> torch.Tensor:
        with spmd_no_typecheck_context():
            ctx.rank = torch.distributed.get_rank(group)
            ctx.world_size = torch.distributed.get_world_size(group)
            tensor = tensor.contiguous()
            gathered = tensor.new_empty((tensor.size(0) * ctx.world_size, *tensor.shape[1:]))
            torch.distributed.all_gather_into_tensor(gathered, tensor, group=group)
            return gathered

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        with spmd_no_typecheck_context():
            return grad_output.chunk(ctx.world_size, dim=0)[ctx.rank], None


def extend_torchtitan_engine_config() -> None:
    """Register TPU-specific optional fields on ``TorchtitanEngineConfig`` for Hydra/OmegaConf."""
    extras: list[tuple[str, Any, Any]] = [
        ("use_splash_attention", Optional[bool], None),
        ("use_simple_fsdp", Optional[bool], None),
        ("tpu_eager_mode", Optional[str], None),
        ("pad_to_length", bool, False),
        ("pad_to_length_bucket", int, _DEFAULT_TPU_SEQ_BUCKET_SIZE),
    ]
    missing = [item for item in extras if item[0] not in TorchtitanEngineConfig.__dataclass_fields__]
    if not missing:
        return
    # Re-attach existing Field objects whose class attributes were removed by @dataclass
    # (e.g., fields with default_factory such as wrap_policy) before re-running dataclass().
    for fname, fobj in list(TorchtitanEngineConfig.__dataclass_fields__.items()):
        if fname in TorchtitanEngineConfig.__annotations__:
            setattr(TorchtitanEngineConfig, fname, fobj)
    for name, typ, default in missing:
        TorchtitanEngineConfig.__annotations__[name] = typ
        setattr(TorchtitanEngineConfig, name, default)
    for fn in ("__init__", "__repr__", "__eq__"):
        if fn in TorchtitanEngineConfig.__dict__:
            delattr(TorchtitanEngineConfig, fn)
    dataclass(TorchtitanEngineConfig)


extend_torchtitan_engine_config()


def _parse_optional_bool_env(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    norm = raw.strip().lower()
    if norm in ("1", "true", "yes", "on"):
        return True
    if norm in ("0", "false", "no", "off"):
        return False
    return None


def resolve_tpu_torchtitan_options(engine_config: Any) -> tuple[bool, bool, Optional[str]]:
    """Resolve ``(use_splash_attention, use_simple_fsdp, tpu_eager_mode)`` for TPU TorchTitan.

    Precedence for each option:
    1. Explicit environment variable override (``VERL_TPU_USE_SPLASH_ATTENTION``,
       ``VERL_TPU_USE_SIMPLE_FSDP``, ``VERL_TPU_EAGER_MODE``).
    2. Explicit ``engine_config`` field value when non-``None``.
    3. Automatic TPU defaults based on ``engine_config.use_torch_compile``:
       when ``use_torch_compile=True``, enables ``TPUSplashAttention``, ``SimpleFSDP``
       (for pure FSDP/HSDP where TP/PP/CP/EP == 1), and ``DEFER_AND_FUSE``.
    """
    use_torch_compile = bool(getattr(engine_config, "use_torch_compile", False))
    cp_enabled = int(getattr(engine_config, "context_parallel_size", 1) or 1) > 1

    splash_env = _parse_optional_bool_env("VERL_TPU_USE_SPLASH_ATTENTION")
    splash_cfg = getattr(engine_config, "use_splash_attention", None)
    if splash_env is not None:
        use_splash_attention = splash_env
    elif splash_cfg is not None:
        use_splash_attention = bool(splash_cfg)
    else:
        use_splash_attention = use_torch_compile or cp_enabled

    if use_torch_compile and not use_splash_attention:
        raise ValueError(
            "use_torch_compile=True on TPU requires use_splash_attention=True: torch_tpu's compiled "
            "F.scaled_dot_product_attention produces NaN q/k/v gradients, so every optimizer step would be "
            "skipped. Use splash attention + torch.compile (use_splash_attention=True), or set "
            "use_torch_compile=False."
        )

    pure_fsdp = (
        getattr(engine_config, "tensor_parallel_size", 1) == 1
        and getattr(engine_config, "pipeline_parallel_size", 1) == 1
        and getattr(engine_config, "expert_parallel_size", 1) == 1
    )
    simple_fsdp_env = _parse_optional_bool_env("VERL_TPU_USE_SIMPLE_FSDP")
    simple_fsdp_cfg = getattr(engine_config, "use_simple_fsdp", None)
    if simple_fsdp_env is not None:
        use_simple_fsdp = simple_fsdp_env
    elif simple_fsdp_cfg is not None:
        use_simple_fsdp = bool(simple_fsdp_cfg)
    else:
        use_simple_fsdp = bool(use_torch_compile and pure_fsdp)

    if use_simple_fsdp and not pure_fsdp:
        raise ValueError("use_simple_fsdp supports pure FSDP/HSDP only (tensor/pipeline/expert parallel size 1)")

    eager_env = os.environ.get("VERL_TPU_EAGER_MODE")
    eager_cfg = getattr(engine_config, "tpu_eager_mode", None)
    if eager_env is not None and eager_env.strip() != "":
        norm_eager = eager_env.strip().upper()
        tpu_eager_mode = None if norm_eager in ("NONE", "NULL") else norm_eager
    elif eager_cfg is not None:
        norm_cfg = str(eager_cfg).strip().upper()
        tpu_eager_mode = None if norm_cfg in ("", "NONE", "NULL") else norm_cfg
    else:
        tpu_eager_mode = "DEFER_AND_FUSE" if use_torch_compile else None

    if tpu_eager_mode not in (None, "DEFER_AND_FUSE", "DEFER_NEVER"):
        raise ValueError(f"tpu_eager_mode {tpu_eager_mode!r} not supported")

    return use_splash_attention, use_simple_fsdp, tpu_eager_mode


_PARALLEL_DIMS_CACHE: dict[tuple[Any, ...], Any] = {}
_PARALLEL_DIMS_PATCHED = False


def install_parallel_dims_cache_for_tpu() -> None:
    """Reuse ``ParallelDims`` and its ``DeviceMesh`` objects across engines in the same process.

    ``ActorRolloutRefWorker`` initializes both ``actor`` and ``ref`` ``TorchTitanTPUEngine`` instances
    in the same worker process. TorchTitan's ``ParallelDims`` caches ``DeviceMesh`` instances on ``self``
    because Dynamo traces ``DeviceMesh`` as a ``FakeScriptObject`` matched by object identity, whereas
    ``DTensor``'s ``ShardingPropagator`` LRU cache hashes ``DeviceMesh`` by value. Caching
    ``ParallelDims.from_config`` and making ``ParallelDims.build_mesh`` idempotent ensures ``actor`` and
    ``ref`` share the exact same ``DeviceMesh`` Python instances and process groups.
    """
    global _PARALLEL_DIMS_PATCHED
    if _PARALLEL_DIMS_PATCHED:
        return
    try:
        from torchtitan.distributed.parallel_dims import ParallelDims
    except ImportError:
        return

    orig_from_config = getattr(ParallelDims.from_config, "__func__", None)
    orig_build_mesh = getattr(ParallelDims, "build_mesh", None)
    if orig_from_config is None or orig_build_mesh is None:
        return

    @classmethod  # type: ignore[misc]
    def _cached_from_config(cls, parallelism_config: Any, world_size: int) -> Any:
        world_pg_id = (
            id(torch.distributed.group.WORLD)
            if torch.distributed.is_initialized() and torch.distributed.group.WORLD is not None
            else None
        )
        key = (
            world_pg_id,
            world_size,
            dataclasses.astuple(parallelism_config)
            if dataclasses.is_dataclass(parallelism_config)
            else id(parallelism_config),
        )
        cached = _PARALLEL_DIMS_CACHE.get(key)
        if cached is None:
            cached = orig_from_config(cls, parallelism_config, world_size)
            _PARALLEL_DIMS_CACHE[key] = cached
        return cached

    def _cached_build_mesh(self: Any) -> Any:
        if getattr(self, "_world_mesh", None) is not None and getattr(self, "_single_axis_meshes", None):
            return self._world_mesh
        return orig_build_mesh(self)

    ParallelDims.from_config = _cached_from_config
    ParallelDims.build_mesh = _cached_build_mesh
    _PARALLEL_DIMS_PATCHED = True


def configure_torch_compile_for_tpu(recompile_limit: int = 64) -> None:
    """Configure Dynamo for per-TransformerBlock ``torch.compile(backend="tpu")``.

    verl feeds packed sequences whose length is bucketed to a multiple of
    ``VERL_TPU_SEQ_BUCKET_SIZE``, so each block sees a small set of static shapes.
    - ``automatic_dynamic_shapes=False``: after the second distinct shape Dynamo
      would otherwise retrace with symbolic shapes, which the XLA backend handles
      badly (padding + recompiles, no fusion). Keep every bucket static.
    - ``recompile_limit``: one graph per (bucket, grad-mode) pair; the default
      limit (8) is hit quickly and then Dynamo fails under ``fullgraph=True``.

    Note: ``torch._dynamo.config`` stores overrides in a ``ContextVar``, so values set
    during ``init_model`` do not carry over to later Ray actor RPCs unless
    ``_config[name].default`` is updated too.
    """
    entries = getattr(torch._dynamo.config, "_config", {})

    def _set(name: str, value: Any) -> None:
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, value)
        if name in entries:
            entries[name].default = value

    _set("automatic_dynamic_shapes", False)
    _set("assume_static_by_default", True)
    _set("capture_scalar_outputs", True)
    _set("skip_fwd_side_effects_in_bwd_under_checkpoint", True)
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            _set(name, max(getattr(torch._dynamo.config, name), recompile_limit))
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        _set(
            "accumulated_recompile_limit",
            max(torch._dynamo.config.accumulated_recompile_limit, recompile_limit * 16),
        )


def splash_block_size_for(seq_len: int, max_block_size: int = 512, min_block_size: int = 128) -> int:
    """Largest power-of-two splash block size <= ``max_block_size`` that divides ``seq_len``.

    The splash kernel requires every block size to divide the sequence length. verl pads
    packed sequences to a multiple of ``VERL_TPU_SEQ_BUCKET_SIZE`` (256), so a fixed 512
    block (torchtitan's default) fails for e.g. 768 tokens. Returns 0 if no block
    >= ``min_block_size`` divides ``seq_len``.
    """
    block = max_block_size
    while block >= min_block_size:
        if seq_len % block == 0:
            return block
        block //= 2
    return 0


def get_tpu_seq_bucket_size(engine_config: Any = None) -> int:
    """Return the sequence length bucket multiple used on TPU to bound XLA compilations.

    Precedence:
    1. ``VERL_TPU_SEQ_BUCKET_SIZE`` environment variable when set and positive.
    2. ``engine_config.pad_to_length_bucket`` when ``engine_config.pad_to_length`` is enabled.
    3. ``_DEFAULT_TPU_SEQ_BUCKET_SIZE`` (256).
    """
    env_raw = os.environ.get("VERL_TPU_SEQ_BUCKET_SIZE")
    bucket = _DEFAULT_TPU_SEQ_BUCKET_SIZE
    if env_raw is not None and env_raw.strip() != "":
        try:
            val = int(env_raw)
            if val > 0:
                bucket = val
        except ValueError:
            pass
    elif engine_config is not None and bool(getattr(engine_config, "pad_to_length", False)):
        cfg_bucket = getattr(engine_config, "pad_to_length_bucket", None)
        if isinstance(cfg_bucket, int) and cfg_bucket > 0:
            bucket = cfg_bucket
    if engine_config is not None:
        cp_size = int(getattr(engine_config, "context_parallel_size", 1) or 1)
        if cp_size > 1:
            bucket = math.lcm(bucket, 2 * cp_size)
    return bucket


def bucket_length(length: int, bucket_size: Optional[int] = None) -> int:
    """Round ``length`` up to the nearest positive multiple of ``bucket_size``."""
    step = bucket_size if bucket_size is not None and bucket_size > 0 else get_tpu_seq_bucket_size()
    length = max(int(length), 1)
    return ((length + step - 1) // step) * step


def unwrap_metadata(val: Any) -> Any:
    """Unwrap a singleton tensor or per-rank list/tuple metadata value to a Python scalar."""
    if isinstance(val, list | tuple):
        val = val[0] if len(val) > 0 else val
    if isinstance(val, torch.Tensor):
        return val.item() if val.numel() == 1 else val
    return val


def tpu_synchronize(
    tensors: Optional[torch.Tensor] = None,
    *,
    wait: bool = True,
    strict: bool = False,
) -> None:
    """Materialize and synchronize pending TPU operations (for ``tensors`` or the whole device).

    Args:
        tensors: Optional tensor (e.g. ``loss``) to materialize; ``None`` synchronizes the whole TPU device.
        wait: Wait for device execution to complete when ``True``; dispatch asynchronously when ``False``.
        strict: Raise ``RuntimeError`` if ``torch_tpu`` cannot synchronize instead of logging a warning.
    """
    if isinstance(tensors, torch.Tensor) and tensors.device.type != "tpu":
        return
    try:
        from torch_tpu._internal import sync as torch_tpu_sync

        if tensors is None:
            torch_tpu_sync.synchronize(wait=wait)
        else:
            torch_tpu_sync.synchronize(tensors, wait=wait)
        return
    except ImportError as e:
        if strict:
            raise RuntimeError(f"TPU synchronization failed: {e}") from e
    except Exception as e:
        if strict:
            raise RuntimeError(f"TPU synchronization failed: {e}") from e
        logger.warning(f"Could not synchronize via torch_tpu: {e}")
        return

    tpu_mod = getattr(torch, "tpu", None)
    if tpu_mod is not None and hasattr(tpu_mod, "synchronize"):
        try:
            tpu_mod.synchronize()
        except Exception as e:
            if strict:
                raise RuntimeError(f"TPU synchronization failed: {e}") from e
            logger.debug("torch.tpu.synchronize() failed during TPU sync: %s", e)


def tpu_eager_mode_context(mode: Optional[str]):
    """Context manager running the enclosed ops under the ``torch_tpu`` eager ``mode`` (no-op for ``None``).

    Mirrors torchtitan's ``tpu_config.eager_mode`` (``torchtitan/experiments/tpu/gmain.py``), which TPU
    recipes set to ``DEFER_AND_FUSE``: ops outside the compiled blocks are deferred and compiled into
    fused XLA programs at the next materialization point instead of being launched one XLA program per op
    (the ``torch_tpu`` default, ``DEFER_NEVER``).
    """
    if mode is None:
        return nullcontext()
    try:
        from torch_tpu._internal import execution_mode

        return execution_mode.set_eager_mode(getattr(execution_mode.EagerMode, mode))
    except ImportError:
        return nullcontext()


def compute_global_batch_num_tokens(data: TensorDict, dp_group: Any, tp_size: int, cp_size: int = 1) -> Any:
    """Compute the global batch token count for loss normalization on TPU via CPU all-reduce."""
    batch_num_tokens = data["loss_mask"].sum().cpu()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
        batch_num_tokens = batch_num_tokens / (tp_size * cp_size)
    return batch_num_tokens.item()


def align_micro_batch_shapes_across_ranks(micro_batches: list[TensorDict], bucket_size: Optional[int] = None) -> None:
    """Make every rank pad micro-batch ``i`` to the same packed length and response-length bucket.

    SimpleFSDP traces the all-gather/reduce-scatter into each compiled (``spmd_safe``) TransformerBlock, and
    collectives inside a program require every participant to run the same program: mismatched shapes halt
    the TPU (``sync_flag_public_access_error``). One CPU all-reduce(MAX) over all micro-batches' buckets gives
    each micro-batch a common ``tpu_padded_seq_len`` / ``tpu_max_response_len`` that
    :func:`pad_packed_inputs_for_tpu` pads up to.
    """
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return
    step = bucket_size if bucket_size is not None and bucket_size > 0 else get_tpu_seq_bucket_size()
    lens: list[int] = []
    for mb in micro_batches:
        ids = mb["input_ids"]
        n_tokens = ids.values().numel() if getattr(ids, "is_nested", False) else ids.numel()
        resp = mb["responses"] if "responses" in mb.keys() else None
        max_resp = (
            int(resp.offsets().diff().max().item()) if resp is not None and getattr(resp, "is_nested", False) else 0
        )
        lens += [bucket_length(n_tokens, step), bucket_length(max_resp, step) if max_resp else 0]
    lens_tensor = torch.tensor(lens, dtype=torch.int64)
    torch.distributed.all_reduce(lens_tensor, op=torch.distributed.ReduceOp.MAX)
    resolved_lens = lens_tensor.tolist()
    for i, mb in enumerate(micro_batches):
        tu.assign_non_tensor_data(mb, "tpu_padded_seq_len", int(resolved_lens[2 * i]))
        tu.assign_non_tensor_data(mb, "tpu_max_response_len", int(resolved_lens[2 * i + 1]))


def get_tpu_simple_fsdp_parallelize_fn(torchtitan_name: str) -> Callable[..., nn.Module]:
    """Return torchtitan's TPU recipe ``parallelize_fn`` for ``torchtitan_name``; it shards with SimpleFSDP.

    With ``ParallelismConfig.use_simple_fsdp`` set, the recipe (``torchtitan/experiments/tpu/<model>/infra/
    parallelize.py``) wraps the TransformerBlocks with ``spmd_safe``, applies AC and per-block ``torch.compile``
    through the model's own ``parallelize_fn`` (``skip_dp=True``), picks shard dims that divide evenly, and
    shards every parameter with ``graph_trainer``'s SimpleFSDP. Under ``torch.compile`` the FSDP collectives
    live inside each compiled TransformerBlock.
    """
    try:
        module_name, fn_name = _TPU_SIMPLE_FSDP_RECIPES[torchtitan_name]
    except KeyError:
        raise ValueError(
            f"use_simple_fsdp=True needs a torchtitan TPU recipe, and there is none for {torchtitan_name!r} "
            f"(supported: {sorted(_TPU_SIMPLE_FSDP_RECIPES)}); set use_simple_fsdp=False"
        ) from None
    recipe_parallelize_fn = getattr(importlib.import_module(module_name), fn_name)

    def _parallelize(model: nn.Module, **kwargs: Any) -> nn.Module:
        model = recipe_parallelize_fn(model, **kwargs)
        init_states_without_simple_fsdp_parametrization(model)
        return model

    return _parallelize


def init_states_without_simple_fsdp_parametrization(model: nn.Module) -> None:
    """Make ``model.init_states`` run with SimpleFSDP's parametrization disabled.

    torchtitan's ``Trainer`` parallelizes first and initializes second. While the parametrization is active,
    reading ``.weight`` returns the all-gathered plain tensor, so ``Decoder.init_states`` re-tying
    ``tok_embeddings.weight = lm_head.weight`` (tied embeddings, e.g. Qwen3 0.6B-4B) raises ``TypeError: cannot
    assign ... as parameter 'weight'``. With it disabled, the re-tie points both modules at the same sharded
    parameter (SimpleFSDP shards them separately). torchtitan's graph trainer and HF backend models initialize
    this way; its TPU qwen3 / llama3 recipes do not yet.
    """
    from torchtitan.experiments.graph_trainer import simple_fsdp

    init_states = model.init_states

    def _init_states(*args: Any, **kwargs: Any) -> None:
        # disable_active_parametrization() is not reentrant, so only take it while the parametrization is active.
        active = simple_fsdp._active_parametrization
        with simple_fsdp.disable_active_parametrization() if active else nullcontext():
            init_states(*args, **kwargs)

    model.init_states = _init_states  # type: ignore[method-assign]


class TPUVarlenAttention(torch.nn.Module):
    """Packed-document causal attention module for eager TPU execution."""

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: Any = None,
        scale: float | None = None,
        out_transform: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if out_transform is not None:
            # TorchTitan applies out_transform as an epilogue over (out_BLNH, lse_BLN); see
            # torchtitan/models/common/attention.py. Attention-sink models such as gpt-oss rely
            # on it. This module does not produce an LSE, so silently ignoring out_transform
            # would return numerically wrong attention output.
            raise NotImplementedError(
                "TPUVarlenAttention does not support the out_transform epilogue used by "
                "attention-sink models (e.g. gpt-oss). Supported models must not set it."
            )

        was_3d = q_BLNH.dim() == 3
        if was_3d:
            xq, xk, xv = q_BLNH.unsqueeze(0), k_BLNH.unsqueeze(0), v_BLNH.unsqueeze(0)
        else:
            xq, xk, xv = q_BLNH, k_BLNH, v_BLNH

        # Probe the value, not just the attribute name: a mask object can declare the attribute
        # and still leave it unset, in which case the packed path is not applicable.
        cu_seqs = None
        for attr in ("cu_seq_q", "cu_seqlens_q", "cu_seqlens"):
            candidate = getattr(attention_masks, attr, None)
            if candidate is not None:
                cu_seqs = candidate
                break

        if cu_seqs is not None:
            total_tokens = xq.shape[1]
            positions = torch.arange(total_tokens, device=xq.device)
            seq_indices = (positions.unsqueeze(1) >= cu_seqs.unsqueeze(0)).sum(dim=1) - 1
            same_seq_mask = seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)
            causal_mask = positions.unsqueeze(1) >= positions.unsqueeze(0)
            mask = (same_seq_mask & causal_mask).unsqueeze(0).unsqueeze(0)
        elif isinstance(attention_masks, torch.Tensor):
            if attention_masks.dim() == 4:
                mask = attention_masks.to(torch.bool)
            else:
                seq_len = xq.shape[1]
                positions = torch.arange(seq_len, device=xq.device)
                causal_mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                padding_mask = attention_masks.unsqueeze(1).unsqueeze(2).to(torch.bool)
                mask = causal_mask & padding_mask
        else:
            seq_len = xq.shape[1]
            positions = torch.arange(seq_len, device=xq.device)
            mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        if q.shape[1] != k.shape[1]:
            num_repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        out = attn_out.transpose(1, 2)
        return out.squeeze(0) if was_3d else out


class TPUSplashAttention(torch.nn.Module):
    """TorchTitan inner attention backed by the TPU splash attention Pallas kernel.

    Accepts 3D ``(T, H, D)`` (folded token stream) or 4D ``(B, S, H, D)`` inputs, applying
    causal + packed-document masking from ``segment_ids``. Block sizes are chosen per call
    from the (static, bucketed) sequence length instead of being fixed at 512. Falls back
    to the original module if no block size fits.
    """

    def __init__(self, original_module: torch.nn.Module, local_window_size: int | None = None):
        super().__init__()
        self.original_module = original_module
        self.local_window_size = local_window_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        attention_masks: Any = None,
        positions: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        seq_len = int(q.shape[0] if q.dim() == 3 else q.shape[1])
        block = splash_block_size_for(seq_len)
        if block == 0 or q.device.type == "cpu":
            return self.original_module(
                q,
                k,
                v,
                scale=scale,
                enable_gqa=enable_gqa,
                attention_masks=attention_masks,
                positions=positions,
                segment_ids=segment_ids,
                **kwargs,
            )

        from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa
        from torchtitan.models.common.attention import segment_ids_from_positions

        q_mesh = q_placements = None
        if isinstance(q, DTensor):
            q_mesh, q_placements = q.device_mesh, q.placements
            q = q.to_local()
        k = k.to_local() if isinstance(k, DTensor) else k
        v = v.to_local() if isinstance(v, DTensor) else v
        positions = positions.to_local() if isinstance(positions, DTensor) else positions
        segment_ids = segment_ids.to_local() if isinstance(segment_ids, DTensor) else segment_ids
        if segment_ids is None and positions is not None:
            segment_ids = segment_ids_from_positions(positions)

        was_3d = q.dim() == 3
        if was_3d:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        if segment_ids is not None and segment_ids.dim() == 1:
            segment_ids = segment_ids.unsqueeze(0)

        out = splash_sdpa(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            segment_ids=segment_ids,
            scale=scale,
            is_causal=True,
            local_window_size=self.local_window_size,
            enable_gqa=enable_gqa,
            block_q=block,
            block_kv=block,
            block_dkv=block,
            block_kv_compute=block,
            block_q_dkv=block,
            block_kv_dkv=block,
            block_kv_dkv_compute=block,
        ).transpose(1, 2)
        if was_3d:
            out = out.squeeze(0)
        if q_mesh is not None:
            out = DTensor.from_local(out, q_mesh, q_placements)
        return out


def apply_splash_attention_tpu(modules: Any) -> int:
    """Swap every attention module's ``inner_attention`` for ``TPUSplashAttention``.

    The splash kernel takes the per-token ``segment_ids`` that ``Decoder.forward`` derives
    from ``positions`` (which restart at 0 for every packed sequence), so it applies the
    same causal + document-boundary mask as the dense [1, 1, S, S] mask built in
    ``pad_packed_inputs_for_tpu`` without materializing it. Bucket padding tokens have
    position 0, so each one becomes its own segment and only attends to itself; their
    outputs are discarded.

    Must run before the first forward: per-block ``torch.compile`` traces lazily, so the
    swapped module is what gets compiled. Returns the number of replaced modules.
    """
    if modules is None:
        return 0
    from torchtitan.models.common.attention import VarlenInnerAttention

    model_list = modules if isinstance(modules, list | tuple | torch.nn.ModuleList) else [modules]
    replaced = 0
    already_splash = 0
    saw_nn_module = False
    for model in model_list:
        if not isinstance(model, torch.nn.Module):
            continue
        saw_nn_module = True
        for module in model.modules():
            inner = getattr(module, "inner_attention", None)
            if not isinstance(inner, torch.nn.Module):
                continue
            has_local_attn_hook = callable(getattr(inner, "set_local_attention", None))
            local_attn_impl = getattr(inner, "_local_attn_override", None) or getattr(inner, "_local_attention", None)
            if isinstance(inner, TPUSplashAttention) or (
                has_local_attn_hook and isinstance(local_attn_impl, TPUSplashAttention)
            ):
                already_splash += 1
                continue
            window = getattr(inner, "window_size", None)
            local_window_size = None
            if isinstance(window, tuple) and len(window) == 2 and window[0] >= 0:
                local_window_size = int(window[0])
            # torchtitan's VarlenInnerAttention calls the CUDA flash-attention varlen op, which TPU lacks.
            fallback = (
                TPUVarlenAttention()
                if isinstance(inner, VarlenInnerAttention)
                else getattr(inner, "local_forward", inner)
            )
            splash = TPUSplashAttention(fallback, local_window_size=local_window_size)
            if has_local_attn_hook:
                inner.set_local_attention(splash)
            else:
                module.inner_attention = splash
            replaced += 1
    if saw_nn_module and replaced == 0 and already_splash == 0:
        raise ValueError("use_splash_attention=True but no module with an `inner_attention` was found")
    if replaced:
        logger.warning("Splash attention enabled on TPU: replaced %d inner attention modules", replaced)
    return replaced


def replace_varlen_attention_with_tpu_attention(modules: Any) -> int:
    """Replaces inner_attention submodules on model instances with TPUVarlenAttention."""
    model_list = modules if isinstance(modules, list | tuple | torch.nn.ModuleList) else [modules]
    replaced = 0
    for root in model_list:
        if not isinstance(root, torch.nn.Module):
            continue
        for sub in root.modules():
            if not hasattr(sub, "inner_attention"):
                continue
            inner = sub.inner_attention
            if callable(getattr(inner, "set_local_attention", None)):
                local_impl = getattr(inner, "_local_attn_override", None) or getattr(inner, "_local_attention", None)
                if not isinstance(local_impl, TPUVarlenAttention | TPUSplashAttention):
                    inner.set_local_attention(TPUVarlenAttention())
                    replaced += 1
            elif not isinstance(inner, TPUVarlenAttention | TPUSplashAttention):
                sub.inner_attention = TPUVarlenAttention()
                replaced += 1
    if replaced:
        logger.info("Replaced %d inner_attention submodule(s) with TPUVarlenAttention.", replaced)
    return replaced


def pad_packed_inputs_for_tpu(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    micro_batch: TensorDict,
    device: Any,
    build_attention_mask: bool = True,
    bucket_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """Bucket-pad packed 1D sequences and optionally build a static 4D causal document mask on CPU before H2D transfer.

    With ``build_attention_mask=False`` (splash attention, which derives document boundaries
    from ``positions``) the dense mask is skipped and ``None`` is returned in its place.
    """
    step = bucket_size if bucket_size is not None and bucket_size > 0 else get_tpu_seq_bucket_size()
    input_ids_cpu = input_ids.values().detach().cpu()
    position_ids_cpu = position_ids.values().detach().cpu()

    labels_cpu = torch.roll(input_ids_cpu, shifts=-1, dims=0)

    orig_seq_len = int(input_ids_cpu.shape[-1])
    # tpu_padded_seq_len: common length across ranks from align_micro_batch_shapes_across_ranks (SimpleFSDP).
    padded_seq_len = max(
        bucket_length(orig_seq_len, step),
        int(tu.get_non_tensor_data(data=micro_batch, key="tpu_padded_seq_len", default=0) or 0),
    )
    pad_len = padded_seq_len - orig_seq_len

    pos_2d_cpu = position_ids_cpu[0].unsqueeze(0) if position_ids_cpu.dim() == 2 else position_ids_cpu.unsqueeze(0)
    if pad_len > 0:
        input_ids_cpu = F.pad(input_ids_cpu, (0, pad_len), value=0)
        labels_cpu = F.pad(labels_cpu, (0, pad_len), value=0)
        position_ids_cpu = F.pad(position_ids_cpu, (0, pad_len), value=0)
        pos_2d_cpu = F.pad(pos_2d_cpu, (0, pad_len), value=0)

    attention_mask_cpu = None
    if build_attention_mask:
        if getattr(input_ids, "is_nested", False):
            seq_lens = input_ids.offsets().diff().detach().cpu()
            seq_ids_1d = torch.repeat_interleave(torch.arange(1, len(seq_lens) + 1, dtype=torch.int64), seq_lens)
            if pad_len > 0:
                seq_ids_1d = F.pad(seq_ids_1d, (0, pad_len), value=0)
            seq_ids = seq_ids_1d.unsqueeze(0)
        else:
            first_dummy = pos_2d_cpu[:, :1] - 1
            boundary = torch.diff(pos_2d_cpu, prepend=first_dummy, dim=-1) != 1
            boundary[:, 0] = True
            seq_ids = boundary.cumsum(dim=-1)

        idx = torch.arange(padded_seq_len, dtype=seq_ids.dtype).unsqueeze(0)
        valid = idx < orig_seq_len
        seq_ids = torch.where(valid, seq_ids, -idx - 1)
        same_seq_mask = seq_ids.unsqueeze(2) == seq_ids.unsqueeze(1)
        causal_mask = idx.unsqueeze(2) >= idx.unsqueeze(1)
        attention_mask_cpu = (same_seq_mask & causal_mask).unsqueeze(1)

    if "responses" in micro_batch.keys() and getattr(micro_batch["responses"], "is_nested", False):
        resp_lens = micro_batch["responses"].offsets().diff().cpu()
        raw_max_resp = int(resp_lens.max().item())
        max_resp = max(
            bucket_length(raw_max_resp, step),
            int(tu.get_non_tensor_data(data=micro_batch, key="tpu_max_response_len", default=0) or 0),
        )
        tu.assign_non_tensor_data(micro_batch, "max_response_len", max_resp)

    return (
        input_ids_cpu.to(device=device).contiguous(),
        position_ids_cpu.to(device=device).contiguous(),
        labels_cpu.to(device=device).contiguous(),
        attention_mask_cpu.to(device=device).contiguous() if attention_mask_cpu is not None else None,
        orig_seq_len,
    )


class _TPUPaddedJaggedProxy:
    """Lightweight proxy exposing a 1D bucket-padded TPU tensor via ``.values()``."""

    def __init__(self, padded_values: torch.Tensor, base_nested: Any):
        self._padded_values = padded_values
        self._base_nested = base_nested
        self.is_nested = True

    def values(self) -> torch.Tensor:
        return self._padded_values

    def offsets(self) -> torch.Tensor:
        return self._base_nested.offsets()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_nested, name)


class _TPULossDataProxy:
    """Read-only view over a micro-batch ``TensorDict`` that overrides specific keys for loss calculation."""

    def __init__(self, base: TensorDict, overrides: dict[str, Any]):
        self._base = base
        self._overrides = overrides

    def __getitem__(self, key: str) -> Any:
        if key in self._overrides:
            return self._overrides[key]
        return self._base[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._overrides:
            return self._overrides[key]
        return self._base.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._overrides or key in self._base

    def keys(self):
        return self._base.keys()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _copy_tensordict_without_device(td: TensorDict) -> TensorDict:
    """Shallow-copy ``td`` (entries shared, host metadata kept) into a TensorDict with no device.

    ``TensorDict(td, batch_size=td.batch_size)`` cannot be used for this: on tensordict >= 0.11 it re-validates
    every ``NonTensorData`` entry (``dp_size``, ``batch_num_tokens``, ...) as a 0-d value and raises
    ``batch dimension mismatch``. Without a device, TPU tensors can be stored next to the CPU metadata.
    """
    out = td.copy()
    if out.device is not None:
        out.clear_device_()
    return out


def _response_gather_index(
    micro_batch: TensorDict, response_width: int, padded_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build, on CPU, the index/mask that gather every response's per-token outputs from a packed sequence.

    Mirrors verl's ``no_padding_2_padding`` convention: response token ``r`` of sequence ``i`` is predicted at packed
    position ``seq_end_i - resp_len_i - 1 + r``. Returns ``index`` (int64) and ``valid`` (bool), both shaped
    ``[batch_size, response_width]``; padded slots point at position 0 and are masked out.
    """
    prompt_lens = micro_batch["prompts"].offsets().diff().cpu()
    response_lens = micro_batch["responses"].offsets().diff().cpu()
    starts = ((prompt_lens + response_lens).cumsum(dim=0) - response_lens - 1).to(torch.int64)
    cols = torch.arange(response_width, dtype=torch.int64).unsqueeze(0)
    valid = cols < response_lens.unsqueeze(1)
    index = torch.where(valid, starts.unsqueeze(1) + cols, torch.zeros_like(cols)).clamp(0, max(0, padded_len - 1))
    return index, valid


def _gather_responses_static_layout(
    padded_values: torch.Tensor, index: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Gather responses out of the bucket-padded packed tensor into a static ``[bsz * (width + 1)]`` layout.

    Row ``i`` of the ``[bsz, width]`` gather is laid out as one segment of ``width + 1`` slots (responses first,
    one trailing pad slot), so a sequence with prompt length 1 and response length ``width`` selects exactly that
    row in ``no_padding_2_padding``. Every shape depends only on the bucketed lengths, never on the token counts.
    """
    device = padded_values.device
    rows = torch.gather(padded_values.unsqueeze(0).expand(index.shape[0], -1), 1, index.to(device=device))
    rows = rows * valid.to(device=device, dtype=padded_values.dtype)
    return F.pad(rows, (0, 1)).reshape(-1)


def prepare_tpu_loss_inputs(model_output: dict[str, Any], micro_batch: TensorDict) -> tuple[dict[str, Any], Any]:
    """Adapt differentiable TPU outputs for the unmodified SFT and PPO loss APIs.

    SFT consumes full bucket-padded buffers and a matching loss mask. PPO receives a real TensorDict with padded
    response fields on device. Its model outputs are gathered (static gather, CPU-built indices) from the full
    bucket-padded device tensors into a static layout, so ``ppo_loss`` -> ``no_padding_2_padding`` only ever sees
    shapes that depend on the sequence-length bucket, the micro-batch size and the response-length bucket. Trimming
    the outputs to the exact token count instead would make every micro-batch with new lengths compile new XLA
    programs (forward loss and the fused LM-head backward) for the whole run.
    """
    log_prob = model_output.get("log_probs")
    tpu_padded_log_prob = getattr(log_prob, TPU_PADDED_VALUES_ATTR, None)
    if tpu_padded_log_prob is None:
        return model_output, micro_batch

    if "old_log_probs" in micro_batch and "advantages" in micro_batch:
        # Preserve host sequence metadata while moving response fields independently
        # of the source TensorDict's device setting.
        loss_data = _copy_tensordict_without_device(micro_batch)
        responses = micro_batch["responses"]
        if micro_batch["prompts"].is_nested:
            response_width = int(unwrap_metadata(tu.get_non_tensor_data(micro_batch, "max_response_len", -1)))
            aligned_width = int(unwrap_metadata(tu.get_non_tensor_data(micro_batch, "tpu_max_response_len", 0)) or 0)
            actual_width = int(responses.offsets().diff().max().item())
            if response_width < 0:
                response_width = actual_width
            response_width = max(response_width, aligned_width)
            if response_width < actual_width:
                raise ValueError("max_response_len must accommodate every response")
        else:
            response_width = responses.shape[1]

        fields = ["response_mask", "old_log_probs", "advantages"]
        fields.extend(key for key in ("ref_log_prob", "rollout_is_weights") if key in micro_batch)
        padded_fields = micro_batch.select(*fields).to_padded_tensor()
        if micro_batch["prompts"].is_nested:
            # Legacy batches can retain response fields wider than the packed response bucket.
            response_width = max(response_width, *(tensor.shape[1] for tensor in padded_fields.values()))
            tu.assign_non_tensor_data(loss_data, "max_response_len", response_width)
        for key, tensor in padded_fields.items():
            if tensor.shape[1] > response_width:
                raise ValueError(f"{key} exceeds max_response_len ({tensor.shape[1]} > {response_width})")
            loss_data[key] = F.pad(tensor, (0, response_width - tensor.shape[1])).to(tpu_padded_log_prob.device)

        loss_model_output = dict(model_output)
        padded_keys = [
            key for key, value in model_output.items() if getattr(value, TPU_PADDED_VALUES_ATTR, None) is not None
        ]
        static_layout = (
            micro_batch["prompts"].is_nested
            and getattr(responses, "is_nested", False)
            and all(getattr(model_output[key], TPU_PADDED_VALUES_ATTR).dim() == 1 for key in padded_keys)
            # every packed output the loss slices must come with its bucket-padded device tensor
            and not any(
                getattr(value, "is_nested", False) and key not in padded_keys for key, value in model_output.items()
            )
        )
        if static_layout:
            batch_size = int(micro_batch.batch_size[0])
            segment = response_width + 1
            index_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            layout = torch.nested.nested_tensor_from_jagged(
                torch.zeros(batch_size * segment), torch.arange(batch_size + 1, dtype=torch.int64) * segment
            )
            for key in padded_keys:
                padded_values = getattr(model_output[key], TPU_PADDED_VALUES_ATTR)
                padded_len = int(padded_values.shape[0])
                if padded_len not in index_cache:
                    index_cache[padded_len] = _response_gather_index(micro_batch, response_width, padded_len)
                index, valid = index_cache[padded_len]
                loss_model_output[key] = _TPUPaddedJaggedProxy(
                    _gather_responses_static_layout(padded_values, index, valid), layout
                )
            # Matching sequence metadata for the static layout: prompt length 1 and response length
            # `response_width` for every row, so the per-row slices of `no_padding_2_padding` are static too.
            offsets = torch.arange(batch_size + 1, dtype=torch.int64)
            loss_data["prompts"] = torch.nested.nested_tensor_from_jagged(
                torch.zeros(batch_size, dtype=torch.int64), offsets
            )
            loss_data["responses"] = torch.nested.nested_tensor_from_jagged(
                torch.zeros(batch_size * response_width, dtype=torch.int64), offsets * response_width
            )
        else:
            for key in padded_keys:
                tensor = model_output[key]
                # Trim trailing bucket padding inside the plugin without detaching gradients.
                valid_count = int(tensor.offsets()[-1].item())
                loss_model_output[key] = _TPUPaddedJaggedProxy(
                    getattr(tensor, TPU_PADDED_VALUES_ATTR)[:valid_count], tensor
                )

        return loss_model_output, loss_data

    if "loss_mask" not in micro_batch.keys() or "responses" in micro_batch.keys():
        return model_output, micro_batch

    loss_mask = micro_batch["loss_mask"]
    loss_mask_vals = loss_mask.values() if getattr(loss_mask, "is_nested", False) else loss_mask
    rolled_mask = torch.roll(loss_mask_vals.detach().cpu(), shifts=-1, dims=0)
    target_len = int(tpu_padded_log_prob.shape[0])
    if rolled_mask.shape[0] < target_len:
        rolled_mask = F.pad(rolled_mask, (0, target_len - int(rolled_mask.shape[0])), value=0)
    prerolled_mask = torch.roll(rolled_mask, shifts=1, dims=0).to(device=tpu_padded_log_prob.device).contiguous()

    loss_model_output = dict(model_output)
    loss_model_output["log_probs"] = _TPUPaddedJaggedProxy(tpu_padded_log_prob, log_prob)
    loss_data = _TPULossDataProxy(
        micro_batch,
        {"loss_mask": _TPUPaddedJaggedProxy(prerolled_mask, loss_mask)},
    )
    return loss_model_output, loss_data

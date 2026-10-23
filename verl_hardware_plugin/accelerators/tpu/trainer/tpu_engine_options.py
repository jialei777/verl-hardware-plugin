# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU-specific options of the TorchTitan engine, read from ``engine_kwargs.tpu`` of its config.

verl configures rollout and checkpoint backends through a per-backend ``engine_kwargs`` dict
(``rollout.engine_kwargs.vllm``, ``checkpoint_engine.engine_kwargs.tpu-sync``). The TPU TorchTitan
engine follows the same pattern::

    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_splash_attention=True
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.tpu_eager_mode=DEFER_AND_FUSE

The config reaches every worker through verl, so nothing needs to be forwarded through the
environment. For compatibility, the older top-level fields (``torchtitan.use_splash_attention`` ...)
and the ``VERL_TPU_*`` environment variables are still read, with a one-time deprecation warning;
an environment variable still takes precedence over the config, as before.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Key of the TPU options in ``TorchtitanEngineConfig.engine_kwargs``.
TPU_ENGINE_KWARGS_KEY = "tpu"


@dataclasses.dataclass(frozen=True)
class TPUEngineOptions:
    """TPU options of the TorchTitan engine. ``None`` means "use the default derived from the rest of the config".

    Attributes:
        use_splash_attention: Use the Pallas splash attention kernel (default: ``use_torch_compile``).
        use_simple_fsdp: Shard with SimpleFSDP (default: ``use_torch_compile`` when PP, CP and EP are 1).
        use_fused_linear_cross_entropy: Use the Pallas fused linear + cross-entropy kernel for log-probs
            (default: ``use_torch_compile`` when TP is 1).
        fuse_qkv: Fuse attention Q/K/V projections in the model (default: ``True``; ``False`` avoids
            the FSDP all-gather inside ``FusedQKVLinear._split_qkv_on_save`` during ``state_dict()``).
        tpu_eager_mode: ``torch_tpu`` eager mode, ``DEFER_AND_FUSE`` or ``DEFER_NEVER``
            (default: ``DEFER_AND_FUSE`` with ``use_torch_compile``).
        optim_impl: torchtitan optimizer implementation (default: ``fused``, ``for-loop`` with TP).
        seq_bucket_size: Packed sequence length bucket in tokens (default: ``pad_to_length_bucket`` when
            ``pad_to_length`` is set, else 256).
    """

    use_splash_attention: Optional[bool] = None
    use_simple_fsdp: Optional[bool] = None
    use_fused_linear_cross_entropy: Optional[bool] = None
    fuse_qkv: Optional[bool] = None
    tpu_eager_mode: Optional[str] = None
    optim_impl: Optional[str] = None
    seq_bucket_size: Optional[int] = None


_KINDS: dict[str, type] = {
    "use_splash_attention": bool,
    "use_simple_fsdp": bool,
    "use_fused_linear_cross_entropy": bool,
    "fuse_qkv": bool,
    "tpu_eager_mode": str,
    "optim_impl": str,
    "seq_bucket_size": int,
}
assert set(_KINDS) == {f.name for f in dataclasses.fields(TPUEngineOptions)}

# Deprecated channels, read for compatibility: option -> environment variable / top-level config field.
_LEGACY_ENV_VARS = {
    "use_splash_attention": "VERL_TPU_USE_SPLASH_ATTENTION",
    "use_simple_fsdp": "VERL_TPU_USE_SIMPLE_FSDP",
    "use_fused_linear_cross_entropy": "VERL_TPU_USE_FUSED_LINEAR_CROSS_ENTROPY",
    "tpu_eager_mode": "VERL_TPU_EAGER_MODE",
    "optim_impl": "VERL_TPU_OPTIM_IMPL",
    "seq_bucket_size": "VERL_TPU_SEQ_BUCKET_SIZE",
}
_LEGACY_CONFIG_FIELDS = (
    "use_splash_attention",
    "use_simple_fsdp",
    "use_fused_linear_cross_entropy",
    "fuse_qkv",
    "tpu_eager_mode",
)

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def _parse(name: str, value: Any, source: str) -> Any:
    """Converts ``value`` (a config value or an environment string) to the option's type; ``None``/"" is unset."""
    if value is None:
        return None
    kind = _KINDS[name]
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        if kind is bool:
            if isinstance(value, bool):
                return value
            lowered = str(value).lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            raise ValueError(value)
        if kind is int:
            parsed = int(value)
            if parsed <= 0:
                raise ValueError(value)
            return parsed
        # Strings are passed through; e.g. tpu_eager_mode="NONE" explicitly disables the eager mode.
        return str(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid TPU engine option {name}={value!r} from {source}") from e


def get_tpu_engine_options(engine_config: Any) -> TPUEngineOptions:
    """Returns the TPU options of ``engine_config``: ``engine_kwargs.tpu``, then the deprecated channels.

    Precedence per option: ``VERL_TPU_*`` environment variable (deprecated) > ``engine_kwargs.tpu`` >
    top-level config field (deprecated) > ``None`` (the caller's default).

    Raises:
        ValueError: on an unknown key in ``engine_kwargs.tpu`` or a value of the wrong type.
    """
    engine_kwargs = getattr(engine_config, "engine_kwargs", None) or {}
    raw = engine_kwargs.get(TPU_ENGINE_KWARGS_KEY) if isinstance(engine_kwargs, Mapping) else None
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(_KINDS))
    if unknown:
        raise ValueError(
            f"Unknown TPU engine option(s) {unknown} in engine_kwargs.{TPU_ENGINE_KWARGS_KEY}; "
            f"supported: {sorted(_KINDS)}"
        )
    values = {name: _parse(name, value, f"engine_kwargs.{TPU_ENGINE_KWARGS_KEY}") for name, value in raw.items()}

    for name in _LEGACY_CONFIG_FIELDS:
        legacy = getattr(engine_config, name, None)
        if legacy is not None and values.get(name) is None:
            _warn_once(
                f"cfg:{name}",
                f"torchtitan.{name} is deprecated; "
                f"set torchtitan.engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.{name} instead.",
            )
            values[name] = _parse(name, legacy, f"torchtitan.{name}")

    for name, env_name in _LEGACY_ENV_VARS.items():
        raw_env = os.environ.get(env_name)
        env = _parse(name, raw_env, env_name)
        if env is not None:
            _warn_once(
                f"env:{env_name}",
                f"{env_name}={raw_env!r} overrides the engine config and is deprecated; set "
                f"torchtitan.engine_kwargs.{TPU_ENGINE_KWARGS_KEY}.{name} instead.",
            )
            values[name] = env

    return TPUEngineOptions(**values)

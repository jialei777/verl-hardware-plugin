# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU accelerator modules."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# GKE injects MEGASCALE_* (multislice) variables into every TPU Job of a JobSet, so a Ray cluster built from one
# JobSet with a trainer slice and a rollout slice looks like one multislice job to libtpu, which then waits for
# the other "slices" forever. The trainer and the rollout each build their own single-slice mesh, so drop them
# before the first PJRT client is created (VERL_TPU_IGNORE_MEGASCALE=1).
if os.environ.get("VERL_TPU_IGNORE_MEGASCALE", "").strip().lower() in ("1", "true", "yes", "on"):
    for _name in [n for n in os.environ if n.startswith("MEGASCALE_")]:
        os.environ.pop(_name, None)

# Default libtpu XLA flags applied before torch_tpu / PJRT initializes:
# - xla_tpu_use_enhanced_launch_barrier=false: prevents distributed hangs across independent Ray meshes
# - xla_tpu_enable_bitcast_reduce_fusion_graduation=false /
#   xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false: avoids the XLA SpatialMajorConvolution
#   TransformWindow CHECK failure ("split of a padded window dim is not supported") when a bitcast + pad +
#   reduce is fused into a preceding matmul in backward (e.g. Qwen3.5 GatedDeltaNet causal conv1d backward).
_DEFAULT_LIBTPU_FLAGS = (
    "--xla_tpu_use_enhanced_launch_barrier=false",
    "--xla_tpu_enable_bitcast_reduce_fusion_graduation=false",
    "--xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false",
)


def ensure_default_libtpu_init_args(existing: str | None = None) -> str:
    """Append default ``LIBTPU_INIT_ARGS`` flags whose keys are not already present."""
    cur = (existing if existing is not None else os.environ.get("LIBTPU_INIT_ARGS", "")).strip()
    parts = [cur] if cur else []
    for flag in _DEFAULT_LIBTPU_FLAGS:
        key = flag.split("=", 1)[0]
        if key not in cur:
            parts.append(flag)
    val = " ".join(parts)
    os.environ["LIBTPU_INIT_ARGS"] = val
    return val


ensure_default_libtpu_init_args()

try:
    from verl_hardware_plugin.accelerators.tpu import platform_tpu  # noqa: F401
except Exception as exc:
    logger.debug("TPU platform registration failed: %s", exc)

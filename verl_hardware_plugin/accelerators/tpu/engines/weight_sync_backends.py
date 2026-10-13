# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Names of the TPU weight sync backends (``actor_rollout_ref.rollout.checkpoint_engine.backend``).

* ``ray-plasma-store``: trainer rank 0 puts the full weights into the Ray object store (plasma) and every
  rollout worker pulls them from there (``TPUCheckpointEngine``).
* ``tpu-sync``: the trainer chips send the weights peer to peer to the rollout chips with tpu-sync
  (Raiden); each rollout chip receives only its tensor-parallel shard (``RaidenCheckpointEngine``).

The former names ``tpu`` and ``raiden`` are still accepted and emit a ``FutureWarning``.
"""

import warnings
from typing import Optional

RAY_PLASMA_STORE_BACKEND = "ray-plasma-store"
TPU_SYNC_BACKEND = "tpu-sync"

# Former backend names, mapped to the current ones.
DEPRECATED_BACKEND_NAMES = {
    "tpu": RAY_PLASMA_STORE_BACKEND,
    "raiden": TPU_SYNC_BACKEND,
}

_warned_deprecated_names: set[str] = set()


def canonical_backend(backend: Optional[str]) -> Optional[str]:
    """Returns the current name of a weight sync backend.

    Args:
        backend: The configured ``checkpoint_engine.backend``.

    Returns:
        ``backend``, or its current name if it is a deprecated one (warned once per process).
    """
    if backend is None or backend not in DEPRECATED_BACKEND_NAMES:
        return backend
    current = DEPRECATED_BACKEND_NAMES[backend]
    if backend not in _warned_deprecated_names:
        _warned_deprecated_names.add(backend)
        # A warning rather than a log message: in verl's driver process the root logger is set to ERROR
        # (by model_hosting_container_standards, see docs/accelerators/tpu/weight_sync.md), which would hide it.
        warnings.warn(
            f"actor_rollout_ref.rollout.checkpoint_engine.backend={backend} is deprecated, use {current} instead.",
            FutureWarning,
            stacklevel=2,
        )
    return current

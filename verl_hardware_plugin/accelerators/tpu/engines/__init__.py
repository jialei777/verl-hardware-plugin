# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU training and checkpoint engines."""


def _ensure_torchtitan_lr_scheduler_compat() -> None:
    """Expose ``LRSchedulersContainer`` on ``torchtitan.components.optimizer`` for older TorchTitan builds.

    verl imports ``LRSchedulersContainer`` from ``torchtitan.components.optimizer``
    (``verl/workers/engine/torchtitan/transformer_impl.py``), but the TorchTitan build in the TPU images still
    defines it in ``torchtitan.components.lr_scheduler``. Without this alias, importing verl's TorchTitan engine
    fails, so ``TorchTitanTPUEngineWithLMHead`` (a subclass of it) is never registered.

    This package is imported by ``platform_tpu`` before anything imports ``verl.workers.engine``. It is a no-op
    when TorchTitan is not installed or already provides the name.
    """
    try:
        import torchtitan.components.optimizer as tt_optimizer
    except ImportError:
        return
    if hasattr(tt_optimizer, "LRSchedulersContainer"):
        return
    try:
        from torchtitan.components.lr_scheduler import LRSchedulersContainer
    except ImportError:
        return
    tt_optimizer.LRSchedulersContainer = LRSchedulersContainer


_ensure_torchtitan_lr_scheduler_compat()

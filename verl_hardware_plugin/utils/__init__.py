# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Multi-Chip Support Utilities."""

from .config_manager import FLEnvManager, may_enable_flag_gems
from .function_wrapper import FUNCTION_WRAPPER

__all__ = ["FLEnvManager", "may_enable_flag_gems", "FUNCTION_WRAPPER"]

# Copyright (c) OpenMMLab. All rights reserved.

from .api import client, pipeline, serve
from .messages import (
    GenerationConfig,
    PytorchEngineConfig,
    TurbomindEngineConfig,
    VisionConfig,
)

# Import DriftEngine configuration from messages.py (canonical source)
try:
    from .messages import DriftEngineConfig
except ImportError:  # pragma: no cover
    DriftEngineConfig = None
from .model import ChatTemplateConfig
from .tokenizer import Tokenizer
from .version import __version__, version_info

# TurboMind (C++ backend) is optional at import time: in environments where
# the `_turbomind` extension is not built, we still want lightweight helpers
# (messages, metrics, etc.) to be importable without raising. Downstream code
# that actually needs TurboMind should explicitly check for `None`.
try:
    from .turbomind.turbomind import TurboMind
except Exception:  # pragma: no cover - backend availability is env-dependent
    TurboMind = None

__all__ = [
    "pipeline",
    "serve",
    "client",
    "Tokenizer",
    "GenerationConfig",
    "__version__",
    "version_info",
    "ChatTemplateConfig",
    "PytorchEngineConfig",
    "TurbomindEngineConfig",
    "VisionConfig",
    "TurboMind",
    "DriftEngineConfig",
]

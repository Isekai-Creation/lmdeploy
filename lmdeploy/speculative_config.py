# Copyright (c) OpenMMLab. All rights reserved.
"""Speculative decoding configuration for LMDeploy engines."""

from dataclasses import dataclass
from typing import Optional, List, Dict
import warnings


@dataclass
class SpeculativeConfig:
    """
    Unified speculative decoding configuration.

    Compatible with both PyTorch and TurboMind backends.
    Designed to be simple by default (like PyTorch) but allow advanced
    tuning (like TensorRT-LLM) when needed.

    Args:
        method: Speculative decoding algorithm. Options:
            - 'draft_target': Simple two-model approach (TurboMind only)
            - 'eagle': EAGLE 1/2 (PyTorch)
            - 'eagle3': EAGLE 3 (PyTorch + TurboMind)
            - 'ngram': Prompt lookup (TurboMind only)
            - 'deepseek_mtp': Deepseek MTP (PyTorch only)
        model: Path to draft model. Required for eagle/eagle3/draft_target.
            Should be a smaller, faster model with same tokenizer as target.
        num_speculative_tokens: Number of tokens to speculate per step.
            Typical values: 3-5 for EAGLE, 1-3 for draft_target.

    Advanced EAGLE options (TurboMind only, optional):
        max_path_len: Maximum tree depth. Auto-set if None. Default: None.
        max_decoding_tokens: Max tokens per step. Auto-set if None. Default: None.
        max_non_leaves_per_layer: Max non-leaf nodes per level. Auto-set if None. Default: None.
        capture_layers: Which layers to capture hidden states from.
            List of layer indices, or None for auto (last layer). Default: None.

    NGram options (TurboMind only):
        max_matching_ngram_size: Maximum n-gram size (default: 4)
        is_public_pool: Share n-gram pool across requests (default: True)

    Example (Simple - PyTorch style):
        >>> # Works with both PyTorch and TurboMind
        >>> config = SpeculativeConfig(
        ...     method='eagle3',
        ...     model='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
        ...     num_speculative_tokens=3
        ... )

    Example (Advanced - TurboMind tuning):
        >>> # Advanced options only used by TurboMind
        >>> config = SpeculativeConfig(
        ...     method='eagle3',
        ...     model='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
        ...     num_speculative_tokens=5,
        ...     max_path_len=7,              # Custom tree depth
        ...     max_decoding_tokens=15,      # Custom token limit
        ...     capture_layers=[-1, -5]      # Capture last 2 layers
        ... )

    Example (NGram - no draft model):
        >>> config = SpeculativeConfig(
        ...     method='ngram',
        ...     num_speculative_tokens=3,
        ...     max_matching_ngram_size=4
        ... )
    """

    # Core parameters (both PyTorch and TurboMind)
    method: str = "eagle"
    model: str = ""
    num_speculative_tokens: int = 5

    # Advanced EAGLE options (TurboMind only, optional with auto-defaults)
    max_path_len: Optional[int] = None
    max_decoding_tokens: Optional[int] = None
    max_non_leaves_per_layer: Optional[int] = None
    capture_layers: Optional[List[int]] = None

    # NGram options (TurboMind only)
    max_matching_ngram_size: int = 4
    is_public_pool: bool = True

    def __post_init__(self):
        """Validate configuration and set sensible defaults."""
        valid_methods = [
            "draft_target",  # TurboMind only
            "eagle",  # PyTorch
            "eagle3",  # PyTorch + TurboMind
            "ngram",  # TurboMind only
            "deepseek_mtp",  # PyTorch only
        ]

        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}'. " f"Must be one of {valid_methods}"
            )

        # Require model for certain methods
        if self.method in ["draft_target", "eagle", "eagle3"] and not self.model:
            raise ValueError(f"Method '{self.method}' requires 'model' to be specified")

        if self.num_speculative_tokens < 1:
            raise ValueError(
                "num_speculative_tokens must be >= 1, "
                f"got {self.num_speculative_tokens}"
            )

        # Set EAGLE defaults if not specified (TurboMind only)
        if self.method in ["eagle", "eagle3"]:
            if self.max_path_len is None:
                self.max_path_len = 5  # Sensible default
            if self.max_decoding_tokens is None:
                # Auto-set based on num_speculative_tokens
                self.max_decoding_tokens = self.num_speculative_tokens * 2
            if self.max_non_leaves_per_layer is None:
                self.max_non_leaves_per_layer = 10
            if self.capture_layers is None:
                self.capture_layers = [-1]  # Last layer by default

            # EAGLE-specific validation: all structural parameters must be
            # positive and internally consistent so that TurboMind can safely
            # size EagleModule / EagleBuffers.
            if self.max_path_len <= 0:
                raise ValueError(
                    f"max_path_len must be > 0 for method '{self.method}', "
                    f"got {self.max_path_len}"
                )
            if self.max_decoding_tokens <= 0:
                raise ValueError(
                    f"max_decoding_tokens must be > 0 for method '{self.method}', "
                    f"got {self.max_decoding_tokens}"
                )
            if self.max_non_leaves_per_layer <= 0:
                raise ValueError(
                    f"max_non_leaves_per_layer must be > 0 for method '{self.method}', "
                    f"got {self.max_non_leaves_per_layer}"
                )

            # max_path_len cannot exceed max_decoding_tokens and should be at
            # least as large as num_speculative_tokens (otherwise the tree
            # cannot represent the speculative step).
            if self.max_path_len > self.max_decoding_tokens:
                raise ValueError(
                    "max_path_len must be <= max_decoding_tokens "
                    f"for method '{self.method}', got "
                    f"max_path_len={self.max_path_len}, "
                    f"max_decoding_tokens={self.max_decoding_tokens}"
                )
            if self.max_path_len < self.num_speculative_tokens:
                raise ValueError(
                    "max_path_len must be >= num_speculative_tokens "
                    f"for method '{self.method}', got "
                    f"max_path_len={self.max_path_len}, "
                    f"num_speculative_tokens={self.num_speculative_tokens}"
                )

    # ---- TurboMind alignment helpers -------------------------------------------------

    def to_turbomind_spec_dict(self) -> Dict[str, object]:
        """Return a dict matching TurboMind's expected speculative_config block.

        This dict uses the same keys that the C++ Triton backend reads in
        ``LlamaTritonModel.cc`` (``method``, ``model``, ``num_speculative_tokens``,
        ``max_path_len``, ``max_decoding_tokens``, ``max_non_leaves_per_layer``),
        so it can be used when constructing engine configs or sanity-checking
        YAML-based configs for TurboMind.
        """
        d: Dict[str, object] = {
            "method": self.method,
        }
        if self.model:
            d["model"] = self.model
        # Only include structural fields when they are explicitly set or have
        # been defaulted in __post_init__.
        d["num_speculative_tokens"] = self.num_speculative_tokens
        if self.max_path_len is not None:
            d["max_path_len"] = self.max_path_len
        if self.max_decoding_tokens is not None:
            d["max_decoding_tokens"] = self.max_decoding_tokens
        if self.max_non_leaves_per_layer is not None:
            d["max_non_leaves_per_layer"] = self.max_non_leaves_per_layer
        return d


def check_turbomind_spec_alignment(spec_cfg: SpeculativeConfig, engine_spec: Dict[str, object]) -> None:
    """Emit a warning if SpeculativeConfig and engine-side spec differ.

    Args:
        spec_cfg: The high-level SpeculativeConfig used in Python.
        engine_spec: A dict that reflects the speculative_config block seen
            by TurboMind (for example, parsed from an engine YAML or built
            by a higher-level config generator).

    This helper compares the fields that matter for TurboMind EAGLE
    integration (method, model, num_speculative_tokens, max_path_len,
    max_decoding_tokens, max_non_leaves_per_layer). If any of them differ
    between Python and ``engine_spec``, a UserWarning is emitted describing
    the mismatch so offline users can diagnose configuration drift.
    """
    expected = spec_cfg.to_turbomind_spec_dict()
    mismatches = {}
    for key, expected_val in expected.items():
        if key not in engine_spec:
            mismatches[key] = ("<missing>", expected_val)
        else:
            current_val = engine_spec[key]
            if current_val != expected_val:
                mismatches[key] = (current_val, expected_val)

    if mismatches:
        details = ", ".join(
            f"{k}: engine={cur!r}, expected={exp!r}" for k, (cur, exp) in mismatches.items()
        )
        warnings.warn(
            f"Turbomind speculative_config mismatch detected: {details}",
            UserWarning,
        )

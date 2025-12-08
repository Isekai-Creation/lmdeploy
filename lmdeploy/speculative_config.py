# Copyright (c) OpenMMLab. All rights reserved.
"""Speculative decoding configuration for LMDeploy engines."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
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

    # EAGLE debug/metrics verbosity (TurboMind only). When True, enable
    # detailed EAGLE traces and/or per-step metrics logging on the TurboMind
    # backend for engines using this SpeculativeConfig.
    eagle_debug: bool = False
    eagle_metrics_debug: bool = False

    # Target-tree decode (TurboMind only). When True, allow the TurboMind
    # backend to run a dedicated target decode over the EAGLE speculation
    # tree and produce per-node target token ids, instead of relying solely
    # on single-step target logits to fabricate target_tokens.
    enable_target_tree: bool = False

    # SpecPV partial KV verification (TurboMind EAGLE3 only). When enabled,
    # TurboMind may verify EAGLE trees against a partial KV cache (sink +
    # retrieval + window + speculative buffer) instead of the full prefix
    # KV at long context. These fields mirror the EngineParam specpv_*
    # members and Triton speculative_config entries.
    enable_specpv: bool = False
    specpv_block_size: int = 16
    specpv_n_sink_blocks: int = 2
    specpv_n_retrieval_blocks: int = 256
    specpv_n_window_blocks: int = 8
    specpv_n_spec_tokens_buf: int = 128
    specpv_partial_threshold: int = 4096
    specpv_full_refresh_steps: int = 32

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
                # Default tree depth. For single-token speculation we keep
                # this small; for multi-token we allow a bit more room so
                # the tree can represent extra candidates.
                self.max_path_len = 5  # Sensible default
            if self.max_decoding_tokens is None:
                # Auto-set based on num_speculative_tokens. Ensure that
                # max_decoding_tokens is always at least as large as
                # max_path_len so that the tree depth constraint can be
                # satisfied even for small num_speculative_tokens.
                self.max_decoding_tokens = max(
                    self.num_speculative_tokens * 2, self.max_path_len
                )
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
            # cannot represent the speculative step). When users provide
            # partially conflicting values, we auto-clamp max_path_len down
            # to max_decoding_tokens rather than failing hard, as long as
            # the result is still usable.
            if self.max_path_len > self.max_decoding_tokens:
                warnings.warn(
                    "SpeculativeConfig: max_path_len=%d exceeds max_decoding_tokens=%d "
                    "for method '%s'; clamping max_path_len to max_decoding_tokens."
                    % (self.max_path_len, self.max_decoding_tokens, self.method)
                )
                self.max_path_len = self.max_decoding_tokens
            if self.max_path_len < self.num_speculative_tokens:
                raise ValueError(
                    "max_path_len must be >= num_speculative_tokens "
                    f"for method '{self.method}', got "
                    f"max_path_len={self.max_path_len}, "
                    f"num_speculative_tokens={self.num_speculative_tokens}"
                )

            # Ensure the speculative step size is not larger than the
            # configured decoding budget per step.
            if self.max_decoding_tokens is not None and self.num_speculative_tokens > self.max_decoding_tokens:
                raise ValueError(
                    "num_speculative_tokens must be <= max_decoding_tokens "
                    f"for method '{self.method}', got "
                    f"num_speculative_tokens={self.num_speculative_tokens}, "
                    f"max_decoding_tokens={self.max_decoding_tokens}"
                )

    # ---- TurboMind alignment helpers -------------------------------------------------

    def to_turbomind_spec_dict(self) -> Dict[str, object]:
        """Return a dict matching TurboMind's expected speculative_config block.

        This dict uses the same keys that the C++ Triton backend reads in
        ``LlamaTritonModel.cc`` (``method``, ``model``, ``num_speculative_tokens``,
        ``max_path_len``, ``max_decoding_tokens``, ``max_non_leaves_per_layer``,
        plus advanced EAGLE flags like ``eagle_debug``, ``eagle_metrics_debug``,
        ``enable_target_tree`` and optional SpecPV fields), so it can be used
        when constructing engine configs or sanity-checking YAML-based configs
        for TurboMind.
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

        # Debug/metrics verbosity flags are always included explicitly so
        # that TurboMind can distinguish between "unset" and "false".
        d["eagle_debug"] = bool(self.eagle_debug)
        d["eagle_metrics_debug"] = bool(self.eagle_metrics_debug)
        # Target-tree decode flag is always explicit as well so that
        # the Triton backend can gate the advanced EAGLE3 path.
        d["enable_target_tree"] = bool(self.enable_target_tree)

        # SpecPV flags are included explicitly so that TurboMind can see
        # the partial-KV configuration when present. Engines that do not
        # support SpecPV will simply ignore these fields.
        d["enable_specpv"] = bool(self.enable_specpv)
        d["specpv_block_size"] = int(self.specpv_block_size)
        d["specpv_n_sink_blocks"] = int(self.specpv_n_sink_blocks)
        d["specpv_n_retrieval_blocks"] = int(self.specpv_n_retrieval_blocks)
        d["specpv_n_window_blocks"] = int(self.specpv_n_window_blocks)
        d["specpv_n_spec_tokens_buf"] = int(self.specpv_n_spec_tokens_buf)
        d["specpv_partial_threshold"] = int(self.specpv_partial_threshold)
        d["specpv_full_refresh_steps"] = int(self.specpv_full_refresh_steps)
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


def validate_eagle_runtime_config(engine_config: Any, spec_cfg: Optional[SpeculativeConfig]) -> None:
    """Runtime validation for TurboMind EAGLE configs (single-GPU, offline use).

    This helper is intentionally stricter for Engineer-B's offline /
    single-GPU path: misconfigurations raise ValueError so that the
    pipeline fails fast instead of running with partially broken EAGLE
    semantics.
    """
    if spec_cfg is None:
        return

    # Only EAGLE/EAGLE3 are supported here.
    if spec_cfg.method not in ("eagle", "eagle3"):
        raise ValueError(
            f"EAGLE SpeculativeConfig.method={spec_cfg.method!r} is not supported; "
            "expected 'eagle3' (or 'eagle')."
        )

    if engine_config is None:
        raise ValueError(
            "EAGLE runtime config: engine_config is None while "
            f"SpeculativeConfig(method='{spec_cfg.method}') is in use; TurboMind cannot "
            "run EAGLE without an engine config."
        )

    cfg_spec = getattr(engine_config, "speculative_config", None)
    if cfg_spec is None:
        # For offline/single-GPU helpers, allow attaching the SpeculativeConfig
        # directly to the engine_config when it has not yet been wired, so that
        # validation can still proceed and keep behaviour aligned with callers
        # like benchmark_speculative.py and eagle_inspect.
        try:
            setattr(engine_config, "speculative_config", spec_cfg)
            cfg_spec = spec_cfg
        except Exception as exc:  # pragma: no cover - very defensive
            raise ValueError(
                "EAGLE runtime config: engine_config.speculative_config is None while "
                f"SpeculativeConfig(method='{spec_cfg.method}') is provided; "
                "TurboMind speculative decoding is not configured."
            ) from exc

    # Cross-check engine-side speculative config against the Python one
    # when we can get a mapping out of it.
    engine_spec_dict: Dict[str, object] = {}
    try:
        if hasattr(cfg_spec, "to_turbomind_spec_dict"):
            engine_spec_dict = cfg_spec.to_turbomind_spec_dict()  # type: ignore[assignment]
        elif hasattr(cfg_spec, "__dict__"):
            engine_spec_dict = dict(cfg_spec.__dict__)
    except Exception:
        engine_spec_dict = {}

    if engine_spec_dict:
        check_turbomind_spec_alignment(spec_cfg, engine_spec_dict)

    # num_speculative_tokens must be >= 1
    if spec_cfg.num_speculative_tokens < 1:
        raise ValueError(
            f"num_speculative_tokens must be >= 1; got {spec_cfg.num_speculative_tokens}."
        )

    # Single-GPU only for multi-token (for now).
    tp = getattr(engine_config, "tp", 1)
    if spec_cfg.num_speculative_tokens > 1 and tp != 1:
        raise ValueError(
            "EAGLE3 multi-token requires tp=1. "
            f"Got num_speculative_tokens={spec_cfg.num_speculative_tokens}, tp={tp}."
        )

    # Structural fields must be positive when multi-token is requested.
    if spec_cfg.num_speculative_tokens > 1:
        if spec_cfg.max_path_len is not None and spec_cfg.max_path_len <= 0:
            raise ValueError(
                f"max_path_len must be > 0 for multi-token; got {spec_cfg.max_path_len}."
            )
        if spec_cfg.max_decoding_tokens is not None and spec_cfg.max_decoding_tokens <= 0:
            raise ValueError(
                f"max_decoding_tokens must be > 0 for multi-token; got {spec_cfg.max_decoding_tokens}."
            )

    # Session length should be large enough for the per-step decoding budget.
    if spec_cfg.max_decoding_tokens is not None:
        session_len = getattr(engine_config, "session_len", None)
        if session_len is not None and session_len < spec_cfg.max_decoding_tokens:
            raise ValueError(
                f"session_len={session_len} < max_decoding_tokens={spec_cfg.max_decoding_tokens}; "
                "this may truncate EAGLE decoding."
            )

    # Ensure metrics are enabled so EAGLE statistics surface cleanly.
    enable_metrics = getattr(engine_config, "enable_metrics", True)
    if not enable_metrics:
        warnings.warn(
            "EAGLE runtime config: engine_config.enable_metrics is False; "
            "TurboMind EAGLE runs will not populate req_metrics.spec_info.",
            UserWarning,
        )

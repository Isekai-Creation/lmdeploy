"""
Speculative Decoding Manager for TurboMind.

Coordinates draft and target models for speculative decoding.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SpeculativeDecodingManager:
    """
    Manages speculative decoding for TurboMind.

    Supports:
    - Draft/Target: Simple two-model approach
    - EAGLE3: Tree-based speculation with hidden state capture
    - NGram: Prompt lookup without draft model
    """

    def __init__(self, config, target_model_comm):
        """
        Initialize speculative decoding manager.

        Args:
            config: SpeculativeConfig instance
            target_model_comm: Target model communication object
        """
        from lmdeploy.speculative_config import SpeculativeConfig as SpecConfigV1
        from lmdeploy.messages import SpeculativeConfig as SpecConfigV2

        # Accept SpeculativeConfig from either module (backward compatibility)
        if not isinstance(config, (SpecConfigV1, SpecConfigV2)):
            raise TypeError(f"config must be SpeculativeConfig, got {type(config)}")

        self.config = config
        self.target_model_comm = target_model_comm
        self.draft_model_comm = None
        self.enabled = True

        logger.info(f"Initializing speculative decoding with method: {config.method}")

        # Initialize based on method
        if config.method == "draft_target":
            self._init_draft_target()
        elif config.method in ["eagle", "eagle3"]:
            # eagle3 is the same as eagle for TurboMind
            self._init_eagle()
        elif config.method == "ngram":
            self._init_ngram()
        else:
            raise ValueError(f"Unknown speculative decoding method: {config.method}")

    def _init_draft_target(self):
        """Initialize draft/target speculative decoding."""
        if not self.config.model:
            raise ValueError("Draft/target method requires 'model' to be specified")

        logger.info(f"Loading draft model from: {self.config.model}")

        try:
            # Import here to avoid circular dependencies
            from lmdeploy.turbomind.turbomind import TurboMind
            from lmdeploy.messages import TurbomindEngineConfig

            # Create a lightweight engine config for draft model
            draft_engine_config = TurbomindEngineConfig(
                tp=1,  # Draft model typically uses single GPU
                max_batch_size=1,  # Process one draft at a time
                session_len=512,  # Shorter context for draft
            )

            # Load draft model
            logger.info("Creating TurboMind instance for draft model...")
            self.draft_model = TurboMind(
                model_path=self.config.model, engine_config=draft_engine_config
            )

            logger.info(f"Draft model loaded successfully: {self.config.model}")

        except Exception as e:
            logger.error(f"Failed to load draft model: {e}")
            raise RuntimeError(
                f"Could not load draft model from {self.config.model}: {e}"
            )

    def _init_eagle(self):
        """Initialize EAGLE / EAGLE3 speculative decoding.

        Native C++ implementation handles draft model loading and generation.
        If the draft model path contains HuggingFace format files, we
        auto-convert to TurboMind format first.
        """
        if not self.config.model:
            raise ValueError("EAGLE method requires 'model' to be specified")

        import os

        draft_path = self.config.model

        # Check if this is a HuggingFace format model (has config.json or *.safetensors)
        is_hf_format = (
            os.path.exists(os.path.join(draft_path, "config.json"))
            or os.path.exists(os.path.join(draft_path, "model.safetensors"))
            or len([f for f in os.listdir(draft_path) if f.endswith(".safetensors")])
            > 0
        )

        # Check if TurboMind format already exists (has TM-format weights like norm.weight file)
        is_tm_format = os.path.exists(os.path.join(draft_path, "norm.weight"))

        if is_hf_format and not is_tm_format:
            logger.info(
                f"EAGLE model at {draft_path} is HuggingFace format, converting to TurboMind..."
            )
            try:
                from lmdeploy.turbomind.eagle_draft_converter import (
                    prepare_eagle_draft_from_hf,
                )

                # Convert in-place or to a subdirectory
                tm_dir = os.path.join(draft_path, "turbomind_eagle")
                os.makedirs(tm_dir, exist_ok=True)

                # Find base model path if available (from parent dir or config)
                base_model_dir = None
                # Try to find base model in parent dir (common layout: base_model/eagle3/)
                parent = os.path.dirname(draft_path)
                if os.path.exists(os.path.join(parent, "config.json")):
                    base_model_dir = parent

                # Run the conversion
                converted_path = prepare_eagle_draft_from_hf(
                    hf_model_dir=draft_path,
                    out_dir=tm_dir,
                    base_model_dir=base_model_dir,
                    logger=logger,
                )

                logger.info(
                    f"EAGLE model converted to TurboMind format at: {converted_path}"
                )
                # Update config to use converted path
                self.config.model = converted_path
                draft_path = converted_path
            except Exception as e:
                logger.error(f"EAGLE HF->TM conversion failed: {e}")
                import traceback

                logger.error(traceback.format_exc())
                logger.warning(
                    "Continuing with original path; C++ may fail to load weights"
                )

        logger.info("Initializing native EAGLE speculative decoding")
        logger.info("EAGLE draft model (managed natively by TurboMind): %s", draft_path)
        logger.info(
            "EAGLE config: max_path_len=%s, max_decoding_tokens=%s, num_speculative_tokens=%s",
            getattr(self.config, "max_path_len", None),
            getattr(self.config, "max_decoding_tokens", None),
            getattr(self.config, "num_speculative_tokens", None),
        )

        # Native implementation handles everything in C++.
        # Python must not create or own a separate draft model instance
        # for EAGLE/EAGLE3; TurboMind is the single source of truth.
        self.draft_model = None
        self.draft_tokenizer = None

    def _init_ngram(self):
        """Initialize NGram speculative decoding."""
        logger.info(
            f"NGram config: max_matching_ngram_size="
            f"{self.config.max_matching_ngram_size}, "
            f"is_public_pool={self.config.is_public_pool}"
        )

        # TODO: Initialize NGram pool
        # self.ngram_pool = NGramPool(...)

        logger.warning(
            "NGram speculative decoding initialized but not yet "
            "fully implemented. NGram pool is TODO."
        )

    def is_enabled(self):
        """Check if speculative decoding is enabled."""
        return self.enabled

    def get_method(self):
        """Get the speculative decoding method."""
        return self.config.method

    def get_num_speculative_tokens(self):
        """Get number of speculative tokens per step."""
        return self.config.num_speculative_tokens

    def generate_draft_tokens(self, input_ids, num_tokens=None):
        """
        Generate draft tokens using the draft model.

        Args:
            input_ids: Input token IDs (list or tensor)
            num_tokens: Number of tokens to generate (uses config if None)

        Returns:
            List of draft token IDs
        """
        if num_tokens is None:
            num_tokens = self.config.num_speculative_tokens

        # For native EAGLE/EAGLE3, TurboMind's C++ backend owns the draft
        # model and speculative loop. Python‑side draft generation is only
        # used for `draft_target` / `ngram` methods.
        if self.config.method in ("eagle", "eagle3"):
            logger.debug(
                "EAGLE/EAGLE3 is handled natively in TurboMind; "
                "skipping Python draft generation"
            )
            return []

        draft_model = getattr(self, "draft_model", None)
        if draft_model is None:
            logger.warning("Draft model not loaded, cannot generate draft tokens")
            return []

        try:
            # Convert input_ids to a flat Python list for the HF model
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            elif not isinstance(input_ids, list):
                input_ids = [int(input_ids)]

            draft_tokens = self._generate_eagle_draft_tokens(input_ids, num_tokens)

            logger.debug(
                "Generated %d draft tokens: %s", len(draft_tokens), draft_tokens
            )
            return draft_tokens
        except Exception as e:
            logger.error(f"Draft token generation failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def generate_draft_tokens_sync(
        self, input_ids, num_tokens=None, hidden_states=None
    ):
        """Synchronous draft token generation.

        This helper mirrors the interface used in the high‑level design doc
        (generate_draft_tokens_sync) while delegating to the existing
        generate_draft_tokens implementation. The hidden_states argument is
        reserved for future integration with TurboMind's captured activations.
        """
        _ = hidden_states  # currently unused, kept for API compatibility
        return self.generate_draft_tokens(input_ids, num_tokens=num_tokens)

    def _generate_eagle_draft_tokens(self, input_ids, num_tokens):
        """Generate draft tokens using EAGLE3 model."""
        import torch

        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")

        draft_tokens = []

        with torch.no_grad():
            # Use KV cache for efficiency
            past_key_values = None

            for _ in range(num_tokens):
                # Run draft model
                outputs = self.draft_model(
                    input_tensor if past_key_values is None else input_tensor[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                # Get next token
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()

                draft_tokens.append(next_token)

                # Update for next iteration
                input_tensor = torch.cat(
                    [input_tensor, torch.tensor([[next_token]], device="cuda")], dim=1
                )

                # Update KV cache
                past_key_values = outputs.past_key_values

        return draft_tokens

    def verify_draft_tokens(self, draft_tokens, target_logits):
        """Verify draft tokens against target model outputs.

        This thin wrapper exists for callers that work with the manager
        directly. Internally it delegates to :class:`DraftTokenVerifier`
        which implements both greedy and probabilistic acceptance.
        """
        if not draft_tokens:
            return {"accepted_tokens": [], "num_accepted": 0, "acceptance_rate": 0.0}

        try:
            from lmdeploy.turbomind.draft_verifier import DraftTokenVerifier

            verifier = DraftTokenVerifier()
            return verifier.verify_tokens(draft_tokens, target_logits)
        except Exception as e:
            logger.error(f"Draft token verification failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return {"accepted_tokens": [], "num_accepted": 0, "acceptance_rate": 0.0}

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
        from lmdeploy.speculative_config import SpeculativeConfig

        if not isinstance(config, SpeculativeConfig):
            raise TypeError(f"config must be SpeculativeConfig, got {type(config)}")

        self.config = config
        self.target_model_comm = target_model_comm
        self.draft_model_comm = None
        self.enabled = True

        logger.info(f"Initializing speculative decoding with method: {config.method}")

        # Initialize based on method
        if config.method == "draft_target":
            self._init_draft_target()
        elif config.method == "eagle":
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
        """Initialize EAGLE3 speculative decoding."""
        if not self.config.model:
            raise ValueError("EAGLE method requires 'model' to be specified")

        logger.info(f"Loading EAGLE draft model from: {self.config.model}")
        logger.info(
            f"EAGLE config: max_path_len={self.config.max_path_len}, "
            f"max_decoding_tokens={self.config.max_decoding_tokens}"
        )

        # TODO: Load EAGLE draft model and initialize tree structures
        # self.draft_model_comm = load_eagle_model(self.config.model)
        # self.eagle_tree = SpeculationTree(...)

        logger.warning(
            "EAGLE3 speculative decoding initialized but not yet "
            "fully implemented. Draft model loading and tree "
            "structures are TODO."
        )

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

        if not hasattr(self, "draft_model"):
            logger.warning("Draft model not loaded, cannot generate draft tokens")
            return []

        try:
            # Convert input_ids to list if needed
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            elif not isinstance(input_ids, list):
                input_ids = [input_ids]

            # Create a session for draft model inference
            draft_session = self.draft_model.create_instance()

            # Use async helper to generate draft tokens
            from lmdeploy.turbomind.speculative_async_helper import (
                generate_draft_tokens_sync,
            )

            draft_tokens = generate_draft_tokens_sync(
                draft_session,
                input_ids,
                num_tokens,
                session_id=0,  # Use session 0 for draft
            )

            logger.debug(f"Generated {len(draft_tokens)} draft tokens: {draft_tokens}")
            return draft_tokens

        except Exception as e:
            logger.error(f"Draft token generation failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def verify_draft_tokens(self, draft_tokens, target_logits):
        """
        Verify draft tokens against target model outputs.

        Args:
            draft_tokens: Draft token IDs
            target_logits: Target model logits

        Returns:
            Acceptance results
        """
        # TODO: Implement verification
        logger.warning("Draft token verification not yet implemented")
        return {"accepted": [], "num_accepted": 0}

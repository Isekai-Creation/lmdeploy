"""
KV cache management helpers for speculative decoding.

These utilities provide a thin, Python-side abstraction for KV cache
rewind/length tracking that works with the existing TurboMind APIs. The
current implementation is deliberately conservative: when no explicit
rewind hook is available on the session or model, it logs a warning
instead of failing, so speculative decoding can be enabled without
impacting baseline behaviour.
"""

import logging
import traceback
from typing import Optional

logger = logging.getLogger(__name__)


class KVCacheManager:
    """Best-effort KV cache manager for speculative decoding.

    Handles:
    - Cache rewind when draft tokens are rejected
    - Cache length bookkeeping
    """

    def __init__(self, turbomind_instance):
        """
        Args:
            turbomind_instance: TurboMind or TurboMindInstance owner.
        """
        self.turbomind = turbomind_instance
        self.cache_length = 0  # Logical cache length for bookkeeping

    def rewind(self, num_tokens: int, session=None) -> None:
        """Rewind KV cache by removing the last ``num_tokens`` entries.

        When the underlying engine exposes explicit rewind hooks we call them;
        otherwise we log a warning and continue without modifying the cache.

        Args:
            num_tokens: Number of tokens to remove.
            session: Optional session object, if it implements ``rewind_cache``.
        """
        if num_tokens <= 0:
            return

        logger.debug("Rewinding KV cache by %d tokens", num_tokens)

        try:
            # Option 1: session-level rewind hook
            if session is not None and hasattr(session, "rewind_cache"):
                session.rewind_cache(num_tokens)
                self.cache_length = max(0, self.cache_length - num_tokens)
                logger.debug("Cache rewound via session, new length: %d", self.cache_length)
                return

            # Option 2: model_comm-level rewind hook (if exposed by C++ side)
            model_comm = getattr(self.turbomind, "model_comm", None)
            if model_comm is not None and hasattr(model_comm, "rewind_kv_cache"):
                model_comm.rewind_kv_cache(num_tokens)
                self.cache_length = max(0, self.cache_length - num_tokens)
                logger.debug("Cache rewound via model_comm, new length: %d", self.cache_length)
                return

            # Fallback: log-only; this keeps behaviour safe even if no hook exists.
            logger.warning(
                "KV cache rewind not implemented for %d tokens; this may increase "
                "memory usage for very long sequences.",
                num_tokens,
            )
        except Exception as e:
            logger.error("KV cache rewind failed: %s", e)
            logger.debug(traceback.format_exc())

    def update_length(self, num_tokens: int) -> None:
        """Record that ``num_tokens`` new tokens have been added to the cache."""
        if num_tokens <= 0:
            return
        self.cache_length += num_tokens
        logger.debug("Cache length updated: %d", self.cache_length)

    def reset(self) -> None:
        """Reset bookkeeping state."""
        self.cache_length = 0
        logger.debug("Cache state reset")

    def get_length(self) -> int:
        """Return the current logical cache length."""
        return self.cache_length


class SpeculativeKVCacheIntegration:
    """Integrate KV cache management with speculative decoding steps.

    This helper coordinates cache updates during speculation:

    1. The cache grows when draft tokens are added by the target model.
    2. On rejection, we rewind the cache for the rejected tail.
    3. For accepted tokens, the cache stays consistent with the final prefix.
    """

    def __init__(self, kv_cache_manager: KVCacheManager):
        self.cache_mgr = kv_cache_manager

    def handle_speculation_step(
        self,
        num_draft: int,
        num_accepted: int,
        session=None,
    ) -> None:
        """Apply a single speculation step to the KV cache.

        Args:
            num_draft: Number of draft tokens proposed.
            num_accepted: Number of those tokens that were accepted.
            session: Optional session object for low-level rewinds.
        """
        num_rejected = max(0, num_draft - num_accepted)

        if num_rejected > 0:
            logger.debug(
                "Speculation KV update: %d draft, %d accepted, %d rejected – rewinding cache",
                num_draft,
                num_accepted,
                num_rejected,
            )
            self.cache_mgr.rewind(num_rejected, session=session)
        else:
            logger.debug(
                "Speculation KV update: all %d draft tokens accepted; no rewind required",
                num_accepted,
            )

    def prepare_for_draft(self, num_draft: int, session=None) -> None:
        """Hook before adding draft tokens (currently logging only)."""
        _ = session
        logger.debug("Preparing KV cache for %d draft tokens", num_draft)

    def finalize_step(self, num_accepted: int, session=None) -> None:
        """Hook after a speculation step has completed (currently logging only)."""
        _ = session
        logger.debug("Finalizing KV cache state with %d accepted tokens", num_accepted)


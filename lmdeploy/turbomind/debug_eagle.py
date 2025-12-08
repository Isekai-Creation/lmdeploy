"""
Helpers for inspecting EAGLE draft / target / accepted tokens.

These utilities are intended for offline debugging of TurboMind EAGLE3
behaviour. They operate purely on recorded token id sequences and a
tokenizer; they do not talk to the engine directly.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


def _to_int_list(ids: Optional[Iterable[int]]) -> List[int]:
    """Convert an iterable of ids (or None) to a plain list of ints."""
    if ids is None:
        return []
    # Handle common tensor-like objects that expose ``tolist``.
    if hasattr(ids, "tolist"):
        ids = ids.tolist()  # type: ignore[assignment]
    return [int(x) for x in ids]


def _decode(tokenizer, ids: Sequence[int]) -> str:
    """Decode a sequence of token ids using a generic tokenizer."""
    if not ids:
        return ""
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids)
    if hasattr(tokenizer, "detokenize"):
        return tokenizer.detokenize(ids)
    raise ValueError("Tokenizer must provide either 'decode' or 'detokenize'")


def format_eagle_alignment(
    draft_ids_path: Iterable[int],
    target_ids_path: Iterable[int],
    accepted_tokens: Iterable[int],
    tokenizer,
    baseline_text: Optional[str] = None,
) -> str:
    """Format a human-readable alignment for a single EAGLE tree path.

    Args:
        draft_ids_path: Token ids proposed by the draft model along a path.
        target_ids_path: Corresponding target ids from the base model.
        accepted_tokens: Tokens accepted by EAGLE along the path.
        tokenizer: Tokenizer with a ``decode`` or ``detokenize`` method.
        baseline_text: Optional baseline text (e.g. DynamicDecode output)
            to compare against the accepted tokens.

    Returns:
        A multi-line string describing ids and decoded text for each
        sequence, suitable for logging or printing.
    """
    draft_ids = _to_int_list(draft_ids_path)
    target_ids = _to_int_list(target_ids_path)
    accepted_ids = _to_int_list(accepted_tokens)

    draft_text = _decode(tokenizer, draft_ids)
    target_text = _decode(tokenizer, target_ids)
    accepted_text = _decode(tokenizer, accepted_ids)

    lines = []
    lines.append("EAGLE alignment (ids -> text):")
    lines.append(f"  draft_ids    = {draft_ids}")
    lines.append(f"  target_ids   = {target_ids}")
    lines.append(f"  accepted_ids = {accepted_ids}")
    lines.append(f"  draft_text   = {draft_text!r}")
    lines.append(f"  target_text  = {target_text!r}")
    lines.append(f"  accepted_text= {accepted_text!r}")

    if baseline_text is not None:
        lines.append(f"  baseline_text= {baseline_text!r}")

    return "\n".join(lines)


def print_eagle_alignment(
    draft_ids_path: Iterable[int],
    target_ids_path: Iterable[int],
    accepted_tokens: Iterable[int],
    tokenizer,
    baseline_text: Optional[str] = None,
) -> None:
    """Print a formatted EAGLE alignment for quick inspection."""
    print(
        format_eagle_alignment(
            draft_ids_path=draft_ids_path,
            target_ids_path=target_ids_path,
            accepted_tokens=accepted_tokens,
            tokenizer=tokenizer,
            baseline_text=baseline_text,
        )
    )


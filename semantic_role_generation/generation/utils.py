"""Text helpers shared by the prompt builder and the generator."""

import re
from typing import Iterable, List, Optional


def contains_whole_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def target_leaks(prompt: str, words: Iterable[str]) -> bool:
    """True if any target or foil/corrput word already appears in the prompt."""
    return any(contains_whole_word(prompt, w) for w in words)


def replace_last_scaffold(prompt: str, old: str, new: str) -> str:
    """Replace the scaffold that ends the prompt; unchanged if it is not there."""
    return re.sub(rf"{re.escape(old)}\s*$", new, prompt)


def find_scaffold_in_prompt(prompt: str, scaffolds: List[str]) -> Optional[str]:
    """Return the scaffold the prompt ends with, longest match first."""
    lowered = prompt.lower()
    for sc in sorted(scaffolds, key=lambda s: len(s.split()), reverse=True):
        if re.search(r"\b" + re.escape(sc) + r"\s*$", lowered):
            return sc
    return None


def build_full_sentence(clean: str, head: str) -> str:
    return f"{clean} {head}."

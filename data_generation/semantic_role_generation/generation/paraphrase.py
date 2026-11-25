"""Within-role scaffold paraphrases for sanity checks.

Rewrites each clean prompt with a different scaffold of the same role and
word length ("through the" -> "along the"), keeping targets and foils/corrupt.
"""

from typing import List, Optional, Tuple

import pandas as pd

from ...lexicon import Lexicon
from .utils import find_scaffold_in_prompt


def paraphrase_prompts(prompts: pd.DataFrame, role: str, lexicon: Lexicon,
                       scaffolds: Optional[List[str]] = None) -> Tuple[pd.DataFrame, dict]:
    """Return the paraphrased rows and counts of rows skipped."""
    scaffolds = scaffolds or lexicon.paraphrases(role)
    skipped = {"no_scaffold": 0, "no_alternative": 0}
    rows = []
    for _, row in prompts.iterrows():
        clean = str(row["clean"])
        sc = find_scaffold_in_prompt(clean, scaffolds)
        if sc is None:
            skipped["no_scaffold"] += 1
            continue
        alts = [s for s in scaffolds if s != sc and len(s.split()) == len(sc.split())]
        if not alts:
            skipped["no_alternative"] += 1
            continue
        new_row = row.copy()
        new_row["clean"] = _replace_last_words(clean, sc, alts[0])
        rows.append(new_row)
    return pd.DataFrame(rows, columns=prompts.columns), skipped


def _replace_last_words(prompt: str, old: str, new: str) -> str:
    """direct change to simple frame"""
    tokens, old_tokens = prompt.split(), old.split()
    m = len(old_tokens)
    for i in range(len(tokens) - m, -1, -1):
        if " ".join(w.lower() for w in tokens[i:i + m]) == old:
            return " ".join(tokens[:i] + new.split() + tokens[i + m:])
    return prompt

"""Generates IOI minimal pairs in the EAP-IG CSV format."""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..lexicon import Lexicon
from .config import IOIConfig
from .templates import TEMPLATE_SETS

IOI_COLUMNS = ["clean", "corrupted", "corrupted_hard", "correct_idx", "incorrect_idx",
               "target_word", "incorrect_word", "template_idx"]


class IOIGenerator:
    """Builds clean prompts ending before the indirect object, plus corruptions.

    clean:          "Then, Aaron and Amber went to the store. Aaron gave a drink to"
    corrupted:      giver replaced by a third name   (... store. Tyler gave a drink to)
    corrupted_hard: giver replaced by the IO          (... store. Amber gave a drink to)
    target = IO (" Amber"), corrupt head = S (" Aaron").

    Example:
        lex = Lexicon(LexiconConfig.default())
        rows = IOIGenerator(lex, IOIConfig.default(n_examples=500)).generate()
    """

    def __init__(self, lexicon: Lexicon, config: Optional[IOIConfig] = None):
        self.lex = lexicon
        self.config = config or IOIConfig.default()
        self.rng = random.Random(self.config.seed)
        templates = TEMPLATE_SETS[self.config.prompt_type]
        self.templates = templates[:self.config.n_templates] if self.config.n_templates else list(templates)
        if len(self.lex.names) < 3:
            raise ValueError("Need at least three single-token names")

    def generate(self) -> List[Dict]:
        rows = []
        while len(rows) < self.config.n_examples:
            template = self.rng.choice(self.templates)
            template_idx = self.templates.index(template)
            io, s, third = self.rng.sample(self.lex.names, 3)
            place, obj = self.rng.choice(self.lex.ioi_places), self.rng.choice(self.lex.ioi_objects)
            filled = template.replace("[PLACE]", place).replace("[OBJECT]", obj)
            rows.append(self._make_row(filled, io, s, third, template_idx))
            if self.config.symmetric and len(rows) < self.config.n_examples:
                rows.append(self._make_row(filled, s, io, third, template_idx))
        return rows

    def generate_all(self, write: bool = True) -> pd.DataFrame:
        df = pd.DataFrame(self.generate(), columns=IOI_COLUMNS)
        if write:
            out = Path(self.config.out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            with open(out.with_suffix(".report.json"), "w", encoding="utf-8") as f:
                json.dump({"config": dict(vars(self.config)), "tokenizer_name": self.lex.config.tokenizer_name,
                           "n_rows": len(df), "n_templates": len(self.templates)}, f, indent=2)
        return df

    # ---- helpers ----
    def _make_row(self, template: str, io: str, s: str, third: str, template_idx: int) -> Dict:
        full = template.replace("[A]", io).replace("[B]", s).replace("[C]", third)
        first, second = self._split_clauses(full)
        clean = self._truncate(first + second, io)
        corrupted = self._truncate(first + self._replace_giver(second, s, third), io)
        hard_name = io if self.config.hard_corruption else third
        corrupted_hard = self._truncate(first + self._replace_giver(second, s, hard_name), io)
        return {
            "clean": clean, "corrupted": corrupted, "corrupted_hard": corrupted_hard,
            "correct_idx": self.lex.token_id(io), "incorrect_idx": self.lex.token_id(s),
            "target_word": io, "incorrect_word": s, "template_idx": template_idx,
        }

    @staticmethod
    def _split_clauses(text: str):
        """Split after the first clause so the giver is replaced only in the second."""
        for sep in (". ", ", "):
            if sep in text:
                i = text.index(sep) + len(sep)
                return text[:i], text[i:]
        return "", text

    @staticmethod
    def _replace_giver(second_clause: str, giver: str, new_name: str) -> str:
        return re.sub(rf"\b{re.escape(giver)}\b", new_name, second_clause, count=1)

    @staticmethod
    def _truncate(text: str, io: str) -> str:
        """Drop the final indirect object so the prompt ends with 'to'."""
        return re.sub(rf"\s+{re.escape(io)}\s*$", "", text)

"""Generates greater-than prompts and the year-token metric that scores them."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..lexicon import Lexicon
from .config import GreaterThanConfig

GREATER_THAN_COLUMNS = ["clean", "corrupted", "label"]


class GreaterThanGenerator:
    """Builds year-completion prompts with a YY -> 01 corruption.

    Example:
        lex = Lexicon(LexiconConfig.default())
        df = GreaterThanGenerator(lex, GreaterThanConfig.default()).generate_all()
    """

    def __init__(self, lexicon: Lexicon, config: Optional[GreaterThanConfig] = None):
        self.lex = lexicon
        self.config = config or GreaterThanConfig.default()
        self.rng = random.Random(self.config.seed)
        self.nouns = list(lexicon.greater_than_nouns)
        self._check_year_tokens()

    def generate(self) -> List[Dict]:
        cfg = self.config
        rows = []
        for _ in range(cfg.n_examples):
            noun = self.rng.choice(self.nouns)
            century = self.rng.choice(cfg.centuries)
            year = self.rng.randint(cfg.year_min, cfg.year_max)
            rows.append({
                "clean": cfg.template.format(noun=noun, century=century, year=year),
                "corrupted": cfg.template.format(noun=noun, century=century, year=cfg.corrupted_year),
                "label": year,
            })
        return rows

    def generate_all(self, write: bool = True) -> pd.DataFrame:
        df = pd.DataFrame(self.generate(), columns=GREATER_THAN_COLUMNS)
        if write:
            out = Path(self.config.out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            with open(out.with_suffix(".report.json"), "w", encoding="utf-8") as f:
                json.dump({"config": dict(vars(self.config)), "tokenizer_name": self.lex.config.tokenizer_name,
                           "n_rows": len(df)}, f, indent=2)
        return df

    def year_token_ids(self) -> List[int]:
        """Token id of each two-digit year "00".."99" as the model would complete it."""
        return [self.lex.ids(f"{y:02d}")[0] for y in range(100)]

    def _check_year_tokens(self):
        bad = [y for y in range(100) if len(self.lex.ids(f"{y:02d}")) != 1]
        if bad:
            raise ValueError(f"Two-digit years are not single tokens for this tokenizer: {bad[:5]}...")

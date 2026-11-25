"""Generates per-role next-token datasets and writes them to disk."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..lexicon import Lexicon
from .config import GenerationConfig
from .prompts import PromptBuilder
from .utils import build_full_sentence, target_leaks

PROMPT_COLUMNS = ["clean", "corrupted", "corrupted_hard", "correct_idx", "incorrect_idx",
                  "target_word", "incorrect_word", "target_role", "corrupt_role"]
INTERVENTION_COLUMNS = ["clean", "corrupted", "intervention_type", "target_role", "target_word",
                        "replacement_word", "position", "sentence_idx", "semantic_frame",
                        "correct_idx", "incorrect_idx", "sensicality"]
DROP_REASONS = ["parity", "leak", "multi_token", "no_pair", "identical", "other"]


class RoleDatasetGenerator:
    """Generates clean/corrupted minimal pairs for semantic roles.

    Example:
        lex = Lexicon(LexiconConfig.default())
        gen = RoleDatasetGenerator(lex, GenerationConfig.default(out_dir="./data/semroles"))
        report = gen.generate_all()
    """

    def __init__(self, lexicon: Lexicon, config: Optional[GenerationConfig] = None):
        self.lex = lexicon
        self.config = config or GenerationConfig.default()
        self.rng = random.Random(self.config.seed)
        self.builder = PromptBuilder(lexicon, self.config, self.rng)
        self.roles = list(self.config.roles or lexicon.roles)

    def generate_role(self, role: str) -> Dict:
        """Build examples for one role; returns prompt rows, intervention rows and records."""
        cfg, lex, b = self.config, self.lex, self.builder
        prompts, interventions, records = [], [], []
        kept, attempts = 0, 0
        drops = {k: 0 for k in DROP_REASONS}

        while kept < cfg.n_per_role and attempts < cfg.n_per_role * cfg.patience:
            attempts += 1
            p = b.make_clean(role)
            if not lex.is_single_token(p.target):
                drops["multi_token"] += 1; continue
            if target_leaks(p.text, [p.target]):
                drops["leak"] += 1; continue

            c = b.make_role_cue_corruption(p)
            if c is None:
                drops["no_pair"] += 1; continue
            if c.text == p.text and cfg.skip_identical_corruption:
                drops["identical"] += 1; continue
            if lex.token_length(c.text) != lex.token_length(p.text):
                drops["parity"] += 1; continue
            if target_leaks(c.text, [p.target, c.corrupt]):
                drops["leak"] += 1; continue

            hard = b.make_corrupted_hard(p)
            if target_leaks(hard, [p.target, c.corrupt]):
                drops["leak"] += 1; continue

            intervs = b.make_interventions(p)
            if not intervs:
                drops["other"] += 1; continue

            full = build_full_sentence(p.text, p.target)
            prompts.append({
                "clean": p.text, "corrupted": c.text, "corrupted_hard": hard,
                "correct_idx": lex.token_id(p.target), "incorrect_idx": lex.token_id(c.corrupt),
                "target_word": p.target, "incorrect_word": c.corrupt,
                "target_role": role, "corrupt_role": c.corrupt_role,
            })
            for iv in intervs:
                interventions.append({
                    "clean": full, "corrupted": iv["full_sentence"],
                    "intervention_type": iv["type"], "target_role": role,
                    "target_word": p.target, "replacement_word": iv["replacement_head"],
                    "position": "last", "sentence_idx": kept, "semantic_frame": p.frame,
                    "correct_idx": lex.token_id(p.target),
                    "incorrect_idx": lex.token_id(iv["replacement_head"]),
                    "sensicality": iv["sensicality"],
                })
            records.append(self._record(role, kept, p, c, full, intervs))
            kept += 1

        return {"prompts": prompts, "interventions": interventions, "records": records,
                "kept": kept, "attempts": attempts, "drops": drops}

    def generate_all(self, write: bool = True) -> Dict[str, Dict]:
        """Generate every configured role and write a report; returns per-role counts."""
        out_dir = Path(self.config.out_dir)
        report = {}
        for role in self.roles:
            result = self.generate_role(role)
            if write:
                self.write_role(role, result, out_dir / role)
            report[role] = {k: result[k] for k in ("kept", "attempts", "drops")}
            print(f"[{role}] kept={result['kept']} attempts={result['attempts']} drops={result['drops']}")
        if write:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "report.json", "w", encoding="utf-8") as f:
                json.dump({"config": self._config_dict(), "roles": report}, f, indent=2)
        return report

    def write_role(self, role: str, result: Dict, role_dir: Path) -> None:
        role_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result["prompts"], columns=PROMPT_COLUMNS).to_csv(role_dir / "prompts.csv", index=False)
        pd.DataFrame(result["interventions"], columns=INTERVENTION_COLUMNS).to_csv(
            role_dir / "interventions.csv", index=False)
        with open(role_dir / "dataset.jsonl", "w", encoding="utf-8") as f:
            for rec in result["records"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- helpers ----
    def _record(self, role, idx, p, c, full, intervs) -> Dict:
        args = [{"role": "Agent", "head": p.agent}]
        if p.theme is not None:
            args.append({"role": "Theme", "head": p.theme})
        args.append({"role": role, "scaffold": p.scaffold, "head": p.target})
        return {
            "base": {
                "id": f"{role.lower()}_{idx:04d}",
                "verb": p.verb,
                "verb_type": "stative" if role in ("Time", "Manner") else "action",
                "arguments": args,
                "full_sentence": full,
                "clean_incomplete": p.text,
                "corrupted": c.text,
                "next_token_target": p.target,
                "corrupt": c.corrupt,
                "sentence_sensicality": "sensical",
                "metadata": {
                    "semantic_frame": p.frame, "perturbation_position": "last",
                    "target_role": role, "scaffold": p.scaffold, "cue_swap_to_role": c.corrupt_role,
                },
            },
            "interventions": intervs,
        }

    def _config_dict(self) -> Dict:
        d = dict(vars(self.config))
        d["tokenizer_name"] = self.lex.config.tokenizer_name
        d["scaffold_markers"] = self.lex.config.scaffold_markers
        return d

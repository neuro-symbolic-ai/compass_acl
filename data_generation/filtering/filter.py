"""Scores generated minimal pairs with a model and writes the filtered splits."""

import glob
import json
import os
import random
from typing import Dict, List, Optional

import pandas as pd

from .config import FilterConfig
from .scoring import load_model, next_token_probs, prefers_target, space_token_id
from .utils import read_role_folder, write_split

CONTEXT_PREFIX = {"clean": "clean", "corrupted": "corr", "corrupted_hard": "hard"}


class RoleFilter:
    """Keeps the minimal pairs a model answers correctly.

    For every prompt column in config.contexts the model's next-token
    probabilities for the target and corrupt heads are recorded. A pair "wins"
    a context when the target outranks every corrupt head. Splits are defined
    by win patterns, for example both_correct = win in all three contexts.

    Example:
        f = RoleFilter(FilterConfig.default(model_name="EleutherAI/pythia-1b", device="cuda"))
        coverage = f.filter_all("./data/semroles_v1", "./data/semroles_v1_filtered")
    """

    def __init__(self, config: Optional[FilterConfig] = None, tokenizer=None, model=None):
        self.config = config or FilterConfig.default()
        random.seed(self.config.seed)
        if tokenizer is None or model is None:
            tokenizer, model = load_model(self.config.model_name, self.config.device,
                                          self.config.half_precision_on_cuda)
        self.tok, self.model = tokenizer, model

    # ---- scoring ----
    def score_role(self, prompts: pd.DataFrame, role_name: str) -> pd.DataFrame:
        """Return prompts with probability and win columns for each context."""
        cfg = self.config
        target_ids, corrupt_id_lists, ok = [], [], []
        for _, row in prompts.iterrows():
            try:
                tid = space_token_id(self.tok, str(row["target_word"]))
                cids = [space_token_id(self.tok, str(row["incorrect_word"]))]
                target_ids.append(tid); corrupt_id_lists.append(cids); ok.append(True)
            except ValueError:
                target_ids.append(None); corrupt_id_lists.append([]); ok.append(False)
        valid = prompts[ok].reset_index(drop=True)
        if valid.empty:
            return valid
        target_ids = [t for t, k in zip(target_ids, ok) if k]
        corrupt_id_lists = [c for c, k in zip(corrupt_id_lists, ok) if k]
        candidates = sorted({t for t in target_ids} | {c for cl in corrupt_id_lists for c in cl})

        probs = {ctx: next_token_probs(self.tok, self.model, valid[ctx].tolist(), candidates,
                                       cfg.device, cfg.batch_size) for ctx in cfg.contexts}
        rows = []
        for i, row in valid.iterrows():
            tid, cids = target_ids[i], corrupt_id_lists[i]
            rec = {**row.to_dict(), "correct_idx": tid, "incorrect_idx": cids[0],
                   "target_token_id": tid, "corrupt_token_ids": ";".join(map(str, cids))}
            for ctx in cfg.contexts:
                p = probs[ctx][i]; pre = CONTEXT_PREFIX.get(ctx, ctx)
                rec[f"{pre}_p_target"] = p.get(tid, 0.0)
                rec[f"{pre}_p_corruptmax"] = max((p.get(c, 0.0) for c in cids), default=0.0)
                rec[f"{pre}_win"] = prefers_target(p, tid, cids)
            rec["target_role"] = row.get("target_role", role_name)
            rows.append(rec)
        return pd.DataFrame(rows)

    def select_split(self, scored: pd.DataFrame, pattern: List[bool]) -> pd.DataFrame:
        mask = pd.Series(True, index=scored.index)
        for ctx, want in zip(self.config.contexts, pattern):
            col = f"{CONTEXT_PREFIX.get(ctx, ctx)}_win"
            mask &= scored[col] if want else ~scored[col]
        return scored[mask].copy()

    # ---- pipeline ----
    def filter_role(self, role_dir: str, out_dir: str) -> Optional[Dict]:
        loaded = read_role_folder(role_dir)
        if loaded is None:
            return None
        prompts, interventions, records = loaded
        role_name = os.path.basename(os.path.normpath(role_dir))
        scored = self.score_role(prompts, role_name)
        if scored.empty:
            return {"input_rows": len(prompts), "scored_rows": 0, "skipped": "no single-token rows"}
        scored["role"] = role_name
        info = {"input_rows": len(prompts), "scored_rows": len(scored)}
        for split_name, pattern in self.config.splits.items():
            subset = self.select_split(scored, pattern)
            write_split(subset, interventions, records, os.path.join(out_dir, role_name, split_name),
                        self.config.split_by_role)
            info[split_name] = len(subset)
        if self.config.save_diagnostics:
            diag = os.path.join(out_dir, "diagnostics"); os.makedirs(diag, exist_ok=True)
            scored.to_csv(os.path.join(diag, f"{role_name}_scored.csv"), index=False)
        info["_scored"] = scored
        return info

    def filter_all(self, data_root: str, out_dir: str) -> Dict:
        """Filter every role folder under data_root; returns and writes coverage.json."""
        os.makedirs(out_dir, exist_ok=True)
        role_dirs = sorted(p for p in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(p))
        coverage, all_scored = {"model": self.config.model_name, "roles": {}}, []
        for role_dir in role_dirs:
            info = self.filter_role(role_dir, out_dir)
            if info is None:
                continue
            scored = info.pop("_scored", None)
            if scored is not None:
                all_scored.append(scored)
            coverage["roles"][os.path.basename(role_dir)] = info
            print(f"[{os.path.basename(role_dir)}] {info}")
        if all_scored and self.config.save_diagnostics:
            pd.concat(all_scored, ignore_index=True).to_csv(
                os.path.join(out_dir, "diagnostics", "all_scored.csv"), index=False)
        with open(os.path.join(out_dir, "coverage.json"), "w", encoding="utf-8") as f:
            json.dump(coverage, f, indent=2)
        return coverage

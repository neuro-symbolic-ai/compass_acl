"""Behavioural statistics of the filtered pairs: how strongly the model prefers the target."""

from typing import Tuple

import numpy as np
import pandas as pd

EPS = 1e-12 # a param for


def compute_behavioral_stats(scored: pd.DataFrame, subset_cols=("clean_win", "corr_win", "hard_win")
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-role summary of probability and log-probability gaps.

    Operates on the rows that win every context in subset_cols (the
    both_correct subset by default). Returns (per_role_stats, annotated_rows).
    """
    cols = [c for c in subset_cols if c in scored.columns]
    bc = scored[np.logical_and.reduce([scored[c] for c in cols])].copy() if cols else scored.copy()
    if bc.empty:
        return pd.DataFrame(), bc

    for pre in ("clean", "corr", "hard"):
        if f"{pre}_p_target" not in bc.columns:
            continue
        t, c = bc[f"{pre}_p_target"], bc[f"{pre}_p_corruptmax"]
        bc[f"{pre}_prob_gap"] = t - c
        bc[f"{pre}_logprob_gap"] = np.log(t.clip(lower=EPS)) - np.log(c.clip(lower=EPS))
        bc[f"{pre}_prob_ratio"] = t / c.clip(lower=EPS)
    if "corr_p_target" in bc.columns:
        bc["cross_context_prob_gap"] = bc["clean_p_target"] - bc["corr_p_target"]
        bc["cross_context_logprob_gap"] = (np.log(bc["clean_p_target"].clip(lower=EPS))
                                          - np.log(bc["corr_p_target"].clip(lower=EPS)))

    role_col = "role" if "role" in bc.columns else "target_role"
    rows = []
    for role, g in bc.groupby(role_col):
        s = {"role": role, "n": len(g)}
        for pre in ("clean", "corr", "hard"):
            if f"{pre}_p_target" not in g.columns:
                continue
            s[f"{pre}_p_target_mean"] = g[f"{pre}_p_target"].mean()
            s[f"{pre}_p_corrupt_mean"] = g[f"{pre}_p_corruptmax"].mean()
            s[f"{pre}_prob_gap_mean"] = g[f"{pre}_prob_gap"].mean()
            s[f"{pre}_prob_gap_median"] = g[f"{pre}_prob_gap"].median()
            s[f"{pre}_logprob_gap_mean"] = g[f"{pre}_logprob_gap"].mean()
            s[f"{pre}_prob_ratio_median"] = g[f"{pre}_prob_ratio"].median()
        if "cross_context_prob_gap" in g.columns:
            s["cross_context_prob_gap_mean"] = g["cross_context_prob_gap"].mean()
            s["cross_context_logprob_gap_mean"] = g["cross_context_logprob_gap"].mean()
        rows.append(s)
    return pd.DataFrame(rows), bc


def print_stats(stats: pd.DataFrame, bc: pd.DataFrame) -> None:
    if stats.empty:
        print("No rows in the selected subset; nothing to summarise.")
        return
    print(f"\n{'Role':<13}{'N':>6}{'p(tgt|clean)':>14}{'p(corr|clean)':>14}{'gap':>9}{'logp gap':>10}")
    for _, r in stats.iterrows():
        print(f"{r['role']:<13}{r['n']:>6.0f}{r['clean_p_target_mean']:>14.4f}"
              f"{r['clean_p_corrupt_mean']:>14.4f}{r['clean_prob_gap_mean']:>9.4f}{r['clean_logprob_gap_mean']:>10.2f}")
    print(f"\nTotal rows: {len(bc)}; mean clean log-prob gap "
          f"{bc['clean_logprob_gap'].mean():.2f} ± {bc['clean_logprob_gap'].std():.2f}")
    pcts = [5, 25, 50, 75, 95]
    vals = np.percentile(bc["clean_logprob_gap"], pcts)
    print("Clean log-prob gap percentiles: " + ", ".join(f"p{p}={v:.2f}" for p, v in zip(pcts, vals)))

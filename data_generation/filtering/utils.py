"""Reading generated role folders and writing filtered splits."""

import json
import os
from typing import Dict, Optional, Tuple

import pandas as pd

REQUIRED_COLUMNS = ["clean", "corrupted", "target_word", "incorrect_word"]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"prompts.csv is missing columns {missing}")
    if "corrupted_hard" not in df.columns:
        df["corrupted_hard"] = df["corrupted"]
    return df


def read_role_folder(role_dir: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]]:
    """Load prompts.csv, interventions.csv and dataset.jsonl from one role folder.

    Returns (prompts, interventions, records_by_full_sentence) or None when
    prompts.csv is absent. The JSONL is optional; without it data.jsonl in the
    splits is left empty.
    """
    p_csv = os.path.join(role_dir, "prompts.csv")
    if not os.path.exists(p_csv):
        return None
    prompts = ensure_columns(pd.read_csv(p_csv))
    i_csv = os.path.join(role_dir, "interventions.csv")
    interventions = pd.read_csv(i_csv) if os.path.exists(i_csv) else pd.DataFrame()

    records = {}
    j_path = os.path.join(role_dir, "dataset.jsonl")
    if os.path.exists(j_path):
        with open(j_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                records[obj["base"]["full_sentence"]] = obj
    if "target_role" not in prompts.columns:
        prompts["target_role"] = os.path.basename(os.path.normpath(role_dir))
    return prompts, interventions, records


def full_sentences(df: pd.DataFrame) -> pd.Series:
    return df["clean"] + " " + df["target_word"] + "."


def write_split(rows: pd.DataFrame, interventions: pd.DataFrame, records: Dict,
                out_dir: str, split_by_role: bool = True) -> Dict[str, int]:
    """Write prompts.csv, interventions.csv and data.jsonl for a split."""
    os.makedirs(out_dir, exist_ok=True)
    _write_bundle(rows, interventions, records, out_dir)
    counts = {}
    if split_by_role and "target_role" in rows.columns:
        for role, g in rows.groupby("target_role"):
            _write_bundle(g, interventions, records, os.path.join(out_dir, str(role).lower()))
            counts[str(role)] = len(g)
    return counts


def _write_bundle(rows, interventions, records, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows.to_csv(os.path.join(out_dir, "prompts.csv"), index=False)
    fulls = full_sentences(rows).tolist()
    if interventions is not None and not interventions.empty:
        interventions[interventions["clean"].isin(fulls)].to_csv(
            os.path.join(out_dir, "interventions.csv"), index=False)
    with open(os.path.join(out_dir, "data.jsonl"), "w", encoding="utf-8") as f:
        for full in fulls:
            if full in records:
                f.write(json.dumps(records[full], ensure_ascii=False) + "\n")

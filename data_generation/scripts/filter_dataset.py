"""Filter generated minimal pairs by a model's next-token behaviour.

Run from the repository root:
    python -m semantic_role_generation.scripts.filter_dataset \
        --data_root ./data/semroles_v1 --out_dir ./data/semroles_v1_filtered \
        --model_name EleutherAI/pythia-1b --device cuda

Use --stats_only to recompute the behavioural statistics from an existing
diagnostics/all_scored.csv without loading a model.
"""

import argparse
import os

import pandas as pd

from ..filtering import FilterConfig, RoleFilter, compute_behavioral_stats, print_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, help="Folder with one subfolder per role")
    ap.add_argument("--out_dir", required=True, help="Output folder for the filtered splits")
    ap.add_argument("--model_name", default="EleutherAI/pythia-1b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats_only", action="store_true", help="Summarise existing diagnostics only")
    args = ap.parse_args()

    diag = os.path.join(args.out_dir, "diagnostics")
    if args.stats_only:
        path = os.path.join(diag, "all_scored.csv")
        if not os.path.exists(path):
            raise SystemExit(f"No diagnostics at {path}; run without --stats_only first")
        scored = pd.read_csv(path)
    else:
        config = FilterConfig.default(model_name=args.model_name, device=args.device,
                                      batch_size=args.batch_size, seed=args.seed)
        RoleFilter(config).filter_all(args.data_root, args.out_dir)
        scored = pd.read_csv(os.path.join(diag, "all_scored.csv"))

    stats, bc = compute_behavioral_stats(scored)
    print_stats(stats, bc)
    stats.to_csv(os.path.join(diag, "behavioral_stats_per_role.csv"), index=False)
    bc.to_csv(os.path.join(diag, "both_correct_with_stats.csv"), index=False)


if __name__ == "__main__":
    main()

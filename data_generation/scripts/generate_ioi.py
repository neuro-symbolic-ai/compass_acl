"""Generate IOI minimal pairs as a CSV.

Run from the repository root:
    python -m semantic_role_generation.scripts.generate_ioi \
        --tokenizer EleutherAI/pythia-1b --out ./data/ioi/ioi_pairs.csv --n 1000
"""

import argparse

from ..ioi import IOIConfig, IOIGenerator
from ..lexicon import Lexicon, LexiconConfig
from ..ioi.templates import TEMPLATE_SETS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="EleutherAI/pythia-1b")
    ap.add_argument("--out", default="./data/ioi/ioi_pairs.csv")
    ap.add_argument("--n", type=int, default=1000, help="Number of pairs")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--prompt_type", choices=list(TEMPLATE_SETS), default="BABA")
    ap.add_argument("--symmetric", action="store_true", help="Also emit the A/B-swapped twin of each prompt")
    args = ap.parse_args()

    lexicon = Lexicon(LexiconConfig(tokenizer_name=args.tokenizer))
    print(f"{len(lexicon.names)} single-token names")
    config = IOIConfig(out_path=args.out, n_examples=args.n, seed=args.seed,
                       prompt_type=args.prompt_type, symmetric=args.symmetric)
    df = IOIGenerator(lexicon, config).generate_all()
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

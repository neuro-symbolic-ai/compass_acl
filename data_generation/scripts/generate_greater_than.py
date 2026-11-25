"""Generate greater-than prompts as a CSV.

Run from the repository root:
    python -m semantic_role_generation.scripts.generate_greater_than \
        --tokenizer gpt2 --out ./data/numbers/greater_than.csv --n 1000
"""

import argparse

from ..lexicon import Lexicon, LexiconConfig
from ..numbers import GreaterThanConfig, GreaterThanGenerator


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--out", default="./data/numbers/greater_than.csv")
    ap.add_argument("--n", type=int, default=1000, help="Number of prompts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lexicon = Lexicon(LexiconConfig(tokenizer_name=args.tokenizer, include_names=False))
    config = GreaterThanConfig(out_path=args.out, n_examples=args.n, seed=args.seed)
    df = GreaterThanGenerator(lexicon, config).generate_all()
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

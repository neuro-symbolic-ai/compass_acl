"""Generate semantic-role next-token datasets.

Run from the repository root:
    python -m semantic_role_generation.scripts.generate_dataset \
        --tokenizer EleutherAI/pythia-1b --out_dir ./data/semroles_v1 --n_per_role 1500
"""

import argparse

from ..semantic_role_generation.generation import GenerationConfig, RoleDatasetGenerator
from ..lexicon import Lexicon, LexiconConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="EleutherAI/pythia-1b", help="Hugging Face tokenizer name")
    ap.add_argument("--out_dir", default="./data/semroles", help="Output folder, one subfolder per role")
    ap.add_argument("--n_per_role", type=int, default=1500, help="Examples to keep per role")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--roles", nargs="*", default=None, help="Subset of roles; default is all ten")
    ap.add_argument("--scaffold_markers", choices=["1token", "2token"], default="1token",
                    help="Scaffold family; 2token lets Time swap with other roles")
    ap.add_argument("--legacy", action="store_true",
                    help="Reproduce the original generator, keeping identical clean/corrupted pairs")
    args = ap.parse_args()

    lexicon = Lexicon(LexiconConfig(tokenizer_name=args.tokenizer, scaffold_markers=args.scaffold_markers,
                                    roles=args.roles))
    print("Lexicon coverage:", lexicon.summary())
    config = GenerationConfig(
        out_dir=args.out_dir,
        n_per_role=args.n_per_role,
        seed=args.seed,
        roles=args.roles,
        skip_identical_corruption=not args.legacy,
    )
    RoleDatasetGenerator(lexicon, config).generate_all()
    print("Done. Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()

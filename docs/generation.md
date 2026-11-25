# Data Generation

This module builds the contrastive minimal pairs used for circuit discovery. For each semantic role it writes clean prompts that end at the role's scaffold, a corrupted prompt whose scaffold cues a different role, a harder corruption with a preface clause, and same-role interventions on the full sentence.

The module is imported as `data_generation.semantic_role_generation` (the `generation` subpackage is re-exported there) and takes its word inventories from the [lexicon](lexicon.md).

## Example

| Field | Value |
| --- | --- |
| clean | `The chef handed the book to the` |
| target | `student` (Goal) |
| corrupted | `The chef handed the book in the` |
| corrupt head | `garden` (Location) |
| corrupted_hard | `After a short delay, the chef handed the book to the` |
| intervention | `The chef handed the book to the student.` → `... to the nurse.` |

Clean and corrupted prompts always have the same number of tokens, so activations can be patched position by position.

## Configuration

```python
from data_generation.semantic_role_generation import GenerationConfig

config = GenerationConfig(
    out_dir="./data/semroles",
    n_per_role=1500,                  # examples to keep per role
    seed=1337,
    roles=None,                       # None uses every role in the lexicon
    patience=30,                      # give up after n_per_role * patience attempts
    adverb_prob=0.3,                  # chance of an adverb after the subject
    adjective_prob=0.25,              # chance of an adjective on the subject or object
    allow_hard_longer=True,           # corrupted_hard adds a preface clause
    n_interventions=2,                # same-role head swaps per example
    skip_identical_corruption=True,   # reject pairs whose corrupted prompt equals the clean one
    allow_same_role_fallback=True,    # use a same-role corrupt head if no cross-role scaffold fits
)
```

Presets:

```python
GenerationConfig.default()   # paper setting, identical pairs rejected
GenerationConfig.legacy()    # original main.py behaviour, identical pairs kept
GenerationConfig.small()     # 50 examples per role, for smoke tests
```

## Generating a dataset

```python
from data_generation.lexicon import Lexicon, LexiconConfig
from data_generation.semantic_role_generation import RoleDatasetGenerator, GenerationConfig

lexicon = Lexicon(LexiconConfig.default(tokenizer_name="EleutherAI/pythia-1b"))
generator = RoleDatasetGenerator(lexicon, GenerationConfig.default(out_dir="./data/semroles_v1"))

report = generator.generate_all()
# {'Goal': {'kept': 1500, 'attempts': 1512, 'drops': {'parity': 0, 'leak': 12, ...}}, ...}
```

Or from the command line, from the repository root:

```bash
python -m data_generation.scripts.generate_dataset \
    --tokenizer EleutherAI/pythia-1b --out_dir ./data/semroles_v1 --n_per_role 1500 --seed 42
```

| Argument | Purpose |
| --- | --- |
| `--tokenizer` | Hugging Face tokenizer used to filter words and check token parity. |
| `--out_dir` | Output folder, one subfolder per role. |
| `--n_per_role` | Examples to keep per role (default: `1500`). |
| `--seed` | Random seed (default: `42`; `GenerationConfig` itself defaults to `1337`). |
| `--roles` | Subset of roles, for example `--roles Goal Location`. |
| `--scaffold_markers` | `1token` (paper setting) or `2token`; see the lexicon guide. |
| `--legacy` | Reproduce the original generator, keeping identical clean/corrupted pairs. |

A single role can be generated without writing files:

```python
result = generator.generate_role("Goal")
result["prompts"][0]        # dict with the prompts.csv columns
result["drops"]             # why attempts were rejected
```

## Outputs

```text
<out_dir>/
├── report.json             # config used (including tokenizer) and per-role counts
└── <Role>/
    ├── prompts.csv         # clean, corrupted, corrupted_hard, correct_idx, incorrect_idx,
    │                       #   target_word, incorrect_word, target_role, corrupt_role
    ├── interventions.csv   # full sentence and its same-role head swap, with token ids
    └── dataset.jsonl       # one record per example: verb, arguments, scaffold, metadata
```

`correct_idx` and `incorrect_idx` are the token ids of the target and corrupt heads for the chosen tokenizer. `prompts.csv` has the four columns that circuit discovery reads: `clean`, `corrupted`, `correct_idx` and `incorrect_idx`.

## How a pair is built

1. **Clean prompt.** `PromptBuilder.make_clean` samples a verb from the role's frame, a subject, a direct object for roles that take one, a scaffold and a target head. Modifiers are added with the configured probabilities.
2. **Role-cue corruption.** `make_role_cue_corruption` replaces the scaffold with one from a role in `ROLE_CUE_SWAP` that keeps the token count, and picks a corrupt head from that role. If no scaffold fits and `allow_same_role_fallback` is on, the prompt is kept and the corrupt head comes from the same role.
3. **Hard corruption.** `make_corrupted_hard` prefixes a preface clause; this prompt may be longer than the clean one.
4. **Checks.** An attempt is dropped if the target is not a single token, if the target or corrupt head already appears in a prompt, if token parity fails, or if the pair is identical and `skip_identical_corruption` is on. Each reason is counted in `drops`.
5. **Interventions.** `make_interventions` swaps the target head for other heads of the same role in the full sentence.

## Paraphrased prompts

For sanity checks, each clean prompt can be rewritten with another scaffold of the same role and length:

```python
import pandas as pd
from data_generation.semantic_role_generation import paraphrase_prompts

prompts = pd.read_csv("./data/semroles_v1/Path/prompts.csv")
paraphrased, skipped = paraphrase_prompts(prompts, "Path", lexicon)
# "walked through the" -> "walked along the"; skipped counts rows with no alternative
```

The result keeps the target and corrupt columns, so it can be filtered like any other prompts file.

## Differences from the original scripts

- `main.py`, `main_updated_time.py` and `main_ioi.py` are one generator; the Time variant is `LexiconConfig.updated_time()`.
- Identical clean/corrupted pairs are rejected by default (`GenerationConfig.legacy()` restores them).
- When several scaffolds fit, one is chosen at random rather than the first in list order.
- The verb recorded in `dataset.jsonl` is the verb in the sentence.
- `prompts.csv` carries `target_role` and `corrupt_role`, and `report.json` records the configuration.

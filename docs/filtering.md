# Filtering

This module scores generated minimal pairs with a language model and keeps the pairs the model answers correctly, so that circuit discovery runs on behaviour the model actually has. It works on the semantic-role folders and on any CSV with the same columns, such as the [IOI](ioi.md) pairs.

The module is imported as `data_generation.filtering`.

## What is scored

For each prompt column in `contexts` (by default `clean`, `corrupted` and `corrupted_hard`) the model's next-token probability is recorded for the target head and the corrupt head. A pair **wins** a context when the target outranks the corrupt head. Splits are defined by win patterns:

| Split | clean | corrupted | corrupted_hard |
| --- | --- | --- | --- |
| `both_correct` | win | win | win |
| `clean_only_correct` | win | lose | lose |

`both_correct` is the split used for circuit discovery: the model prefers the target in the clean prompt and still prefers it under the corrupted cue, so the two prompts differ in the activations that carry the role, not in whether the model knows the answer.

## Configuration

```python
from data_generation.filtering import FilterConfig

config = FilterConfig(
    model_name="EleutherAI/pythia-1b",
    device="cuda",
    batch_size=64,
    seed=3,
    contexts=["clean", "corrupted", "corrupted_hard"],
    splits={
        "both_correct": [True, True, True],
        "clean_only_correct": [True, False, False],
    },
    save_diagnostics=True,            # write per-role and combined scored CSVs
    split_by_role=True,               # also write <split>/<role>/ folders
    half_precision_on_cuda=True,
)
```

Presets:

```python
FilterConfig.default()       # the three contexts and two splits above
FilterConfig.clean_only()    # score only the clean prompt; one split, clean_correct
```

## Filtering a dataset

```python
from data_generation.filtering import RoleFilter, FilterConfig

role_filter = RoleFilter(FilterConfig.default(model_name="EleutherAI/pythia-1b", device="cuda"))
coverage = role_filter.filter_all("./data/semroles_v1", "./data/semroles_v1_filtered")
# {'model': 'EleutherAI/pythia-1b', 'roles': {'Goal': {'input_rows': 1500, 'scored_rows': 1500,
#                                                      'both_correct': 812, 'clean_only_correct': 93}, ...}}
```

Or from the command line, from the repository root:

```bash
python -m data_generation.scripts.filter_dataset \
    --data_root ./data/semroles_v1 --out_dir ./data/semroles_v1_filtered \
    --model_name EleutherAI/pythia-1b --device cuda
```

| Argument | Purpose |
| --- | --- |
| `--data_root` | Folder written by the generator, one subfolder per role. |
| `--out_dir` | Output folder for the splits and diagnostics. |
| `--model_name` | Hugging Face causal language model (default: `EleutherAI/pythia-1b`). |
| `--device` | `cpu` or `cuda` (default: `cpu`). |
| `--batch_size` | Prompts per forward pass (default: `64`). |
| `--seed` | Random seed (default: `3`). |
| `--stats_only` | Recompute the statistics from an existing `diagnostics/all_scored.csv` without a model. |

A loaded tokenizer and model can be reused instead of loading them again:

```python
role_filter = RoleFilter(config, tokenizer=tokenizer, model=model)
```

## Outputs

```text
<out_dir>/
├── coverage.json                       # model name and per-role counts
├── diagnostics/
│   ├── <Role>_scored.csv               # every scored pair with probabilities and wins
│   ├── all_scored.csv
│   ├── behavioral_stats_per_role.csv   # written by the script
│   └── both_correct_with_stats.csv
└── <Role>/
    ├── both_correct/
    │   ├── prompts.csv
    │   ├── interventions.csv
    │   ├── data.jsonl
    │   └── <role>/                     # the same files, one folder per target role
    └── clean_only_correct/
        └── ...
```

The scored CSVs add, for each context prefix (`clean`, `corr`, `hard`): `<prefix>_p_target`, `<prefix>_p_corruptmax` and `<prefix>_win`, plus `target_token_id` and `corrupt_token_ids`. The `both_correct/prompts.csv` of each role is the file to point circuit discovery at (`cnp_dataset_path` in the training configuration).

## Scoring details

`next_token_probs` pads a batch and reads each prompt's logits at its own last token, taken from the attention mask, so the result does not depend on the tokenizer's padding side or on which other prompts share the batch. Only the probabilities of the candidate tokens are kept.

```python
from data_generation.filtering import load_model, next_token_probs, space_token_id, prefers_target

tokenizer, model = load_model("EleutherAI/pythia-1b", device="cuda")
target = space_token_id(tokenizer, "student")
corrupt = space_token_id(tokenizer, "garden")

probs = next_token_probs(tokenizer, model, ["The chef handed the book to the"], [target, corrupt], "cuda")
prefers_target(probs[0], target, [corrupt])   # True if p(student) > p(garden)
```

`space_token_id` raises a `ValueError` for words that are not a single token with a leading space; such rows are skipped and counted in `coverage.json`.

## Behavioural statistics

```python
import pandas as pd
from data_generation.filtering import compute_behavioral_stats, print_stats

scored = pd.read_csv("./data/semroles_v1_filtered/diagnostics/all_scored.csv")
stats, both_correct = compute_behavioral_stats(scored)
print_stats(stats, both_correct)
```

`stats` has one row per role with the mean target and corrupt probabilities, probability gap, log-probability gap and probability ratio in each context, and the cross-context gap between the clean and corrupted prompts. `both_correct` is the subset the statistics were computed on, with the gap columns added.

## Differences from the original script

- Logits are read at each prompt's true last token. The original read position `-1` of a right-padded batch, which for most prompts was a padding position.
- Splits and contexts are configuration, not fixed code.
- Column names use `corrupt` (`clean_p_corruptmax`, `corrupt_token_ids`) in place of `foil`.
- `read_role_folder` works with or without `dataset.jsonl`, which covers paraphrased prompts.

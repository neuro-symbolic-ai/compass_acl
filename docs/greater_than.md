# Greater-Than Data

This module generates the greater-than task of Hanna et al. (2023), as released with EAP-IG. Each prompt names a span of years and the model should continue it with a year later than the start.

The module is imported as `data_generation.numbers`. The nouns come from the [lexicon](lexicon.md) (`GREATER_THAN_NOUNS`).

## Example

| Field | Value |
| --- | --- |
| clean | `The contract lasted from the year 1352 to the year 13` |
| label | `52` |
| corrupted | `The contract lasted from the year 1301 to the year 13` |

The correct continuations are the two-digit years above the label (`53`…`99`). The corrupted prompt sets the start year to `01`, so almost every continuation is valid and the prompt carries no greater-than signal.

## Configuration

```python
from data_generation.numbers import GreaterThanConfig

config = GreaterThanConfig(
    out_path="./data/numbers/greater_than.csv",
    n_examples=1000,
    seed=0,
    centuries=[11, 12, 13, 14, 15, 16, 17],   # XX in 1352
    year_min=2,                               # YY range, inclusive
    year_max=98,
    corrupted_year=1,                         # YY in the corrupted prompt
    template="The {noun} lasted from the year {century:02d}{year:02d} to the year {century:02d}",
)
```

`GreaterThanConfig.default()` matches `greater_than_data.csv` from the EAP-IG release.

## Generating prompts

```python
from data_generation.lexicon import Lexicon, LexiconConfig
from data_generation.numbers import GreaterThanGenerator, GreaterThanConfig

lexicon = Lexicon(LexiconConfig(tokenizer_name="gpt2", include_names=False))
generator = GreaterThanGenerator(lexicon, GreaterThanConfig.default())

df = generator.generate_all()      # writes the CSV and a .report.json next to it
```

Or from the command line, from the repository root:

```bash
python -m data_generation.scripts.generate_greater_than \
    --tokenizer gpt2 --out ./data/numbers/greater_than.csv --n 1000
```

| Argument | Purpose |
| --- | --- |
| `--tokenizer` | Tokenizer used to check that every two-digit year is one token. |
| `--out` | Output CSV path. |
| `--n` | Number of prompts (default: `1000`). |
| `--seed` | Random seed (default: `0`). |

The generator raises a `ValueError` if any of the years `00`…`99` is more than one token for the tokenizer. GPT-2 and Pythia tokenizers pass this check.

## Output

```text
clean, corrupted, label
```

There is no single target token, so the CSV has no `correct_idx`/`incorrect_idx` columns and the [filtering](filtering.md) module does not apply.

## Metric

The task is scored by the probability mass on years above the label minus the mass on years at or below it, read at the last token:

```python
from functools import partial
from data_generation.numbers import make_prob_diff

prob_diff = make_prob_diff(generator.year_token_ids())
metric = partial(prob_diff, loss=True, mean=True)      # for attribution
evaluation = partial(prob_diff, loss=False, mean=False)   # for faithfulness
```

`prob_diff` has the `(logits, clean_logits, input_length, labels, mean, loss)` signature that the `eap` attribution and evaluation functions expect. `labels` is the `label` column as a tensor.

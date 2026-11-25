# IOI Data

This module generates indirect-object-identification (IOI) minimal pairs, the task of Wang et al. (2023), in the CSV format used by the EAP-IG release. It serves as a reference task alongside the semantic-role data.

The module is imported as `data_generation.ioi`. Names, places and objects come from the [lexicon](lexicon.md), so every name is a single token for the chosen tokenizer.

## Example

| Field | Value |
| --- | --- |
| clean | `Then, Aaron and Amber went to the store. Aaron gave a drink to` |
| target (IO) | `Amber` |
| corrupt head (S) | `Aaron` |
| corrupted | `Then, Aaron and Amber went to the store. Tyler gave a drink to` |
| corrupted_hard | `Then, Aaron and Amber went to the store. Amber gave a drink to` |

The clean prompt ends before the indirect object. The model should complete it with the name that has not yet acted (IO) rather than the giver (S). The `corrupted` prompt replaces the giver with a third name (the ABC corruption); `corrupted_hard` replaces it with the IO, so the cue flips to S.

## Templates

| Family | Form | Count |
| --- | --- | --- |
| `BABA` | `[B] and [A] ... [B] gave a [OBJECT] to [A]` | 24 |
| `ABBA` | `[A] and [B] ... [B] gave a [OBJECT] to [A]` | 24 |
| `mixed` | BABA and ABBA together | 48 |
| `ABC` | `[A], [B] and [C] ... [B] and [C] gave a [OBJECT] to [A]` | 8 |
| `BAC` | ABC with A and B swapped in the first clause | 8 |
| `ABC mixed` | ABC and BAC together | 16 |

`[A]` is the IO, `[B]` the subject S, `[C]` a third name. ABBA and BAC are derived from BABA and ABC by swapping the two names in the first clause only.

## Configuration

```python
from data_generation.ioi import IOIConfig

config = IOIConfig(
    out_path="./data/ioi/ioi_pairs.csv",
    n_examples=1000,
    seed=123,
    prompt_type="BABA",       # BABA, ABBA, mixed, ABC, BAC or "ABC mixed"
    n_templates=None,         # use only the first n templates of the family
    symmetric=False,          # also emit the A/B-swapped twin of each prompt
    hard_corruption=True,     # corrupted_hard swaps the giver for the IO; False uses a third name
)
```

Presets: `IOIConfig.default()` (BABA) and `IOIConfig.abc()` (ABC mixed).

## Generating pairs

```python
from data_generation.lexicon import Lexicon, LexiconConfig
from data_generation.ioi import IOIGenerator, IOIConfig

lexicon = Lexicon(LexiconConfig.default(tokenizer_name="EleutherAI/pythia-1b"))
generator = IOIGenerator(lexicon, IOIConfig.default(out_path="./data/ioi/ioi_pairs.csv"))

df = generator.generate_all()      # writes the CSV and a .report.json next to it
rows = generator.generate()        # list of dicts, nothing written
```

Or from the command line, from the repository root:

```bash
python -m data_generation.scripts.generate_ioi \
    --tokenizer EleutherAI/pythia-1b --out ./data/ioi/ioi_pairs.csv --n 1000
```

| Argument | Purpose |
| --- | --- |
| `--tokenizer` | Hugging Face tokenizer; names that are not one token are dropped. |
| `--out` | Output CSV path. |
| `--n` | Number of pairs (default: `1000`). |
| `--seed` | Random seed (default: `123`). |
| `--prompt_type` | Template family (default: `BABA`). |
| `--symmetric` | Also emit the A/B-swapped twin of each prompt. |

## Output

```text
clean, corrupted, corrupted_hard, correct_idx, incorrect_idx, target_word, incorrect_word, template_idx
```

`correct_idx` and `incorrect_idx` are the token ids of ` IO` and ` S`. The first five columns are the ones in EAP-IG's `ioi_llama.csv`; the CSV can be passed to circuit discovery directly, or filtered first with the [filtering](filtering.md) module, which reads the same columns.

## Metric

IOI uses the logit difference between the IO and S tokens at the last position, which is the `cnp_metric` in `eap/cnp.py`.

## Differences from the original script

- Names are filtered by the lexicon up front, so there is no separate re-tokenising step for each model.
- The giver is replaced by name within the second clause, so every template is corrupted; the original located the giver by a fixed verb list and left templates with other verbs unchanged.
- `corrupted_hard` swaps the giver for the IO. The original swapped it for S, which is the giver already, so the hard prompt equalled the clean one.

# Lexicon

The lexicon holds every word inventory used to build semantic-role prompts and IOI prompts: the roles and their prepositional scaffolds, the cue-swap map, verb frames, subjects, objects, role heads, and first names. The `Lexicon` class filters all of them for one tokenizer so that every target is a single token.

The module is imported as `semantic_role_generation.lexicon`.

## Terms

- **Role:** one of ten semantic roles: Goal, Location, Source, Instrument, Topic, Material, Path, Beneficiary, Manner, Time.
- **Scaffold:** the phrase that ends a clean prompt and cues the role of the next token, for example `to the` for Goal.
- **Head:** the correct next token after the scaffold, for example `student`.
- **Parity family:** scaffolds with the same token length. A corrupted prompt swaps the scaffold for another role's scaffold from the same family, so clean and corrupted prompts have the same length.
- **Marker family:** `1token` (Time uses `at` / `on`, the paper setting) or `2token` (Time uses `at the`, `during the`, …, so it can swap with other roles).

## Configuration

```python
from semantic_role_generation.lexicon import Lexicon, LexiconConfig

config = LexiconConfig(
    tokenizer_name="EleutherAI/pythia-1b",
    single_token_only=True,       # drop words that are not one token with a leading space
    scaffold_markers="1token",    # "1token" or "2token"
    roles=None,                   # None keeps all ten roles
    include_names=True,           # also filter the IOI first names
)
lexicon = Lexicon(config)
```

Presets:

```python
LexiconConfig.default(tokenizer_name="EleutherAI/pythia-14m")   # paper setting, 1-token Time
LexiconConfig.updated_time()                                    # 2-token Time scaffolds
```

A tokenizer that is already loaded can be passed directly, which also avoids a second download:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
lexicon = Lexicon(LexiconConfig.default(), tokenizer=tokenizer)
```

`Lexicon` raises a `ValueError` if a role has no single-token heads left for the chosen tokenizer, or if `roles` names an unknown role.

## Role inventories

```python
lexicon.roles                        # ['Goal', 'Location', ..., 'Time']
lexicon.scaffolds("Location")        # ['in the', 'at the', 'near the', 'inside the']
lexicon.scaffold_token_length("Time")  # 1 in the 1-token family, 2 in the 2-token family
lexicon.cue_swap_roles("Goal")       # ['Location', 'Topic', 'Beneficiary']
lexicon.verbs("Topic")               # ['spoke', 'talked', 'wrote', 'reported']
lexicon.frame("Goal")                # 'giving_prep_dative'
lexicon.has_object("Goal")           # True: prompt has a direct object ("handed the book to the")
lexicon.all_scaffolds()              # every scaffold across the active roles, deduplicated
lexicon.paraphrases("Path")          # within-role scaffold paraphrases, for sanity checks
```

## Word lists

All lists below are already filtered to single tokens:

```python
lexicon.agents                       # subjects: teacher, worker, chef, ...
lexicon.themes                       # direct objects: book, file, box, ...
lexicon.heads["Instrument"]          # knife, saw, pen, brush, ...
lexicon.adverbs, lexicon.adj_agent, lexicon.adj_theme   # optional modifiers
lexicon.prefaces                     # clauses for the longer "hard" corruption

lexicon.names                        # first names for IOI prompts
lexicon.ioi_places                   # store, garden, restaurant, ...
lexicon.ioi_objects                  # ring, kiss, bone, ...
```

The unfiltered lists are importable for inspection:

```python
from semantic_role_generation.lexicon import ROLES, SCAFFOLDS, ROLE_CUE_SWAP, VERBS, FRAME
from semantic_role_generation.lexicon import AGENTS, THEMES, HEADS, NAMES, IOI_PLACES, IOI_OBJECTS
```

## Tokenization helpers

```python
lexicon.token_id("student")          # id of " student"; raises if it is not one token
lexicon.is_single_token("morning")   # True / False
lexicon.token_length("The chef handed the book to the")
lexicon.ids("to the")                # token ids without special tokens
```

## Tokenizer coverage

Different tokenizers keep different words. `summary()` reports how many survive, which is worth checking before generating data with a new model:

```python
print(lexicon.summary())
# {'agents': 17, 'themes': 21, 'names': 230, 'heads/Goal': 11, 'heads/Location': 11, ...}
```

## Marker families

In the `1token` family, Time is the only role with one-token scaffolds (`at`, `on`), so it cannot swap with another role while keeping token parity; its corruption is `at ↔ on` and `ROLE_CUE_SWAP["Time"]` is `["Time"]`. The `2token` family gives Time two-token scaffolds, heads that read naturally after them (`start`, `break`, `weekend`, …), and swaps to Location and Goal.

The 2-token overrides are per-role dictionaries (`SCAFFOLDS_2TOKEN`, `ROLE_CUE_SWAP_2TOKEN`, `VERBS_2TOKEN`, `HEADS_2TOKEN`). A role that is absent from them keeps its base entry, so an alternative inventory for another role is one new dictionary entry.

## Extending the lexicon

- **New words:** add them to the lists in `words.py`. They are filtered at load time, so multi-token words are dropped rather than breaking generation.
- **New role:** add it to `ROLES`, `SCAFFOLDS`, `ROLE_CUE_SWAP` and `FRAME` in `roles.py`, add its verbs to `VERBS`, and its heads to `HEADS` in `words.py`. If its prompt has a direct object, add it to `ROLES_WITH_OBJECT`.
- **New marker family:** add the family name to the check in `LexiconConfig.__post_init__` and a matching override dictionary.
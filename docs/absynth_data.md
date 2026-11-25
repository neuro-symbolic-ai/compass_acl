# ABSynth Data

COMPASS is designed for synthetic corpora generated with [ABSynth](https://github.com/nura-j/ABSynth_dataset), which provides controlled linguistic structure for studying how transformers learn.

Each ABSynth sentence includes:

- **Semantic frame annotations:** roles such as agent, patient, theme and location
- **Linguistic features:** POS tags, dependency parses and constituency trees
- **Statistical properties:** Zipfian word frequencies and entropy profiles
- **Complexity levels:** simple, medium and complex sentences

## Loading a corpus

```python
from trace.tokenizer import create_tokenizer_from_data

corpus_path = "path/to/absynth_corpus.json"
tokenizer = create_tokenizer_from_data(vocab_file=corpus_path)
```

The tokenizer builds its vocabulary from the corpus and keeps it consistent with the corpus annotations.

## Generating a corpus

```python
from absynth.corpus import SyntheticCorpusGenerator

generator = SyntheticCorpusGenerator()
corpus = generator.generate_corpus(num_sentences=1000)
corpus.save("my_absynth_corpus.json")
```

For more control over the generated data:

```python
from absynth.corpus import SyntheticCorpusGenerator

generator = SyntheticCorpusGenerator()
corpus = generator.generate_corpus(
    num_sentences=10000,
    complexity_distribution={
        "simple": 0.55,
        "medium": 0.35,
        "complex": 0.1,
    },
    semantic_frame_distribution={
        "transitive_action": 0.4,     # subject-verb-object
        "intransitive_action": 0.25,  # subject-verb
        "communication": 0.2,         # communication verbs
        "motion": 0.15,               # movement and location
    },
)
corpus.save("./data/custom_absynth_corpus.json", indent=2)
```

If `get_dataloader` is called without a `corpus_path`, it generates a corpus with ABSynth instead; `num_sentences` sets its size.

# Linguistic Probes

This module trains probes on hidden states and uses them to track which linguistic features a model encodes during training.

- **Probe types:** linear and multi-label probes
- **POS analysis:** part-of-speech information at basic or detailed granularity
- **Semantic-role analysis:** semantic-role information at basic or detailed granularity
- **Taggers:** rule-based taggers for ABSynth text, with an NLTK fallback
- **Performance tracking:** output accuracy per linguistic category

The workflow has two phases: train probes on a model, then load them to monitor the model during training.

## Phase 1: training probes

```python
from trace.linguistic_probes import LinguisticProbesConfig, ProbeTrainer

config = LinguisticProbesConfig(
    probe_type='multilabel',          # 'linear' or 'multilabel'
    hidden_dim=128,
    epochs=3,
    lr=0.001,
    batch_size=64,
    device="cuda",
    track_pos=True,
    track_semantic=True,
    pos_granularity='detailed',       # 'basic' or 'detailed'
    semantic_granularity='detailed',  # 'basic' or 'detailed'
    layer_indices=None,               # None probes all layers
    save_probes=True,
    save_visualizations=False,
    save_path='./trained_probes',
    log_dir="./analysis_results",
)
# Presets: LinguisticProbesConfig.default(), .minimal() and .comprehensive()

probe_trainer = ProbeTrainer(config, tokenizer)
training_results = probe_trainer.train_all_probes(model=decoder_model, dataloader=train_loader)

for layer_key, layer_results in training_results.items():
    print(f"Layer {layer_key}:")
    if 'pos' in layer_results:
        print(f"  POS accuracy: {layer_results['pos']['accuracy']:.3f}")
    if 'semantic' in layer_results:
        print(f"  Semantic accuracy: {layer_results['semantic']['accuracy']:.3f}")
```

## Granularity

```python
basic_config = LinguisticProbesConfig(
    pos_granularity='basic',          # NOUN, VERB, ADJ, ADV, PREP, DET, CONJ, OTHER
    semantic_granularity='basic',     # AGENT, PATIENT, ACTION, LOCATION, RELATION, CONNECTOR, RESULT, OTHER
)

detailed_config = LinguisticProbesConfig(
    pos_granularity='detailed',       # adds e.g. TRANSITIVE_VERB, COMMUNICATION_VERB
    semantic_granularity='detailed',  # adds e.g. MOTION, COMMUNICATION, DESTINATION
)

print("POS categories:", basic_config.get_pos_categories())
print("Semantic categories:", basic_config.get_semantic_categories())
```

## Phase 2: monitoring during training

```python
from torch.utils.data import DataLoader
from trace.linguistic_probes import LinguisticProbesConfig, POSAnalyzer, SemanticAnalyzer

probe_paths = {
    (0, 'decoder'): './trained_probes/pos_layer0_decoder.pt',
    (6, 'decoder'): './trained_probes/pos_layer6_decoder.pt',
    (11, 'decoder'): './trained_probes/pos_layer11_decoder.pt',
}

monitoring_config = LinguisticProbesConfig(
    track_pos=True,
    track_semantic=True,
    layer_indices={'decoder': [0, 6, 11]},
    log_dir="./analysis_results",
    save_visualizations=True,
    probe_load_path=probe_paths,
)

pos_analyzer = POSAnalyzer(monitoring_config)
semantic_analyzer = SemanticAnalyzer(monitoring_config)
pos_analyzer.load_probes(probe_paths)


def training_step_with_linguistic_monitoring(model, batch, step):
    # ... regular training code ...
    if step % 50 == 0:
        pos_confidences = pos_analyzer.analyze(
            model=model,
            dataloader=DataLoader([batch]),
            tokenizer=tokenizer,
            model_name=f"step_{step}",
        )
        # {(layer_idx, layer_type): {'NOUN': 0.85, 'VERB': 0.72, ...}}
        for layer_key, confidences in pos_confidences.items():
            print(f"Step {step}, layer {layer_key}:")
            for tag, confidence in confidences.items():
                print(f"  {tag}: {confidence:.3f}")
```

## Taggers

```python
from trace.linguistic_probes import POSTagger, SemanticTagger

pos_tagger = POSTagger(granularity='detailed', use_nltk_fallback=True)
semantic_tagger = SemanticTagger(granularity='detailed')

text = "noun_entity transitive_verb_action noun_target preposition_to location_place"
print("POS tags:", pos_tagger.tag_text(text))
print("Semantic tags:", semantic_tagger.tag_text(text))
# POS tags: [('noun_entity', 'NOUN'), ('transitive_verb_action', 'TRANSITIVE_VERB'), ...]
# Semantic tags: [('noun_entity', 'AGENT'), ('transitive_verb_action', 'ACTION'), ...]
```

## Performance tracking

```python
from trace.linguistic_probes import (
    LinguisticProbesConfig,
    POSPerformanceTracker,
    SemanticPerformanceTracker,
)

pos_tracker = POSPerformanceTracker(
    tokenizer=tokenizer,
    log_dir="./performance_tracking",
    config=LinguisticProbesConfig(pos_granularity='basic'),
)
semantic_tracker = SemanticPerformanceTracker(
    tokenizer=tokenizer,
    log_dir="./performance_tracking",
    config=LinguisticProbesConfig(semantic_granularity='basic'),
)


def training_step_with_performance_tracking(model, batch, step):
    logits = model(**batch).logits

    pos_metrics = pos_tracker.process_batch(batch, logits)
    semantic_metrics = semantic_tracker.process_batch(batch, logits)
    pos_tracker.update_epoch_metrics(pos_metrics, step)
    semantic_tracker.update_epoch_metrics(semantic_metrics, step)

    if step % 100 == 0:
        for category, accuracy in pos_metrics['pos_analysis_accuracy'].items():
            print(f"POS {category}: {accuracy:.3f}")
```

## Visualisation

```python
from trace.linguistic_probes import ProbesVisualizer

visualizer = ProbesVisualizer(log_dir="./analysis_results", config=monitoring_config)

confidence_history = {}
for step in range(0, 1000, 50):
    confidence_history[step] = pos_analyzer.analyze(model, eval_dataloader, tokenizer)

visualizer.plot_probe_confidence_analysis(
    confidence_data=confidence_history,
    model_name="my_transformer",
    analysis_type="pos",
    save_plot=True,
    show_plots=False,
)
```

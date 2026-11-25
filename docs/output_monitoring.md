# Output Monitoring

This module measures the quality of model outputs during training by tracking accuracy per grammatical and semantic category.

- **POS accuracy:** accuracy per part-of-speech category (nouns, verbs, adjectives, ...)
- **Semantic-role accuracy:** accuracy per semantic role (agents, patients, actions, ...)
- **Category insights:** which linguistic categories the model finds hardest
- **Evolution plots:** how output quality changes over training

## Basic usage

```python
from trace.output_monitoring import OutputMonitoringAnalyzer, OutputMonitoringConfig

config = OutputMonitoringConfig(
    track_pos_performance=True,
    track_semantic_roles=True,
    pos_granularity='basic',          # 'basic' or 'detailed'
    semantic_granularity='basic',     # 'basic' or 'detailed'
    save_visualizations=True,
    log_dir="./analysis_results",
    show_plots=False,
)
output_analyzer = OutputMonitoringAnalyzer(config)
# or use the defaults: OutputMonitoringAnalyzer(OutputMonitoringConfig.default())

all_monitoring_results = {}


def training_step_with_output_monitoring(model, batch, step):
    logits = model(**batch).logits
    loss = loss_function(logits, batch['labels'])
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        monitoring_results = output_analyzer.analyze(
            batch=batch,
            outputs=logits,
            tokenizer=tokenizer,
            step=step,
        )
        for category, accuracy in monitoring_results.get('pos_accuracy', {}).items():
            print(f"Step {step} - POS {category}: {accuracy:.3f}")
        for category, accuracy in monitoring_results.get('semantic_accuracy', {}).items():
            print(f"Step {step} - Semantic {category}: {accuracy:.3f}")
        all_monitoring_results[step] = monitoring_results
```

## Visualisation

```python
from trace.output_monitoring import OutputMonitoringVisualizer

output_visualizer = OutputMonitoringVisualizer(log_dir="./monitoring_visualizations", config=config)

output_visualizer.plot_pos_performance_evolution(
    monitoring_results=all_monitoring_results,
    model_name="transformer_v2",
    save_plot=True,
)
output_visualizer.plot_semantic_role_performance_evolution(
    monitoring_results=all_monitoring_results,
    model_name="transformer_v2",
    save_plot=True,
)

# Save metrics to CSV
output_visualizer.save_metrics(monitoring_results=all_monitoring_results, model_name="transformer_v2")
```

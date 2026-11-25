# Hessian Analysis

This module analyses the curvature of the loss to study training dynamics and memorisation.

- **Eigenvalue analysis:** extreme eigenvalues, trace estimates and spectral properties
- **Component analysis:** attention, feed-forward and embedding components separately
- **Gradient-Hessian alignment:** how the optimisation direction relates to curvature
- **Memorisation signals:** differences between the training and validation landscapes

## Configuration

```python
from trace.hessian import HessianConfig, HessianAnalyzer

config = HessianConfig(
    n_components=10,                            # number of eigenvalues to compute
    num_batches=100,                            # batches used for the estimate
    device="cuda",
    track_component_hessian=True,
    track_gradient_alignment=True,
    track_train_val_landscape_divergence=True,
    component_list=["attention", "ffn", "hidden_states"],
    log_dir="./analysis_results",
    save_hessian_data=True,
    show_plots=False,
)

analyzer = HessianAnalyzer(config)
```

Preset configurations are also available: `HessianConfig.default()`, `HessianConfig.minimal()` and `HessianConfig.comprehensive()`.

## Single-step analysis

```python
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

results = analyzer.analyze_step(
    model=decoder_model,
    loss_fn=loss_fn,
    train_batch=train_batch,
    val_batch=val_batch,              # optional, used for memorisation signals
    model_type="decoder_only",
    step=100,
)

print(f"Max eigenvalue: {results['hessian']['max_eigenvalue']:.2e}")
print(f"Min eigenvalue: {results['hessian']['min_eigenvalue']:.2e}")
print(f"Trace estimate: {results['hessian']['hessian_trace_estimate']:.2e}")
print(f"Negative eigenvalues: {results['hessian']['negative_count']}")
print(f"Effective rank: {results['hessian']['effective_rank_95']}")
```

## Component analysis

```python
from trace.hessian import ComponentAnalyzer

component_analyzer = ComponentAnalyzer()

component_results = component_analyzer.analyze_all_components(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    component_list=None,              # None analyses all components
    n_components=10,
)
```

## Gradient-Hessian alignment

```python
alignment_results = analyzer.compute_gradient_alignment(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    eigenvalues=eigenvalues,          # from a previous Hessian computation
    eigenvectors=eigenvectors,
)

print(f"Gradient-Hessian alignment: {alignment_results['grad_Hg_alignment']:.4f}")
print(f"Weighted alignment score: {alignment_results['weighted_alignment']:.4f}")
print(f"Curvature/gradient ratio: {alignment_results['grad_Hg_ratio']:.4f}")
```

## Memorisation signals

```python
memorization_signals = analyzer.compute_train_val_divergence(
    model=decoder_model,
    loss_fn=loss_fn,
    train_batch=train_batch,
    val_batch=val_batch,
    model_type="decoder_only",
)

print(f"Landscape divergence: {memorization_signals['train_val_landscape_divergence_score']:.4f}")
print(f"Trace ratio (train/val): {memorization_signals['trace_ratio']:.4f}")
print(f"Eigenvalue overlap: {memorization_signals['eigenvalue_distribution_overlap']:.4f}")
print(f"Effective rank difference: {memorization_signals['effective_rank_diff']}")
```

## Visualisation

```python
from trace.hessian import HessianVisualizer

visualizer = HessianVisualizer(config)

hessian_history = {}
for step in range(0, 1000, 100):
    hessian_history[step] = analyzer.analyze_step(
        model=decoder_model,
        loss_fn=loss_fn,
        train_batch=train_batch,
        val_batch=val_batch,
        step=step,
    )

visualizer.create_comprehensive_report(hessian_history=hessian_history, model_name="my_transformer")

# Individual plots
visualizer.plot_eigenvalue_evolution(hessian_history, "my_transformer")
visualizer.plot_gradient_alignment(hessian_history, "my_transformer")
visualizer.plot_component_comparison(hessian_history, "my_transformer")
visualizer.plot_memorization_metrics(hessian_history, model_name="my_transformer")
```

# Intrinsic Dimensions

This module estimates the intrinsic dimension of hidden representations, layer by layer, to show how a model compresses and organises information.

Supported estimators:

- **TwoNN:** two-nearest-neighbours estimator
- **MLE:** maximum likelihood estimator
- **PCA:** linear dimensionality from principal components

The module is imported as `trace.intrisic_dimensions`, which is the package's folder name.

## Basic usage

```python
from trace.intrisic_dimensions import IntrinsicDimensionsConfig, IntrinsicDimensionAnalyzer

config = IntrinsicDimensionsConfig(
    model_type="decoder_only",
    id_method="TwoNN",
    layers_to_analyze=None,        # None analyses all layers, e.g. {'decoder': [0, 3, 5]}
    save_visualizations=True,
    show_plots=False,
    log_dir="./analysis_results",
)

analyzer = IntrinsicDimensionAnalyzer(config)
# or use the defaults: IntrinsicDimensionAnalyzer(IntrinsicDimensionsConfig.default())

intrinsic_dimensions = analyzer.analyze(
    model=decoder_model,
    data_loader=data_loader,
    layers=[0, 3, 6, 9, 11],
    model_name="my_transformer",
)

# {(layer_idx, layer_type): intrinsic_dimension}
for (layer_idx, layer_type), id_value in intrinsic_dimensions.items():
    print(f"Layer {layer_idx} ({layer_type}): {id_value:.2f}")
```

## Extracting hidden states manually

```python
from trace.intrisic_dimensions import (
    IntrinsicDimensionsConfig,
    extract_hidden_representations,
    compute_intrinsic_dimensions,
)

hidden_states, _, _ = extract_hidden_representations(
    model=encoder_model,
    dataloader=data_loader,
    layer_indices={'encoder': [0, 2, 4]},
    device='cuda',
)

config = IntrinsicDimensionsConfig(id_method="TwoNN")
intrinsic_dims = compute_intrinsic_dimensions(hidden_states, config)
```

## Visualisation

```python
from trace.intrisic_dimensions import IntrinsicDimensionsVisualizer

visualizer = IntrinsicDimensionsVisualizer(log_dir="./analysis_results", config=config)

visualizer.generate_all_visualizations(
    intrinsic_dimensions=intrinsic_dims,
    model_name="transformer_analysis",
    show_plots=True,
)

# Individual plots
visualizer.plot_id_by_layer(intrinsic_dims, "my_model", save_plot=True)
visualizer.plot_id_distribution(intrinsic_dims, "my_model", save_plot=True)
visualizer.plot_final_id(intrinsic_dims, "my_model", save_plot=True)

# Save metrics to CSV
visualizer.save_metrics(intrinsic_dims, "my_model")
```

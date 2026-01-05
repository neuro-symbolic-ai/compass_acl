from typing import Optional, Union, List, Dict, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from trace.intrisic_dimensions.config import IntrinsicDimensionsConfig
from trace.transformer.utils import get_model_source


def extract_hidden_representations(
        model,
        dataloader: DataLoader,
        device: str = None,
        layer_indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        tokenizer=None,
        config: Optional[IntrinsicDimensionsConfig] = None,
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], None, None]:
    """
    Extract hidden representations from transformer layers.
    Automatically detects model type and uses appropriate method.
    """
    model.eval()
    if device is None:
        device = model.cfg.device if hasattr(model, 'cfg') and hasattr(model.cfg, 'device') else 'cpu'

    model_class_name = model.__class__.__name__
    is_hooked_transformer = model_class_name == 'HookedTransformer'


    print(f"Model class: {model_class_name}, Is hooked: {is_hooked_transformer}")
    #
    # if is_hooked_transformer:
    #     print("Using HookedTransformer extraction method")
    #     return _extract_hooked_transformer(model, dataloader, layer_indices, device)
    # else:
    #     print("Using original Transformer extraction method")
    #     return _extract_original_transformer(model, dataloader, layer_indices, device)

    model_source = get_model_source(model)
    if model_source == 'trace_hooked':
        print("Using HookedTransformer extraction method")
        return _extract_hooked_transformer(model, dataloader, layer_indices, device)
    elif model_source == 'trace_vanilla':
        print("Using original Transformer extraction method")
        return _extract_original_transformer(model, dataloader, layer_indices, device)
    elif model_source == 'transformerlens':
        print("Using TransformerLens extraction method")
        return _extract_transformerlens(model, dataloader, layer_indices, device)
    else:
        raise ValueError(f"Unsupported model type: {model_source}")

def _extract_hooked_transformer(model, dataloader, layer_indices, device):
    """Extract from HookedTransformer using ONLY run_with_cache - no manual hooks"""

    # Normalize layer_indices to dict format
    if layer_indices is None:
        ''' Set default layer indices based on model type and use the residual post-activation '''
        if model.model_type == 'decoder_only':
            layer_indices = {'decoder': list(range(model.cfg.n_layers))}
        elif model.model_type == 'encoder_only':
            layer_indices = {'encoder': list(range(model.cfg.n_layers))}
        else:  # encoder_decoder
            layer_indices = {
                'encoder': list(range(model.cfg.num_encoder_layers)),
                'decoder': list(range(model.cfg.num_decoder_layers))
            }
    elif isinstance(layer_indices, list):
        if model.model_type == 'decoder_only':
            layer_indices = {'decoder': layer_indices}
        elif model.model_type == 'encoder_only':
            layer_indices = {'encoder': layer_indices}
        else:
            layer_indices = {'decoder': layer_indices}

    print(f"HookedTransformer layer indices: {layer_indices}")

    # Collect hidden states
    all_hidden_states = {}
    for layer_type, indices in layer_indices.items():
        for idx in indices:
            all_hidden_states[(idx, layer_type)] = []

    # Process batches using run_with_cache ONLY
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting (HookedTransformer)"):
            input_ids = batch["input_ids"].to(device)

            try:
                # Use run_with_cache - this gets ALL activations automatically
                if model.model_type == 'decoder_only':
                    _, cache = model.run_with_cache(input_ids=input_ids)
                elif model.model_type == 'encoder_only':
                    _, cache = model.run_with_cache(input_ids=input_ids)
                else:  # encoder_decoder
                    _, cache = model.run_with_cache(src=input_ids, tgt=input_ids)

                # Debug: show available cache keys
                if len(all_hidden_states) == sum(len(v) for v in all_hidden_states.values()):  # First batch
                    print(f"Available cache keys (first 10): {sorted(list(cache.keys()))[:10]}")

                # Extract the layers we want
                for layer_type, indices in layer_indices.items():
                    for idx in indices:
                        # Get the right activation - try in order of preference
                        activation = None

                        if model.model_type in ['decoder_only', 'encoder_only']:
                            candidate_keys = [
                                f'blocks.{idx}.hook_resid_post',  # Final layer output
                                # f'blocks.{idx}.hook_resid_mid',  # After attention
                                # f'blocks.{idx}.hook_resid_pre',  # Layer input
                            ]
                        else:  # encoder_decoder
                            if layer_type == 'encoder':
                                candidate_keys = [f'encoder_blocks.{idx}.hook_resid_post']
                            else:
                                candidate_keys = [f'decoder_blocks.{idx}.hook_resid_post']

                        for key in candidate_keys:
                            if key in cache:
                                activation = cache[key]
                                break

                        if activation is not None:
                            all_hidden_states[(idx, layer_type)].append(activation.detach().cpu())
                        else:
                            print(f"Warning: No activation found for {layer_type} layer {idx}")

            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    # Concatenate collected states
    hidden_states = {}
    for key, states_list in all_hidden_states.items():
        if states_list:
            hidden_states[key] = torch.cat(states_list, dim=0)
            # print(f"Collected {hidden_states[key].shape} for {key}")
        else:
            print(f"No data for {key}")

    return hidden_states, None, None


def _extract_original_transformer(model, dataloader, layer_indices, device):
    """Extract from original Transformer using manual forward hooks"""

    # Set up layer indices
    if layer_indices is None:
        if model.model_type == 'encoder_only':
            layer_indices = {'encoder': list(range(len(model.encoder.layers)))}
        elif model.model_type == 'decoder_only':
            layer_indices = {'decoder': list(range(len(model.decoder.layers)))}
        else:  # encoder_decoder
            layer_indices = {
                'encoder': list(range(len(model.encoder.layers))),
                'decoder': list(range(len(model.decoder.layers)))
            }
    elif isinstance(layer_indices, list):
        if model.model_type == 'decoder_only':
            layer_indices = {'decoder': layer_indices}
        elif model.model_type == 'encoder_only':
            layer_indices = {'encoder': layer_indices}
        else:
            layer_indices = {'decoder': layer_indices}

    print(f"Original transformer layer indices: {layer_indices}")

    # Storage for hidden states
    hidden_dict = {}
    for layer_type, indices in layer_indices.items():
        for idx in indices:
            hidden_dict[(idx, layer_type)] = []

    # Hook function
    def save_hidden_state(layer_idx, layer_type):
        def hook(module, input, output):
            key = (layer_idx, layer_type)
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_dict[key].append(hidden.detach().cpu())

        return hook

    # Register hooks manually on the original model
    handles = []
    for layer_type, indices in layer_indices.items():
        for layer_idx in indices:
            try:
                if layer_type == 'encoder' and hasattr(model, 'encoder'):
                    layer_module = model.encoder.layers[layer_idx]
                    handle = layer_module.register_forward_hook(
                        save_hidden_state(layer_idx, 'encoder'))
                    handles.append(handle)
                    print(f"Registered hook on encoder layer {layer_idx}")

                elif layer_type == 'decoder' and hasattr(model, 'decoder'):
                    layer_module = model.decoder.layers[layer_idx]
                    handle = layer_module.register_forward_hook(
                        save_hidden_state(layer_idx, 'decoder'))
                    handles.append(handle)
                    print(f"Registered hook on decoder layer {layer_idx}")

            except (AttributeError, IndexError) as e:
                print(f"Could not register hook for {layer_type} layer {layer_idx}: {e}")

    # Forward pass to collect activations
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting (Original)"):
            input_ids = batch["input_ids"].to(device)

            try:
                if model.model_type == 'encoder_only':
                    _ = model(src=input_ids)
                elif model.model_type == 'decoder_only':
                    _ = model(tgt=input_ids)
                else:  # encoder_decoder
                    _ = model(src=input_ids, tgt=input_ids)
            except Exception as e:
                print(f"Forward pass error: {e}")
                continue

    # Clean up hooks
    for handle in handles:
        handle.remove()

    # Concatenate results
    hidden_states = {}
    for key, states_list in hidden_dict.items():
        if states_list:
            hidden_states[key] = torch.cat(states_list, dim=0)
            print(f"Collected {hidden_states[key].shape} for {key}")

    return hidden_states, None, None

def _extract_transformerlens(model, dataloader, layer_indices, device):
    """Extract from TransformerLens models using built-in methods"""
    # Placeholder for future implementation
    if layer_indices is None:
        layer_indices = list(range(model.cfg.n_layers))
    # Collect hidden states
    all_hidden_states = {}
    if 'gpt' in model.cfg.model_name.lower() or 'pythia' in model.cfg.model_name.lower() or 'gemma' in model.cfg.model_name.lower() or 'llama' in model.cfg.model_name.lower():
        layer_indices = {'decoder':layer_indices}
    else:
        layer_indices = {'encoder': layer_indices}
    for layer_type, indices in layer_indices.items():
        for idx in indices:
            all_hidden_states[(idx, layer_type)] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting (TransformerLens)"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            try:
                _, cache = model.run_with_cache(input_ids, attention_mask=attention_mask)
                for layer_type, indices in layer_indices.items():
                    for idx in indices:
                        key = f'blocks.{idx}.hook_resid_post'
                        if key in cache:
                            activation = cache[key]
                            all_hidden_states[(idx, layer_type)].append(activation.detach().cpu())
                        else:
                            print(f"Warning: No activation found for {layer_type} layer {idx}")
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    # Concatenate collected states
    hidden_states = {}
    for key, states_list in all_hidden_states.items():
        if states_list:
            hidden_states[key] = torch.cat(states_list, dim=0)
            # print(f"Collected {hidden_states[key].shape} for {key}")
        else:
            print(f"No data for {key}")

    return hidden_states, None, None



def compute_intrinsic_dimensions(
        hidden_states: Dict[Tuple[int, str], torch.Tensor],
        config: IntrinsicDimensionsConfig,
        device: str = None
) -> Dict[Tuple[int, str], float]:
    """
    Compute intrinsic dimensions for given hidden states.

    Args:
        hidden_states: Dictionary of hidden states from transformer layers
        config: Configuration object for intrinsic dimensions analysis
        device: Device for computation

    Returns:
        Dictionary with intrinsic dimension results
    """
    results = {}
    for layer_name, layer_hidden in hidden_states.items():
        layer_data = layer_hidden.detach().cpu().numpy()
        layer_data = layer_data.reshape(layer_data.shape[0], -1)
        results[layer_name] = config.id_estimator.fit_transform(layer_data)

    return results


def average_intrinsic_dimension(
        intrinsic_dimensions: Dict[Tuple[int, str], float],
        layer_indices: Optional[List[Tuple[int, str]]] = None
) -> float:
    """
    Compute the average intrinsic dimension across specified layers.

    Args:
        intrinsic_dimensions: Dictionary of intrinsic dimensions for each layer
        layer_indices: Optional list or dict of layer indices to average over

    Returns:
        Average intrinsic dimension
    """
    if layer_indices is None:
        layer_indices = list(intrinsic_dimensions.keys())

    total_id = 0.0
    count = 0

    for layer in layer_indices:
        if layer in intrinsic_dimensions:
            total_id += intrinsic_dimensions[layer]
            count += 1

    if count == 0:
        return 0.0

    return total_id / count
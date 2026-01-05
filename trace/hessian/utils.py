import torch
import numpy as np
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
from typing import List, Optional, Callable
from ..transformer import Transformer, get_model_source


def compute_loss(model, loss_fn, data_batch, model_type: Optional[str] = "decoder_only",
                 ignore_index: Optional[int] = -100) -> torch.Tensor:
    """
    Compute loss for a batch of data.
    Works with both original Transformer and HookedTransformer models.
    """
    # Detect model type
    # is_hooked = hasattr(model, 'run_with_cache') and hasattr(model, 'hook_dict') # Does not work
    # model_class_name = model.__class__.__name__
    # is_hooked = model_class_name == 'HookedTransformer'
    #
    # if is_hooked:
    #     # print("Using HookedTransformer loss computation")
    #     return _compute_loss_hooked_model(model, loss_fn, data_batch, model_type, ignore_index)
    # else:
    #     # print("Using original Transformer loss computation")
    #     return _compute_loss_original_model(model, loss_fn, data_batch, model_type, ignore_index)

    model_source = get_model_source(model)
    if model_source == 'trace_hooked':
        return _compute_loss_hooked_model(model, loss_fn, data_batch, model_type, ignore_index)
    elif model_source == 'trace_vanilla':
        return _compute_loss_original_model(model, loss_fn, data_batch, model_type, ignore_index)
    elif model_source == 'transformerlens':
        return compute_loss_transformerlens(model, loss_fn, data_batch, model_type, ignore_index)


def compute_loss_transformerlens(model, loss_fn, data_batch, model_type: Optional[str] = "decoder_only",
                                 ignore_index: Optional[int] = -100) -> torch.Tensor:
    """Compute loss for TransformerLens models"""
    # Use appropriate input format for TransformerLens models
    input_ids = data_batch["input_ids"]
    attention_mask = data_batch.get("attention_mask", None)
    outputs = model(input_ids, attention_mask=attention_mask)
    if 'labels' in data_batch:
        target_labels = data_batch["labels"].clone()
        # if model_type == "decoder_only":
        #     # Handle padding for decoder-only models
        #     pad_idx = getattr(model, 'pad_idx', 0)
        #     target_labels[target_labels == pad_idx] = ignore_index

        loss = loss_fn(
            outputs.reshape(-1, outputs.size(-1)),
            target_labels.reshape(-1)
        )
    else:
        loss = torch.tensor(0.0, device=input_ids.device)
    return loss



def _compute_loss_hooked_model(model, loss_fn, data_batch, model_type, ignore_index):
    """Compute loss for HookedTransformer"""
    # Use appropriate input format for hooked models
    input_ids = data_batch["input_ids"]

    if model_type == "decoder_only":
        outputs = model(input_ids=input_ids)
    elif model_type == "encoder_only":
        outputs = model(input_ids=input_ids)
    elif model_type == "encoder_decoder":
        # For encoder-decoder, might need both src and tgt
        outputs = model(src=input_ids, tgt=input_ids)
    else:
        outputs = model(input_ids=input_ids)

    # Calculate loss
    if "labels" in data_batch:
        target_labels = data_batch["labels"].clone()
        if model_type == "decoder_only":
            # Handle padding for decoder-only models
            pad_idx = getattr(model, 'pad_idx', 0)
            target_labels[target_labels == pad_idx] = ignore_index

        loss = loss_fn(
            outputs.reshape(-1, outputs.size(-1)),
            target_labels.reshape(-1)
        )
    else:
        loss = torch.tensor(0.0, device=input_ids.device)
    return loss


def _compute_loss_original_model(model, loss_fn, data_batch, model_type, ignore_index):
    """Compute loss for original Transformer (unchanged from original)"""
    # Handle different model types
    if model_type == "encoder_decoder":
        src_mask = data_batch["attention_mask"].unsqueeze(1)
        tgt_mask = model.generate_square_subsequent_mask(
            data_batch["decoder_input_ids"].size(1),
            data_batch["decoder_input_ids"].device
        ).unsqueeze(0).expand(data_batch["decoder_input_ids"].size(0), -1, -1)

        outputs = model(
            src=data_batch["input_ids"],
            tgt=data_batch["decoder_input_ids"],
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=(data_batch["attention_mask"] == 0),
            tgt_key_padding_mask=(data_batch["decoder_attention_mask"] == 0)
        )

    elif model_type == "decoder_only":
        tgt_mask = model.generate_square_subsequent_mask(
            data_batch["input_ids"].size(1),
            data_batch["input_ids"].device
        ).unsqueeze(0).expand(data_batch["input_ids"].size(0), -1, -1)

        outputs = model(
            tgt=data_batch["input_ids"],
            tgt_mask=tgt_mask,
        )
    else:
        # encoder-only or other model types
        outputs = model(src=data_batch["input_ids"])

    # Calculate loss
    if "labels" in data_batch:
        if model_type == "decoder_only":
            target_labels = data_batch["labels"].clone()
            pad_idx = getattr(model, 'pad_idx', 0)
            target_labels[target_labels == pad_idx] = ignore_index
        else:
            target_labels = data_batch["labels"]

        loss = loss_fn(
            outputs.reshape(-1, outputs.size(-1)),
            target_labels.reshape(-1)
        )
    else:
        loss = torch.tensor(0.0, device=data_batch["input_ids"].device)

    return loss


def extract_component_parameters(model, component_name: str,
                                 include_bias: Optional[bool] = False) -> List[torch.Tensor]:
    """
    Extract parameters for specific model components.
    Works with both original Transformer and HookedTransformer models.
    Excludes hook-related parameters.
    """
    # Detect model type
    # is_hooked = hasattr(model, 'run_with_cache') and hasattr(model, 'hook_dict')
    # model_class_name = model.__class__.__name__
    # is_hooked = model_class_name == 'HookedTransformer'
    model_source = get_model_source(model)
    # if is_hooked:
    #     params = _extract_hooked_model_components(model, component_name, include_bias)
    # else:
    #     params = _extract_original_model_components(model, component_name, include_bias)
    if model_source == 'trace_vanilla':
        # Original model might have tied weights (e.g., embedding and output projection)
        params = _extract_original_model_components(model, component_name, include_bias)
    elif model_source == 'trace_hooked':
        params = _extract_hooked_model_components(model, component_name, include_bias)
    elif model_source == 'transformerlens':
        # For TransformerLens, use original component extraction
        params = _extract_transformerlens_model_components(model, component_name, include_bias)


        # Test which parameters actually contribute to the loss
    # if params:
    #     model.zero_grad()
    #     # Create a dummy batch for testing
    #     dummy_input = torch.randint(0, 100, (2, 10)).to(next(model.parameters()).device)
    #
    #     try:
    #         # Test forward pass
    #         if is_hooked:
    #             if model.model_type == 'decoder_only':
    #                 outputs = model(input_ids=dummy_input)
    #             elif model.model_type == 'encoder_only':
    #                 outputs = model(input_ids=dummy_input)
    #             else:
    #                 outputs = model(src=dummy_input, tgt=dummy_input)
    #         else:
    #             if model.model_type == 'decoder_only':
    #                 outputs = model(tgt=dummy_input)
    #             elif model.model_type == 'encoder_only':
    #                 outputs = model(src=dummy_input)
    #             else:
    #                 outputs = model(src=dummy_input, tgt=dummy_input)
    #
    #         # Test gradients
    #         dummy_loss = outputs.sum()
    #         test_grads = grad(dummy_loss, params, allow_unused=True)
    #
    #         # Filter to only parameters that get gradients
    #         valid_params = []
    #         for param, grad_val in zip(params, test_grads):
    #             if grad_val is not None:
    #                 valid_params.append(param)
    #
    #         if valid_params:
    #             print(f"Component '{component_name}': {len(valid_params)}/{len(params)} parameters are used")
    #             return valid_params
    #         else:
    #             print(f"Warning: No parameters in component '{component_name}' contribute to loss")
    #             return params  # Return all anyway, let the caller handle it
    #
    #     except Exception as e:
    #         print(f"Warning: Could not test parameter usage for component '{component_name}': {e}")
    #         return params  # Return all parameters if testing fails

    return params


def get_available_transformerlens_components(model) -> List[str]:
    """
    Dynamically detect which components are actually available in a TransformerLens model.
    This prevents errors from trying to extract components that don't have trainable parameters.
    """
    available_components = []
    param_names = [name for name, _ in model.named_parameters()]

    # Check each component type
    component_checks = {
        "attention": lambda: any("attn.W_O" in name for name in param_names),
        "attention_query": lambda: any("attn.W_Q" in name for name in param_names),
        "attention_key": lambda: any("attn.W_K" in name for name in param_names),
        "attention_value": lambda: any("attn.W_V" in name for name in param_names),
        "ffn": lambda: any("mlp.W_out" in name for name in param_names),
        "embeddings": lambda: any("embed.W_E" in name for name in param_names),
        "output_projection": lambda: any("unembed.W_U" in name for name in param_names),
        "norm": lambda: any(any(norm in name for norm in ["ln1", "ln2", "ln_final"]) and
                            any(param in name for param in ["weight", "bias", "scale"])
                            for name in param_names),
        "hidden_states": lambda: any("ln_final" in name and
                                     any(param in name for param in ["weight", "bias", "scale"])
                                     for name in param_names),
        # Composite components
        "attention_all": lambda: any("attn." in name and any(x in name for x in ["W_Q", "W_K", "W_V", "W_O"])
                                     for name in param_names),
        "ffn_all": lambda: any("mlp." in name and any(x in name for x in ["W_in", "W_out"])
                               for name in param_names),
        "input_components": lambda: any(("attn." in name and any(x in name for x in ["W_Q", "W_K", "W_V"])) or
                                        ("mlp.W_in" in name) for name in param_names),
        "output_components": lambda: any(("attn.W_O" in name) or ("mlp.W_out" in name) for name in param_names),
    }

    for component, check_fn in component_checks.items():
        if check_fn():
            available_components.append(component)

    return available_components


def _extract_transformerlens_model_components(model, component_name: str, include_bias: bool) -> List[torch.Tensor]:
    """Extract parameters from TransformerLens HookedTransformer models with robust error handling"""

    # First, check if this component actually exists
    available_components = get_available_transformerlens_components(model)

    if component_name not in available_components and component_name != "all":
        print(f"Component '{component_name}' not available in this TransformerLens model.")
        print(f"Available components: {available_components}")

        # Suggest alternatives
        if component_name == "hidden_states":
            if "output_projection" in available_components:
                print("Suggestion: Use 'output_projection' as a proxy for hidden states analysis.")
            elif "attention" in available_components:
                print("Suggestion: Use 'attention' or 'ffn' components instead.")
        elif component_name == "norm":
            print("Note: This model appears to use non-trainable layer normalization.")

        raise ValueError(f"Component '{component_name}' not found. Available: {available_components}")

    # Component filters for TransformerLens - following their naming convention
    component_filters = {
        # Attention: Only the output projection W_O (consistent with hooked approach)
        "attention": lambda name: (
                "attn.W_O" in name and "blocks." in name
        ),

        # Attention sub-components
        "attention_query": lambda name: "attn.W_Q" in name and "blocks." in name,
        "attention_key": lambda name: "attn.W_K" in name and "blocks." in name,
        "attention_value": lambda name: "attn.W_V" in name and "blocks." in name,

        # FFN/MLP: Only the output projection W_out (consistent with hooked approach)
        "ffn": lambda name: (
                "mlp.W_out" in name and "blocks." in name
        ),

        # Embeddings: Input embedding matrix
        "embeddings": lambda name: (
                "embed.W_E" in name
        ),

        # Normalization layers: Only if they actually have trainable parameters
        "norm": lambda name: (
                any(x in name for x in ["ln1", "ln2", "ln_final"]) and
                any(y in name for y in ["weight", "bias", "scale"])
        ),

        # Hidden states: Use layer norm if available, otherwise output projection
        "hidden_states": lambda name: (
                ("ln_final" in name and any(x in name for x in ["weight", "bias", "scale"])) or
                ("unembed.W_U" in name if not any("ln_final" in n and any(x in n for x in ["weight", "bias", "scale"])
                                                  for n, _ in model.named_parameters()) else False)
        ),

        # Output projection: Final projection to vocabulary (unembed)
        "output_projection": lambda name: (
                "unembed.W_U" in name
        ),

        # All parameters (for debugging/comparison)
        "all": lambda name: True,

        # Additional useful components for TransformerLens

        # All attention components (Q, K, V, O)
        "attention_all": lambda name: (
                "attn." in name and "blocks." in name and
                any(x in name for x in ["W_Q", "W_K", "W_V", "W_O"])
        ),

        # All MLP components (in and out)
        "ffn_all": lambda name: (
                "mlp." in name and "blocks." in name and
                any(x in name for x in ["W_in", "W_out"])
        ),

        # Input components (Q, K, V, MLP input) - parameters that process inputs
        "input_components": lambda name: (
                (("attn." in name and any(x in name for x in ["W_Q", "W_K", "W_V"])) or
                 ("mlp.W_in" in name)) and "blocks." in name
        ),

        # Output components (attention output, MLP output) - parameters that produce outputs
        "output_components": lambda name: (
                (("attn.W_O" in name) or ("mlp.W_out" in name)) and "blocks." in name
        ),
    }

    if component_name not in component_filters:
        raise ValueError(f"Unknown component: {component_name}. Available: {list(component_filters.keys())}")

    print(f'Extracting parameters for component: {component_name} (TransformerLens)')

    # Filter parameters
    if include_bias:
        params = [p for name, p in model.named_parameters()
                  if component_filters[component_name](name) and p.requires_grad]
    else:
        params = [p for name, p in model.named_parameters()
                  if component_filters[component_name](name) and p.requires_grad and not name.endswith(
                ('b_Q', 'b_K', 'b_V', 'b_O', 'b_in', 'b_out', 'b_U'))]

    if not params:
        print(f"No parameters found for component '{component_name}' after filtering.")
        raise ValueError(f"No parameters found for component: {component_name}")

    # Show what was selected
    param_names = []
    for param in params:
        for name, p in model.named_parameters():
            if p is param:
                param_names.append(name)
                break

    print(f"Found {len(params)} parameters for component '{component_name}':")
    for name in param_names:
        print(f"  {name}")

    return params
# def _extract_transformerlens_model_components(model, component_name: str, include_bias: bool) -> List[torch.Tensor]:
#     """Extract parameters from TransformerLens HookedTransformer models"""
#
#     # Component filters for TransformerLens - following their naming convention
#     component_filters = {
#         # Attention: Only the output projection W_O (consistent with hooked approach)
#         "attention": lambda name: (
#                 "attn.W_O" in name and "blocks." in name
#         ),
#
#         # Attention sub-components
#         "attention_query": lambda name: "attn.W_Q" in name and "blocks." in name,
#         "attention_key": lambda name: "attn.W_K" in name and "blocks." in name,
#         "attention_value": lambda name: "attn.W_V" in name and "blocks." in name,
#
#         # FFN/MLP: Only the output projection W_out (consistent with hooked approach)
#         "ffn": lambda name: (
#                 "mlp.W_out" in name and "blocks." in name
#         ),
#
#         # Embeddings: Input embedding matrix
#         "embeddings": lambda name: (
#                 "embed.W_E" in name
#         ),
#
#         # Normalization layers: TransformerLens uses ln1, ln2, ln_final typically
#         "norm": lambda name: (
#                 any(x in name for x in ["ln1", "ln2", "ln_final"]) and "blocks." in name
#         ),
#
#         # Hidden states: Final layer norm (affects final hidden states)
#         # Note: Some TransformerLens models may not have trainable layer norm parameters
#         "hidden_states": lambda name: (
#                 "ln_final" in name or "final_ln" in name or
#                 ("ln" in name and "final" in name)
#         ),
#
#         # Output projection: Final projection to vocabulary (unembed)
#         "output_projection": lambda name: (
#                 "unembed.W_U" in name
#         ),
#
#         # All parameters (for debugging/comparison)
#         "all": lambda name: True,
#
#         # Additional useful components for TransformerLens
#
#         # All attention components (Q, K, V, O)
#         "attention_all": lambda name: (
#                 "attn." in name and "blocks." in name and
#                 any(x in name for x in ["W_Q", "W_K", "W_V", "W_O"])
#         ),
#
#         # All MLP components (in and out)
#         "ffn_all": lambda name: (
#                 "mlp." in name and "blocks." in name and
#                 any(x in name for x in ["W_in", "W_out"])
#         ),
#
#         # Input components (Q, K, V, MLP input) - parameters that process inputs
#         "input_components": lambda name: (
#                 (("attn." in name and any(x in name for x in ["W_Q", "W_K", "W_V"])) or
#                  ("mlp.W_in" in name)) and "blocks." in name
#         ),
#
#         # Output components (attention output, MLP output) - parameters that produce outputs
#         "output_components": lambda name: (
#                 (("attn.W_O" in name) or ("mlp.W_out" in name)) and "blocks." in name
#         ),
#     }
#
#     if component_name not in component_filters:
#         raise ValueError(f"Unknown component: {component_name}. Available: {list(component_filters.keys())}")
#
#     print(f'Extracting parameters for component: {component_name} (TransformerLens)')
#
#     # Filter parameters
#     if include_bias:
#         params = [p for name, p in model.named_parameters()
#                   if component_filters[component_name](name) and p.requires_grad]
#     else:
#         params = [p for name, p in model.named_parameters()
#                   if component_filters[component_name](name) and p.requires_grad and not name.endswith(
#                 ('b_Q', 'b_K', 'b_V', 'b_O', 'b_in', 'b_out', 'b_U'))]
#
#     if not params:
#         print(f"No parameters found for component '{component_name}'. Available parameters:")
#         for name, p in model.named_parameters():
#             if p.requires_grad:
#                 print(f"  {name}: {p.shape}")
#
#         # Special handling for common missing components in TransformerLens
#         if component_name == "hidden_states":
#             print(f"Note: TransformerLens model appears to use non-trainable layer norms.")
#             print(f"Consider using 'output_projection' or other available components instead.")
#         elif component_name == "norm":
#             print(f"Note: No trainable normalization parameters found in this TransformerLens model.")
#
#         raise ValueError(f"No parameters found for component: {component_name}")
#
#     # Show what was selected
#     param_names = []
#     for param in params:
#         for name, p in model.named_parameters():
#             if p is param:
#                 param_names.append(name)
#                 break
#
#     print(f"Found {len(params)} parameters for component '{component_name}':")
#     for name in param_names:
#         print(f"  {name}")
#
#     return params

def _extract_hooked_model_components(model, component_name: str, include_bias: bool) -> List[torch.Tensor]:
    """Extract OUTPUT parameters from HookedTransformer components, excluding hooks"""

    # Define filters that explicitly exclude hook parameters
    def is_hook_param(name):
        """Check if parameter belongs to hook infrastructure"""
        hook_indicators = [
            'hook_embed', 'hook_pos', 'hook_logits', 'hook_normalized',
            'hook_resid', 'hook_attn_out', 'hook_mlp_out', 'hook_cross_attn',
            'hook_q', 'hook_k', 'hook_v', 'hook_pattern', 'hook_z', 'hook_result',
            'hook_pre', 'hook_mid', 'hook_post', 'hook_attn_scores',
            'hook_q_input', 'hook_k_input', 'hook_v_input', 'hook_mlp_in',
            'hook'
        ]
        return any(hook in name for hook in hook_indicators)

    # Component filters for HookedTransformer - ONLY OUTPUT PARAMETERS
    component_filters = {
        # Attention: Only the output projection W_O (not W_Q, W_K, W_V)
        "attention": lambda name: (
                "W_O" in name and "attn." in name and not is_hook_param(name)
        ),

        # Attention sub-components
        "attention_query": lambda name: "W_Q" in name and not is_hook_param(name),
        "attention_key": lambda name: "W_K" in name and not is_hook_param(name),
        "attention_value": lambda name: "W_V" in name and not is_hook_param(name),

        # FFN: Only the output projection W_out (not W_in)
        "ffn": lambda name: (
                "W_out" in name and "mlp." in name and not is_hook_param(name)
        ),

        # Embeddings: Input embedding (this IS the output of embedding lookup)
        "embeddings": lambda name: (
                "embed.weight" in name and not is_hook_param(name)
        ),

        # Normalization layers: These directly affect the output
        "norm": lambda name: (
                any(x in name for x in ["ln1", "ln2", "ln_final"])
                and not is_hook_param(name)
        ),

        # Hidden states: Final layer norm (affects final hidden states)
        "hidden_states": lambda name: (
                "ln_final" in name and not is_hook_param(name)
        ),

        # Output projection: Final projection to vocabulary
        "output_projection": lambda name: (
                "unembed" in name and not is_hook_param(name)
        ),

        # All non-hook parameters (for debugging/comparison)
        "all": lambda name: not is_hook_param(name)
    }

    if component_name not in component_filters:
        raise ValueError(f"Unknown component: {component_name}. Available: {list(component_filters.keys())}")

    # print(f'Extracting OUTPUT parameters for component: {component_name}')

    # Filter parameters
    if include_bias:
        params = [p for name, p in model.named_parameters()
                  if component_filters[component_name](name) and p.requires_grad]
    else:
        params = [p for name, p in model.named_parameters()
                  if component_filters[component_name](name) and p.requires_grad and 'bias' not in name]

    if not params:
        # print(f"No parameters found for component '{component_name}'. Available parameters:")
        for name, p in model.named_parameters():
            if p.requires_grad and not is_hook_param(name):
                print(f"  {name}: {p.shape}")
        raise ValueError(f"No OUTPUT parameters found for component: {component_name}")

    # Show what was selected
    param_names = []
    for param in params:
        for name, p in model.named_parameters():
            if p is param:
                param_names.append(name)
                break

    # print(f"Found {len(params)} OUTPUT parameters for component '{component_name}':")
    # for name in param_names:
    #     print(f" {name}")

    return params


def _extract_original_model_components(model, component_name: str, include_bias: bool) -> List[torch.Tensor]:
    """Extract components from original Transformer model"""

    # Original component filters (unchanged)
    component_filters = {
        "attention": lambda name: "attn" in name,
        "attention_query": lambda name: "attention" in name and "q_linear" in name,
        "attention_key": lambda name: "attention" in name and "k_linear" in name,
        "attention_value": lambda name: "attention" in name and "v_linear" in name,
        "ffn": lambda name: any(x in name for x in ["ffn", "mlp", "feed_forward"]),
        "embeddings": lambda name: "embed" in name,
        "norm": lambda name: "norm" in name or "ln" in name,
        "hidden_states": lambda name: "norm1" in name if getattr(model, 'no_fnn', False) else "norm2" in name,
        "output_projection": lambda name: "output" in name or "classifier" in name or "lm_head" in name,
        "all": lambda name: True
    }

    if component_name not in component_filters:
        raise ValueError(f"Unknown component: {component_name}. Available: {list(component_filters.keys())}")

    # Filter parameters
    if include_bias:
        params = [p for name, p in model.named_parameters()
                  if component_filters[component_name](name) and p.requires_grad]
    else:
        params = [p for name, p in model.named_parameters()
                  if component_filters[component_name](name) and p.requires_grad and 'bias' not in name]

    if not params:
        # print(f"Available parameters for debugging:")
        # for name, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(f"  {name}: {p.shape}")
        raise ValueError(f"No parameters found for component: {component_name}")

    # print(f"Found {len(params)} parameters for component '{component_name}' (original model)")
    return params



def get_hessian_eigenvectors(
        model,
        loss_fn,
        train_data_loader,
        num_batches: int,
        device,
        n_top_vectors: int,
        param_extract_fn: Optional[Callable] = None
):
    """
    Compute Hessian eigenvalues and eigenvectors, handling unused parameters.
    """
    param_extract_fn = param_extract_fn or (lambda x: x.parameters())

    # Get all candidate parameters
    candidate_params = [p for p in param_extract_fn(model) if p.requires_grad]

    # Test which parameters actually get gradients
    model.zero_grad()
    loss = compute_loss(model, loss_fn, train_data_loader)
    test_grads = grad(loss, candidate_params, create_graph=True, allow_unused=True)

    # Filter to only parameters that actually get gradients
    valid_params = []
    for param, grad_val in zip(candidate_params, test_grads):
        if grad_val is not None:
            valid_params.append(param)

    if not valid_params:
        print("Warning: No parameters with valid gradients found")
        return np.array([0.0]), np.array([[0.0]])

    num_params = sum(p.numel() for p in valid_params)
    # print(f"Using {len(valid_params)} parameters with {num_params} total values for Hessian")

    def hessian_vector_product(vector):
        """Compute Hessian-vector product using only valid parameters."""
        model.zero_grad()
        loss = compute_loss(model, loss_fn, train_data_loader)

        # Compute gradients only for valid parameters
        grad_params = grad(loss, valid_params, create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grad_params])

        # Ensure vector size matches
        if vector.size(0) != flat_grad.size(0):
            vector = vector[:flat_grad.size(0)]

        grad_vector_product = torch.dot(flat_grad, vector)

        # Compute second derivatives
        hvp = grad(grad_vector_product, valid_params, retain_graph=True)
        return torch.cat([g.contiguous().view(-1) for g in hvp])

    def matvec(v):
        """Matrix-vector product for scipy's eigsh."""
        v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
        try:
            result = hessian_vector_product(v_tensor)
            return result.cpu().detach().numpy()
        except Exception as e:
            print(f"Warning: Error in Hessian-vector product: {e}")
            return np.zeros_like(v)

    try:
        linear_operator = LinearOperator((num_params, num_params), matvec=matvec)
        eigenvalues, eigenvectors = eigsh(
            linear_operator,
            k=min(n_top_vectors, num_params - 1),
            tol=0.001,
            which='LM',
            return_eigenvectors=True
        )
        eigenvectors = np.transpose(eigenvectors)
        return eigenvalues[::-1], eigenvectors[::-1]
    except Exception as e:
        print(f"Error computing eigenvalues: {e}")
        return np.array([1.0, 0.5]), np.eye(2)
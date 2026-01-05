import einops
import torch
import torch.nn as nn
from typing import List, Optional

from jaxtyping import Float


def attach_hooks(model: nn.Module, names: Optional[List[str]] = None) -> nn.Module:
    """
    Attach gradient monitoring hooks to model parameters.

    Args:
        model: Transformer model to attach hooks to
        names: Optional list of parameter names to hook. If None, hooks all non-embedding parameters

    Returns:
        Model with attached hooks
    """

    def create_hook(param_name: str):
        """Create a hook function with parameter name captured in closure."""
        return lambda grad: print(f"Gradient for {param_name}: {grad.norm()}")

    if names is None:
        names = [name for name, _ in model.named_parameters() if 'embedding' not in name]

    for name, param in model.named_parameters():
        if name in names:
            hook = create_hook(name)
            param.register_hook(hook)

    return model


def remove_hooks(model: nn.Module, names: Optional[List[str]] = None) -> nn.Module:
    """
    Remove gradient hooks from model parameters.

    Args:
        model: Transformer model to remove hooks from
        names: Optional list of parameter names (currently unused)

    Returns:
        Model with hooks removed
    """
    for param in model.parameters():
        param.register_hook(None)
    return model


def expand_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Expand attention mask to support different input dimensions.

    Supports broadcasting mask shapes for multi-head attention:
    - 2D masks are broadcasted over batch size and number of heads
    - 3D masks are broadcasted over number of heads
    - 4D masks are left as-is

    Args:
        mask: Input mask tensor of shape [seq_len, seq_len], [batch, seq_len, seq_len],
              or [batch, heads, seq_len, seq_len]

    Returns:
        Expanded mask tensor of shape [batch, heads, seq_len, seq_len]
    """
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)

    return mask


def get_device() -> torch.device:
    """Get the available device (GPU if available, else CPU).
    The method should check for multiple GPUs and return the first one if available.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        free_mem = []
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            free, _ = torch.cuda.mem_get_info(i)
            free_mem.append(free)
        best_gpu = free_mem.index(max(free_mem))
        return torch.device(f"cuda:{best_gpu}")
    return torch.device("cpu")

def repeat_along_head_dimension_tl(
    tensor: Float[torch.Tensor, "batch pos d_model"],
    n_heads: int,
    clone_tensor=True,
    # `einops.repeat` uses a view in torch, so we generally clone the tensor to avoid using shared storage for each head entry
):
    repeated_tensor = einops.repeat(
        tensor,
        "batch pos d_model -> batch pos n_heads d_model",
        n_heads=n_heads,
    )
    if clone_tensor:
        return repeated_tensor.clone()
    else:
        return repeated_tensor


def repeat_along_head_dimension(tensor, n_heads):
    """
    Repeats a tensor along a new head dimension.
    [batch, pos, d_model] -> [batch, pos, n_heads, d_model]
    inspired by the transformer-lens implementation
    """
    batch, pos, d_model = tensor.shape
    return tensor.unsqueeze(2).expand(batch, pos, n_heads, d_model)


def get_model_source(model):
    """Detect the source of the transformer model"""
    module_path = model.__class__.__module__
    class_name = model.__class__.__name__

    if 'transformer_lens' in module_path.lower():
        return 'transformerlens'
    elif class_name == 'HookedTransformer' and 'trace' in module_path:
        return 'trace_hooked'
    elif hasattr(model, 'model_type'):  # Your vanilla transformer
        return 'trace_vanilla'
    else:
        return 'pretrained_transformers'
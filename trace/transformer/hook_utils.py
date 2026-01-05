"""
Updated utility functions with TransformerLens/EAP-IG compatible hook naming.
Key changes:
- Hook names use blocks.X.attn.hook_Y instead of decoder.layers.X.self_attn.hook_Y
- Added get_act_name() utility function matching TransformerLens
- Updated all hook filtering and management for new naming conventions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from functools import partial
import re

from .hooked_models import HookedTransformer
from .hook_points import ActivationCache


def get_act_name(name: str, layer: Optional[int] = None, layer_type: Optional[str] = None) -> str:
    """
    Generate TransformerLens/EAP-IG compatible activation names.

    Args:
        name: Base activation name ('k', 'q', 'v', 'pattern', 'z', 'result', 'pre', 'mid', 'post', etc.)
        layer: Layer number (0-indexed)
        layer_type: Layer type ('attn', 'mlp') - usually inferred from name

    Returns:
        Full hook name compatible with TransformerLens/EAP-IG

    Examples:
        get_act_name('k', 6) -> 'blocks.6.attn.hook_k'
        get_act_name('pre', 2) -> 'blocks.2.mlp.hook_pre'
        get_act_name('embed') -> 'hook_embed'
        get_act_name('pattern', 0) -> 'blocks.0.attn.hook_pattern'
        get_act_name('resid_post', 3) -> 'blocks.3.hook_resid_post'
    """
    # Handle special global activations
    if name in ['embed', 'pos', 'logits', 'normalized']:
        return f'hook_{name}'

    # Handle layer-specific activations
    if layer is not None:
        # Attention-related activations
        if name in ['q', 'k', 'v', 'pattern', 'z', 'result', 'attn_scores']:
            return f'blocks.{layer}.attn.hook_{name}'

        # MLP-related activations
        elif name in ['pre', 'mid', 'post']:
            return f'blocks.{layer}.mlp.hook_{name}'

        # Residual stream activations
        elif name.startswith('resid_'):
            return f'blocks.{layer}.hook_{name}'

        # Other block-level activations
        elif name in ['attn_out', 'mlp_out']:
            return f'blocks.{layer}.hook_{name}'

        # Cross-attention (for encoder-decoder models)
        elif name in ['cross_attn_out']:
            return f'blocks.{layer}.hook_{name}'

    # Parse shorthand notation (e.g., 'k6', 'pre5', 'pattern3')
    match = re.match(r'([a-zA-Z_]+)(\d+)', name)
    if match:
        base_name = match.group(1)
        layer_num = int(match.group(2))
        return get_act_name(base_name, layer_num)

    # Return as-is if no pattern matches (user might provide full name)
    return name


class HookUtils:
    """Updated utility class with TransformerLens/EAP-IG compatible naming."""

    @staticmethod
    def get_hook_names_by_type(model: HookedTransformer, hook_type: str) -> List[str]:
        """
        Get hook names filtered by type using new naming conventions.

        Args:
            model: Hooked transformer model
            hook_type: Type of hooks ('attention', 'mlp', 'residual', 'embedding', 'all')

        Returns:
            List of matching hook names with blocks.X.Y.hook_Z format
        """
        all_hooks = model.get_hook_names()

        if hook_type == 'all':
            return all_hooks
        elif hook_type == 'attention':
            return [name for name in all_hooks if '.attn.hook_' in name]
        elif hook_type == 'mlp':
            return [name for name in all_hooks if '.mlp.hook_' in name]
        elif hook_type == 'residual':
            return [name for name in all_hooks if 'hook_resid_' in name]
        elif hook_type == 'embedding':
            return [name for name in all_hooks if name in ['hook_embed', 'hook_pos']]
        elif hook_type == 'output':
            return [name for name in all_hooks if name in ['hook_logits', 'hook_normalized']]
        else:
            raise ValueError(f"Unknown hook type: {hook_type}. Options: attention, mlp, residual, embedding, output, all")

    @staticmethod
    def get_layer_hooks(model: HookedTransformer, layer_idx: int, component: str = 'all') -> List[str]:
        """
        Get all hook names for a specific layer using new naming.

        Args:
            model: Hooked transformer model
            layer_idx: Layer index
            component: Component type ('attention', 'mlp', 'residual', 'all')

        Returns:
            List of hook names for the specified layer: blocks.{layer_idx}.{component}.hook_*
        """
        layer_pattern = f'blocks.{layer_idx}.'
        all_hooks = model.get_hook_names()
        layer_hooks = [name for name in all_hooks if layer_pattern in name]

        if component == 'all':
            return layer_hooks
        elif component == 'attention':
            return [name for name in layer_hooks if '.attn.hook_' in name]
        elif component == 'mlp':
            return [name for name in layer_hooks if '.mlp.hook_' in name]
        elif component == 'residual':
            return [name for name in layer_hooks if 'hook_resid_' in name and '.attn.hook_' not in name and '.mlp.hook_' not in name]
        else:
            raise ValueError(f"Unknown component: {component}")

    @staticmethod
    def get_attention_head_hook(layer_idx: int, hook_type: str = 'pattern') -> str:
        """Get hook name for attention heads."""
        return get_act_name(hook_type, layer_idx)

    @staticmethod
    def get_mlp_hook(layer_idx: int, hook_type: str = 'mid') -> str:
        """Get hook name for MLP activations."""
        return get_act_name(hook_type, layer_idx)

    @staticmethod
    def get_residual_hook(layer_idx: int, position: str = 'post') -> str:
        """Get hook name for residual stream."""
        return get_act_name(f'resid_{position}', layer_idx)

    @staticmethod
    def create_ablation_hook(ablation_type: str = 'zero', **kwargs) -> Callable:
        """Create common ablation hooks (unchanged from original)."""
        if ablation_type == 'zero':
            return lambda activation, hook_name: torch.zeros_like(activation)

        elif ablation_type == 'mean':
            def mean_ablation_hook(activation, hook_name):
                mean_val = activation.mean(dim=(0, 1), keepdim=True)
                return mean_val.expand_as(activation)
            return mean_ablation_hook

        elif ablation_type == 'noise':
            noise_std = kwargs.get('noise_std', 0.1)
            def noise_hook(activation, hook_name):
                noise = torch.randn_like(activation) * noise_std
                return activation + noise
            return noise_hook

        elif ablation_type == 'scale':
            scale_factor = kwargs.get('scale_factor', 0.0)
            return lambda activation, hook_name: activation * scale_factor

        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

    @staticmethod
    def create_head_ablation_hook(head_indices: Union[int, List[int]], ablation_type: str = 'zero') -> Callable:
        """Create hook to ablate specific attention heads."""
        if isinstance(head_indices, int):
            head_indices = [head_indices]

        def head_ablation_hook(activation, hook_name):
            if len(activation.shape) == 4:  # Attention pattern [batch, heads, seq, seq]
                modified = activation.clone()
                for head_idx in head_indices:
                    if head_idx < activation.shape[1]:
                        if ablation_type == 'zero':
                            modified[:, head_idx] = 0.0
                        elif ablation_type == 'mean':
                            modified[:, head_idx] = modified[:, head_idx].mean()
                return modified
            else:
                # For other activation types, assume head dimension is second
                modified = activation.clone()
                for head_idx in head_indices:
                    if len(activation.shape) >= 2 and head_idx < activation.shape[1]:
                        if ablation_type == 'zero':
                            modified[:, head_idx] = 0.0
                        elif ablation_type == 'mean':
                            modified[:, head_idx] = modified[:, head_idx].mean()
                return modified

        return head_ablation_hook

    @staticmethod
    def create_neuron_ablation_hook(neuron_indices: Union[int, List[int]], ablation_type: str = 'zero') -> Callable:
        """Create hook to ablate specific MLP neurons."""
        if isinstance(neuron_indices, int):
            neuron_indices = [neuron_indices]

        def neuron_ablation_hook(activation, hook_name):
            modified = activation.clone()
            for neuron_idx in neuron_indices:
                if neuron_idx < activation.shape[-1]:
                    if ablation_type == 'zero':
                        modified[..., neuron_idx] = 0.0
                    elif ablation_type == 'mean':
                        modified[..., neuron_idx] = modified[..., neuron_idx].mean()
            return modified

        return neuron_ablation_hook


class ActivationAnalyzer:
    """Updated activation analyzer with new hook naming."""

    def __init__(self, cache: ActivationCache):
        self.cache = cache

    def get_activation_stats(self, hook_name: str) -> Dict[str, float]:
        """Get basic statistics for an activation (unchanged)."""
        if hook_name not in self.cache:
            raise ValueError(f"Activation {hook_name} not found in cache")

        activation = self.cache[hook_name]

        return {
            'mean': activation.mean().item(),
            'std': activation.std().item(),
            'min': activation.min().item(),
            'max': activation.max().item(),
            'norm': activation.norm().item(),
            'sparsity': (activation == 0).float().mean().item(),
            'shape': list(activation.shape)
        }

    def get_attention_patterns(self, layer: int) -> Optional[torch.Tensor]:
        """Get attention patterns for a specific layer."""
        pattern_hook = get_act_name('pattern', layer)
        if pattern_hook in self.cache:
            return self.cache[pattern_hook]
        return None

    def get_mlp_activations(self, layer: int, position: str = 'mid') -> Optional[torch.Tensor]:
        """Get MLP activations for a specific layer."""
        mlp_hook = get_act_name(position, layer)  # 'pre', 'mid', or 'post'
        if mlp_hook in self.cache:
            return self.cache[mlp_hook]
        return None

    def get_residual_stream(self, layer: int, position: str = 'post') -> Optional[torch.Tensor]:
        """Get residual stream at a specific layer."""
        resid_hook = get_act_name(f'resid_{position}', layer)
        if resid_hook in self.cache:
            return self.cache[resid_hook]
        return None

    def analyze_attention_patterns(self, layer: int) -> Dict[str, Any]:
        """Analyze attention patterns with updated hook names."""
        pattern_hook = get_act_name('pattern', layer)
        if pattern_hook not in self.cache:
            raise ValueError(f"Attention pattern for layer {layer} not found in cache")

        patterns = self.cache[pattern_hook]  # [batch, heads, seq, seq]

        if len(patterns.shape) != 4:
            raise ValueError(f"Expected 4D attention pattern, got {patterns.shape}")

        batch_size, num_heads, seq_len, _ = patterns.shape

        # Compute attention statistics
        entropy = -(patterns * torch.log(patterns + 1e-8)).sum(dim=-1).mean(dim=(0, -1))  # [heads]

        # Locality (attention to nearby tokens)
        pos_diffs = torch.abs(torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1))
        local_mask = (pos_diffs <= 3).float()
        locality = (patterns * local_mask).sum(dim=(-2, -1)).mean(dim=0)  # [heads]

        # Attention to first/last tokens
        first_token_attn = patterns[:, :, :, 0].mean(dim=(0, -1))  # [heads]
        last_token_attn = patterns[:, :, :, -1].mean(dim=(0, -1))  # [heads]

        return {
            'layer': layer,
            'hook_name': pattern_hook,
            'entropy_per_head': entropy.tolist(),
            'locality_per_head': locality.tolist(),
            'first_token_attention': first_token_attn.tolist(),
            'last_token_attention': last_token_attn.tolist(),
            'num_heads': num_heads,
            'sequence_length': seq_len
        }


class InterventionExperiments:
    """Updated intervention experiments with new naming."""

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.cfg = model.cfg  # Use cfg to match TransformerLens

    def layer_ablation_study(
        self,
        input_tokens: torch.Tensor,
        layers_to_test: Optional[List[int]] = None,
        component: str = 'all',
        ablation_type: str = 'zero'
    ) -> Dict[int, Dict[str, Any]]:
        """
        Systematic ablation study across layers with updated hook names.
        """
        if layers_to_test is None:
            layers_to_test = list(range(self.cfg.n_layers))

        # Get baseline
        if hasattr(self.model, 'forward'):
            if self.model.model_type == 'decoder_only':
                baseline_logits = self.model(input_ids=input_tokens)
            elif self.model.model_type == 'encoder_only':
                baseline_logits = self.model(input_ids=input_tokens)
            else:  # encoder_decoder
                baseline_logits = self.model(input_ids=input_tokens)
        else:
            baseline_logits = self.model(input_tokens)

        # Handle different output formats
        if isinstance(baseline_logits, tuple):
            baseline_logits = baseline_logits[0]

        if self.model.model_type != 'encoder_only':
            baseline_loss = nn.CrossEntropyLoss()(
                baseline_logits.view(-1, baseline_logits.shape[-1]),
                input_tokens.view(-1)
            )
        else:
            # For encoder-only, we don't compute loss
            baseline_loss = torch.tensor(0.0)

        results = {}
        ablation_hook = HookUtils.create_ablation_hook(ablation_type)

        for layer_idx in layers_to_test:
            print(f"Testing layer {layer_idx} ablation...")

            # Get hooks for this layer using new naming
            layer_hooks = HookUtils.get_layer_hooks(self.model, layer_idx, component)

            if not layer_hooks:
                continue

            try:
                # Apply ablation to all hooks in this layer
                hook_specs = [(hook_name, ablation_hook) for hook_name in layer_hooks]

                ablated_logits = self.model.run_with_hooks(
                    input_ids=input_tokens,  # Use input_ids for TransformerLens compatibility
                    fwd_hooks=hook_specs
                )

                if isinstance(ablated_logits, tuple):
                    ablated_logits = ablated_logits[0]

                # Compute metrics
                if self.model.model_type != 'encoder_only':
                    ablated_loss = nn.CrossEntropyLoss()(
                        ablated_logits.view(-1, ablated_logits.shape[-1]),
                        input_tokens.view(-1)
                    )
                    loss_change = (ablated_loss - baseline_loss).item()
                else:
                    loss_change = 0.0

                logit_diff = torch.abs(ablated_logits - baseline_logits).mean().item()

                if self.model.model_type != 'encoder_only':
                    token_changes = (ablated_logits.argmax(dim=-1) != baseline_logits.argmax(dim=-1)).float().mean().item()
                else:
                    token_changes = 0.0

                results[layer_idx] = {
                    'loss_change': loss_change,
                    'logit_difference': logit_diff,
                    'token_change_rate': token_changes,
                    'hooks_ablated': layer_hooks,
                    'num_hooks': len(layer_hooks),
                    'component': component
                }

            except Exception as e:
                results[layer_idx] = {'error': str(e)}

        return results

    def head_ablation_study(
        self,
        input_tokens: torch.Tensor,
        layers_to_test: Optional[List[int]] = None
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Systematic ablation of individual attention heads with updated naming.
        """
        if layers_to_test is None:
            layers_to_test = list(range(min(3, self.cfg.n_layers)))

        baseline_logits = self.model(input_ids=input_tokens)
        if isinstance(baseline_logits, tuple):
            baseline_logits = baseline_logits[0]

        baseline_loss = nn.CrossEntropyLoss()(
            baseline_logits.view(-1, baseline_logits.shape[-1]),
            input_tokens.view(-1)
        )

        results = {}

        for layer_idx in layers_to_test:
            for head_idx in range(self.cfg.n_heads):
                print(f"Testing head {head_idx} in layer {layer_idx}...")

                # Create head ablation hook
                head_ablation_hook = HookUtils.create_head_ablation_hook(head_idx)

                # Use new naming convention
                pattern_hook_name = get_act_name('pattern', layer_idx)

                try:
                    ablated_logits = self.model.run_with_hooks(
                        input_ids=input_tokens,
                        fwd_hooks=[(pattern_hook_name, head_ablation_hook)]
                    )

                    if isinstance(ablated_logits, tuple):
                        ablated_logits = ablated_logits[0]

                    # Compute metrics
                    ablated_loss = nn.CrossEntropyLoss()(
                        ablated_logits.view(-1, ablated_logits.shape[-1]),
                        input_tokens.view(-1)
                    )

                    logit_diff = torch.abs(ablated_logits - baseline_logits).mean().item()

                    results[(layer_idx, head_idx)] = {
                        'loss_change': (ablated_loss - baseline_loss).item(),
                        'logit_difference': logit_diff,
                        'hook_name': pattern_hook_name
                    }

                except Exception as e:
                    results[(layer_idx, head_idx)] = {'error': str(e)}

        return results


# Utility functions for TransformerLens compatibility
def get_short_name(hook_name: str) -> str:
    """
    Convert full hook name to short name.

    Examples:
        'blocks.6.attn.hook_k' -> 'k6'
        'blocks.2.mlp.hook_pre' -> 'pre2'
        'hook_embed' -> 'embed'
    """
    if hook_name.startswith('hook_'):
        return hook_name[5:]  # Remove 'hook_' prefix

    # Parse blocks.X.Y.hook_Z pattern
    match = re.match(r'blocks\.(\d+)\.([^.]+)\.hook_(.+)', hook_name)
    if match:
        layer_num = match.group(1)
        component = match.group(2)  # 'attn' or 'mlp'
        activation = match.group(3)  # 'k', 'v', 'pre', etc.
        return f'{activation}{layer_num}'

    return hook_name


def expand_shorthand_hook_names(hook_names: List[str], model: HookedTransformer) -> List[str]:
    """
    Expand shorthand hook names to full names.

    Args:
        hook_names: List of shorthand names like ['k6', 'pre2', 'embed']
        model: Model to get layer count from

    Returns:
        List of full hook names like ['blocks.6.attn.hook_k', 'blocks.2.mlp.hook_pre', 'hook_embed']
    """
    expanded = []
    for name in hook_names:
        # Try to parse shorthand
        match = re.match(r'([a-zA-Z_]+)(\d+)', name)
        if match:
            base_name = match.group(1)
            layer_num = int(match.group(2))
            full_name = get_act_name(base_name, layer_num)
            expanded.append(full_name)
        else:
            # Assume it's already a full name or simple name
            if name in ['embed', 'pos', 'logits']:
                expanded.append(f'hook_{name}')
            else:
                expanded.append(name)

    return expanded


# Factory function for creating intervention hooks with new naming
def create_common_interventions(model: HookedTransformer) -> Dict[str, Dict[str, Any]]:
    """Factory function for common intervention types with updated naming."""

    interventions = {}

    # Attention interventions
    interventions['zero_attention_head_0'] = {
        'hook_name': lambda name: '.attn.hook_pattern' in name,
        'intervention': lambda act, name: act.clone().masked_fill_(
            torch.zeros_like(act[:, 0:1]).bool(), 0.0
        ),
        'description': 'Zero out attention head 0'
    }

    interventions['amplify_attention'] = {
        'hook_name': lambda name: '.attn.hook_pattern' in name,
        'intervention': lambda act, name: act * 1.5,
        'description': 'Amplify all attention by 50%'
    }

    # MLP interventions
    interventions['zero_mlp_neurons'] = {
        'hook_name': lambda name: '.mlp.hook_mid' in name,
        'intervention': lambda act, name: act.clone().masked_fill_(
            act > act.quantile(0.9), 0.0
        ),
        'description': 'Zero top 10% most active MLP neurons'
    }

    # Residual stream interventions
    interventions['add_noise'] = {
        'hook_name': lambda name: 'hook_resid_post' in name,
        'intervention': lambda act, name: act + torch.randn_like(act) * 0.01,
        'description': 'Add small noise to residual stream'
    }

    return interventions
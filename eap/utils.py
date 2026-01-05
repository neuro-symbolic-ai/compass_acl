from typing import List, Optional, Tuple, Union
from functools import partial

from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from einops import einsum
from trace.training.utils import get_model_source
from .graph import Graph, AttentionNode, LogitNode


def tokenize_plus(model: HookedTransformer, inputs: List[str], max_length: Optional[int] = None):
    """
    Tokenizes the input strings using the provided model.

    Args:
        model (HookedTransformer): The model used for tokenization.
        inputs (List[str]): The list of input strings to be tokenized.

    Returns:
        tuple: A tuple containing the following elements:
            - tokens (torch.Tensor): The tokenized inputs.
            - attention_mask (torch.Tensor): The attention mask for the tokenized inputs.
            - input_lengths (torch.Tensor): The lengths of the tokenized inputs.
            - n_pos (int): The maximum sequence length of the tokenized inputs.
    """
    if get_model_source(model) == 'transformerlens':
        if max_length is not None:
            old_n_ctx = model.cfg.n_ctx
            model.cfg.n_ctx = max_length
        tokens = model.to_tokens(inputs, prepend_bos=True, padding_side='right', truncate=(max_length is not None))
        if max_length is not None:
            model.cfg.n_ctx = old_n_ctx
        attention_mask = get_attention_mask(model.tokenizer, tokens, True)
        input_lengths = attention_mask.sum(1)
        n_pos = attention_mask.size(1)
        # print('Tokens shape', tokens.shape)
        # print('Attention mask shape', attention_mask.shape)
        # print('Input lengths shape', input_lengths.shape)
        # print('n_pos', n_pos)
        return tokens, attention_mask, input_lengths, n_pos

    if max_length is not None:
        old_n_ctx = model.cfg.n_ctx
        model.cfg.n_ctx = max_length
    tokens = model.to_tokens(inputs, prepend_bos=True, padding_side='right', truncate=(max_length is not None))
    if max_length is not None:
        model.cfg.n_ctx = old_n_ctx
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, input_lengths, n_pos

def tokenize_plus1(model: HookedTransformer, inputs: List[str], max_length: Optional[int] = None):
    """
    Tokenizes the input strings using the provided model.

    Args:
        model (HookedTransformer): The model used for tokenization.
        inputs (List[str]): The list of input strings to be tokenized.

    Returns:
        tuple: A tuple containing the following elements:
            - tokens (torch.Tensor): The tokenized inputs.
            - attention_mask (torch.Tensor): The attention mask for the tokenized inputs.
            - input_lengths (torch.Tensor): The lengths of the tokenized inputs.
            - n_pos (int): The maximum sequence length of the tokenized inputs.
    """
    if max_length is not None:
        old_n_ctx = model.cfg.n_ctx
        model.cfg.n_ctx = max_length
    tokens = model.to_tokens(inputs, prepend_bos=True, padding_side='right', truncate=(max_length is not None))
    if max_length is not None:
        model.cfg.n_ctx = old_n_ctx
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, input_lengths, n_pos


def make_hooks_and_matrices(
        model: HookedTransformer,
        graph: Graph,
        batch_size: int,
        n_pos: int,
        scores: Optional[Tensor],
        input_lengths: Optional[Tensor] = None,
        track_last_token_only: bool = False
):
    """Makes a matrix, and hooks to fill it and the score matrix up

    Args:
        model (HookedTransformer): model to attribute
        graph (Graph): graph to attribute
        batch_size (int): size of the particular batch you're attributing
        n_pos (int): size of the position dimension
        scores (Tensor): The scores tensor you intend to fill. If None, evaluation only
        input_lengths (Optional[Tensor]): Length of each sequence (needed for last-token tracking)
        track_last_token_only (bool): If True, only track activations at last token position
    """
    separate_activations = model.cfg.use_normalization_before_and_after and scores is None

    if track_last_token_only: # No position dimension - just (batch, n_forward, d_model)
        if separate_activations:
            activation_difference = torch.zeros(
                (2, batch_size, graph.n_forward, model.cfg.d_model),
                device=next(model.parameters()).device,
                dtype=model.cfg.dtype
            )
        else:
            activation_difference = torch.zeros(
                (batch_size, graph.n_forward, model.cfg.d_model),
                device=next(model.parameters()).device,
                dtype=model.cfg.dtype
            )
    else:
        # Original shape with n_pos
        if separate_activations:
            activation_difference = torch.zeros(
                (2, batch_size, n_pos, graph.n_forward, model.cfg.d_model),
                device=next(model.parameters()).device,
                dtype=model.cfg.dtype
            )
        else:
            activation_difference = torch.zeros(
                (batch_size, n_pos, graph.n_forward, model.cfg.d_model),
                device=next(model.parameters()).device,
                dtype=model.cfg.dtype
            )

    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []

    def activation_hook(index, activations, hook, add: bool = True):
        acts = activations.detach()

        try:
            if track_last_token_only:
                # Extract only the last valid token for each item in batch
                if input_lengths is None:
                    raise ValueError("input_lengths must be provided when track_last_token_only=True")

                batch_indices = torch.arange(acts.size(0), device=acts.device)
                last_token_acts = acts[batch_indices, input_lengths - 1]
                # Shape: (batch, d_model) or (batch, n_heads, d_model)
                if separate_activations:
                    if add:
                        activation_difference[0, :, index] += last_token_acts
                    else:
                        activation_difference[1, :, index] += last_token_acts
                else:
                    if add:
                        activation_difference[:, index] += last_token_acts
                    else:
                        activation_difference[:, index] -= last_token_acts
            else:
                # Original behaviour - all positions
                if separate_activations:
                    if add:
                        activation_difference[0, :, :, index] += acts
                    else:
                        activation_difference[1, :, :, index] += acts
                else:
                    if add:
                        activation_difference[:, :, index] += acts
                    else:
                        activation_difference[:, :, index] -= acts

        except RuntimeError as e:
            print(f"Error in activation_hook: {hook.name}")
            print(f"  activation_difference shape: {activation_difference.shape}")
            print(f"  acts shape: {acts.shape}")
            if track_last_token_only:
                print(f"  last_token_acts shape: {last_token_acts.shape}")
            raise e


    def gradient_hook(prev_index: int, bwd_index: Union[slice, int], gradients: torch.Tensor, hook):
        """Takes in a gradient and uses it and activation_difference to compute an update to the score matrix"""
        grads = gradients.detach()

        try:
            if track_last_token_only: # Extract gradients at last token position only
                if input_lengths is None:
                    raise ValueError("input_lengths must be provided when track_last_token_only=True")

                batch_indices = torch.arange(grads.size(0), device=grads.device)
                last_token_grads = grads[batch_indices, input_lengths - 1]
                # Shape: (batch, d_model) or (batch, n_heads, d_model)

                if last_token_grads.ndim == 2:
                    last_token_grads = last_token_grads.unsqueeze(1)

                s = einsum( # einsum without position dimension
                    activation_difference[:, :prev_index],
                    last_token_grads,
                    'batch forward hidden, batch backward hidden -> forward backward'
                )
                s = s.squeeze(1)
                scores[:prev_index, bwd_index] += s
            else:
                # Original behavior - all positions
                if grads.ndim == 3:
                    grads = grads.unsqueeze(2)
                s = einsum(
                    activation_difference[:, :, :prev_index],
                    grads,
                    'batch pos forward hidden, batch pos backward hidden -> forward backward'
                )
                s = s.squeeze(1)
                scores[:prev_index, bwd_index] += s

        except RuntimeError as e:
            print(f"Error in gradient_hook: {hook.name}")
            print(f"  activation_difference shape: {activation_difference.shape}")
            print(f"  grads shape: {grads.shape}")
            print(f"  prev_index: {prev_index}, bwd_index: {bwd_index}")
            print(f"  scores shape: {scores.shape}")
            raise e

    # Rest of the function stays the same - creating hooks for each node
    node = graph.nodes['input']
    fwd_index = graph.forward_index(node)
    fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
    fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))

    for layer in range(graph.cfg['n_layers']):
        node = graph.nodes[f'a{layer}.h0']
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        prev_index = graph.prev_index(node)
        for i, letter in enumerate('qkv'):
            bwd_index = graph.backward_index(node, qkv=letter)
            bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))

        node = graph.nodes[f'm{layer}']
        fwd_index = graph.forward_index(node)
        bwd_index = graph.backward_index(node)
        prev_index = graph.prev_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))

    node = graph.nodes['logits']
    prev_index = graph.prev_index(node)
    bwd_index = graph.backward_index(node)
    bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference

def compute_mean_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, per_position=False):
    """
    Compute the mean activations of a graph's nodes over a dataset.
    """
    def activation_hook(index, activations, hook, means=None, input_lengths=None):
        # defining a hook that will fill up our means tensor. Means is of shape
        # (n_pos, graph.n_forward, model.cfg.d_model) if per_position is True, otherwise
        # (graph.n_forward, model.cfg.d_model) 
        acts = activations.detach()

        # if we gave this hook input lengths, we assume we want to mean over positions
        if input_lengths is not None:
            mask = torch.zeros_like(activations)
            # mask out all padding positions
            mask[torch.arange(activations.size(0)), input_lengths - 1] = 1
            item_means = einsum(acts, mask, 'batch pos ... hidden, batch pos ... hidden -> batch ... hidden')
            
            # mean over the positions we did take, position-wise
            if len(item_means.size()) == 3:
                item_means /= input_lengths.unsqueeze(-1).unsqueeze(-1)
            else:
                item_means /= input_lengths.unsqueeze(-1)

            means[index] += item_means.sum(0)
        else:
            means[:, index] += acts.sum(0)

    # we're going to get all of the out hooks / indices we need for making hooks but we can't make them until we have input length masks
    processed_attn_layers = set()
    hook_points_indices = []
    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            processed_attn_layers.add(node.layer)
        
        if not isinstance(node, LogitNode):
            hook_points_indices.append((node.out_hook, graph.forward_index(node)))

    means_initialized = False
    total = 0
    for batch in tqdm(dataloader, desc='Computing mean'):
        # maybe the dataset is given as a tuple, maybe its just raw strings
        batch_inputs = batch[0] if isinstance(batch, tuple) else batch
        tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, batch_inputs, max_length=128)
        total += len(batch_inputs)

        if not means_initialized:
            # here is where we store the means
            if per_position:
                means = torch.zeros((n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
            else:
                means = torch.zeros((graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
            means_initialized = True

        if per_position:
            input_lengths = None
        add_to_mean_hooks = [(hook_point, partial(activation_hook, index, means=means, input_lengths=input_lengths)) for hook_point, index in hook_points_indices]

        with model.hooks(fwd_hooks=add_to_mean_hooks):
            if isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                              'model_type') and model.cfg.model_type == 'decoder_only':
                model(tgt=tokens, attention_mask=attention_mask)
            elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                'model_type') and model.cfg.model_type == 'encoder_only':
                model(src=tokens, attention_mask=attention_mask)
            elif isinstance(model, torch.nn.Module) and hasattr(model.cfg,
                                                                  'model_type') and model.cfg.model_type == 'encoder_decoder':
                model(src=tokens, tgt=tokens, src_attention_mask=attention_mask, tgt_attention_mask=attention_mask)
            else:
                model(tokens, attention_mask=attention_mask)

    means = means.squeeze(0)
    means /= total
    return means if per_position else means.mean(0)
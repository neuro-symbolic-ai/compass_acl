"""
Transformer components extended with TransformerLens-style hooks.
These add hook points to the original trace transformer components for interpretability.
The code is adapted from TransformerLens: https://github.com/TransformerLensOrg/TransformerLens/blob/50ee38b9a20695e90b8ce6a4aa20fd509c8175d9/transformer_lens/components/transformer_block.py#L25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .utils import expand_mask, repeat_along_head_dimension, repeat_along_head_dimension_tl
from .hook_points import HookPoint


class HookedMultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with hook points for interpretability.

    Hook points:
    - hook_q, hook_k, hook_v: Query, key, value after projection
    - hook_attn_scores: Attention scores before softmax
    - hook_pattern: Attention weights after softmax
    - hook_z: Attention output before final projection
    - hook_result: Final attention output
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Hook points for interpretability
        self.hook_q = HookPoint()  # Query vectors after projection
        self.hook_k = HookPoint()  # Key vectors after projection
        self.hook_v = HookPoint()  # Value vectors after projection
        self.hook_attn_scores = HookPoint()  # Raw attention scores (before softmax)
        self.hook_pattern = HookPoint()  # Attention weights (after softmax)
        self.hook_z = HookPoint()  # Attention output before final projection
        self.hook_result = HookPoint()  # Final attention output

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hook points at key locations.
        """
        batch_size = query.size(0)

        # Linear projections and reshape
        q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Hook points for Q, K, V
        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Hook point for raw attention scores
        scores = self.hook_attn_scores(scores)

        # Apply mask if provided
        if mask is not None:
            mask = expand_mask(mask)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        # Hook point for attention pattern
        attn_weights = self.hook_pattern(attn_weights)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        # Hook point for attention output (before final projection)
        context = self.hook_z(context)

        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output_linear(context)

        # Hook point for final result
        output = self.hook_result(output)

        return output, attn_weights


class HookedAttention(nn.Module):
    """
    Multi-head attention with TransformerLens/EAP-IG compatible hook names and shapes.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections (TransformerLens weight naming)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Hook points (TransformerLens naming)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_z = HookPoint()
        self.hook_result = HookPoint()

    def apply_attention_mask(self, attn_scores, attention_mask=None):
        """Apply masking compatible with both training and EAP"""
        batch_size = attn_scores.size(0)
        num_heads = attn_scores.size(1)
        query_ctx_length = attn_scores.size(-2)
        key_ctx_length = attn_scores.size(-1)

        # Causal mask
        causal_mask = torch.tril(torch.ones(query_ctx_length, key_ctx_length,
                                            device=attn_scores.device, dtype=torch.bool))
        # final_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, pos, pos]
        final_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

        if attention_mask is not None:
            # Expand attention mask to match attn_scores dimensions
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, pos]
            final_mask = final_mask & attention_mask.bool()

        # Use torch.where instead of masked_fill for EAP compatibility
        return torch.where(final_mask, attn_scores,
                           torch.tensor(-1e9, device=attn_scores.device, dtype=attn_scores.dtype))

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size = query.size(0)
        seq_len = query.size(1)

        q = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_head)
        k = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_head)
        v = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_head)

        # Hook points for Q, K, V in standard TransformerLens format
        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)

        # Transpose to [batch, heads, seq, d_head] for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq, d_head]
        k = k.transpose(1, 2)  # [batch, heads, seq, d_head]
        v = v.transpose(1, 2)  # [batch, heads, seq, d_head]

        # Compute attention scores [batch, heads, seq, seq]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # print(f'scores shape: {scores.shape}')
        # print(f'sample scores: {scores[0,0,0,:5]}')

        # Hook point for raw attention scores - keep [batch, heads, seq, seq]
        scores = self.hook_attn_scores(scores)

        # Apply mask if provided
        # scores = self.apply_attention_mask(scores, mask)
        # if mask is not None:
        #     mask = expand_mask(mask)
        #     print(f'mask shape: {mask.shape}')
        #     scores = scores.masked_fill(mask == 0, -1e9)
        #     print(f'scores after mask shape: {scores.shape}')
        if mask is not None and mask.shape[-1] == scores.shape[-1]:  # todo: fix this
            try:
                # Only apply compatible masks
                while mask.dim() < scores.dim():
                    mask = mask.unsqueeze(0)
                mask = mask.expand_as(scores)
                # print(f'mask shape: {mask.shape}')
                scores = scores.masked_fill(mask == 0, -1e9)
                # print(f'scores after mask shape: {scores.shape}')
            except:
                pass
        # print(f'scores shape: {scores.shape}')
        # print(f'sample scores: {scores[0, 0, 0, :5]}')

        # Apply softmax to get attention weights [batch, heads, seq, seq]
        attn_weights = F.softmax(scores, dim=-1)

        # Hook point for attention pattern - keep [batch, heads, seq, seq]
        attn_weights = self.hook_pattern(attn_weights)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        # print(f'attn_weights shape: {attn_weights.shape}')
        # print(f'sample attn_weights: {attn_weights[0,0,0,:5]}')

        # Apply attention weights to values [batch, heads, seq, d_head]
        context = torch.matmul(attn_weights, v)
        # print(f'context shape: {context.shape}')
        # print(f'sample context: {context[0,0,0,:5]}')

        z_for_hook = context.transpose(1, 2)  # [batch, seq, heads, d_head]
        # context = z.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        context = z_for_hook.contiguous().view(batch_size, -1, self.d_model)
        # print(f'context reshaped for W_O shape: {context.shape}')
        # print(f'sample context reshaped for W_O: {context[0,0,:5]}')

        result = self.W_O(context)
        # print(result.shape)
        # print(f'sample result: {result[0,0,:5]}')

        # Create per-head contributions for EAP-IG (but don't use for actual computation)
        # This expands each head to d_model space for hook compatibility
        z_expanded = z_for_hook.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_model // self.d_head)
        z_expanded = z_expanded.contiguous().view(batch_size, seq_len, self.n_heads, self.d_model)
        z_expanded = self.hook_z(z_expanded)  # Store EAP-IG format

        # Hook result - create per-head format for EAP-IG but return original result
        result_for_hook = result.view(batch_size, seq_len, self.n_heads, self.d_head)
        result_expanded = result_for_hook.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_model // self.d_head)
        result_expanded = result_expanded.contiguous().view(batch_size, seq_len, self.n_heads, self.d_model)
        result_expanded = self.hook_result(result_expanded)

        return result

class HookedFeedForward(nn.Module):
    """
    Position-wise feed-forward network with hook points.
    todo: complete the coding with fnn optional in main model
    Hook points:
    - hook_pre: Input to MLP
    - hook_mid: After first linear layer and activation
    - hook_post: Final MLP output
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Hook points
        # self.hook_pre = HookPoint()  # Input to MLP
        self.hook_mlp_in = HookPoint()  # Input to MLP (alternate name)
        self.hook_mid = HookPoint()  # After first layer + activation
        self.hook_post = HookPoint()  # Final output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hook point for MLP input
        x = self.hook_mlp_in(x)

        # First linear layer + activation
        intermediate = F.relu(self.linear1(x))

        # Hook point for intermediate activation
        intermediate = self.hook_mid(intermediate)

        # Second linear layer
        output = self.linear2(self.dropout(intermediate))

        # Hook point for final MLP output
        output = self.hook_post(output)

        return output


class HookedMLP(nn.Module):
    """
    MLP with TransformerLens/EAP-IG compatible hook names.

    Hook points:
    - hook_pre: Input to MLP
    - hook_mid: After first linear layer and activation
    - hook_post: Final MLP output
    """

    def __init__(self, d_model: int, d_mlp: int, dropout: float = 0.1):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)  # TransformerLens naming
        self.W_out = nn.Linear(d_mlp, d_model)  # TransformerLens naming
        self.dropout = nn.Dropout(dropout)

        # Hook points (TransformerLens naming)
        self.hook_pre = HookPoint()  # Input to MLP
        # self.hook_mlp_in = HookPoint()  # Alternate name for input to MLP
        self.hook_mid = HookPoint()  # After first layer + activation
        self.hook_post = HookPoint()  # Final output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with TransformerLens-compatible hooks."""
        # Hook point for MLP input - shape: [batch, seq, d_model]
        # x = self.hook_pre(x)
        x = self.hook_pre(x)  # Alternate hook name

        # First linear layer + activation - shape: [batch, seq, d_mlp]
        mid = F.relu(self.W_in(x))

        # Hook point for intermediate activation - shape: [batch, seq, d_mlp]
        mid = self.hook_mid(mid)

        # Second linear layer with dropout - shape: [batch, seq, d_model]
        out = self.W_out(self.dropout(mid))

        # Hook point for final MLP output - shape: [batch, seq, d_model]
        out = self.hook_post(out)

        return out


class HookedBlock(nn.Module):
    """
    Unified transformer block compatible with TransformerLens/EAP-IG.

    Supports all architectures:
    - encoder_only: Self-attention + MLP
    - decoder_only: Self-attention + MLP (with causal mask)
    - encoder: Self-attention + MLP (for encoder-decoder)
    - decoder: Self-attention + cross-attention + MLP (for encoder-decoder)

    Hook points:
    - hook_resid_pre: Input to the block
    - hook_attn_out: Attention output (before adding to residual)
    - hook_resid_mid: After attention + residual (before MLP)
    - hook_mlp_out: MLP output (before adding to residual)
    - hook_resid_post: Final block output

    For decoder blocks in encoder-decoder:
    - hook_cross_attn_out: Cross-attention output
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_mlp: int,
            dropout: float = 0.1,
            no_mlp: bool = False,
            block_type: str = 'decoder_only',
            use_split_qkv_input: bool = False,
            use_attn_result: bool = False,
    ):
        super().__init__()
        self.block_type = block_type
        self.n_heads = n_heads
        self.no_mlp = no_mlp
        self.use_split_qkv_input = use_split_qkv_input
        self.use_attn_result = use_attn_result

        # Self-attention (all block types have this)
        self.attn = HookedAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)  # Pre-attention LayerNorm

        # Cross-attention (only for decoder in encoder-decoder)
        if block_type == 'decoder':
            self.cross_attn = HookedAttention(d_model, n_heads, dropout)
            self.ln2 = nn.LayerNorm(d_model)  # Pre-cross-attention LayerNorm
            self.ln3 = nn.LayerNorm(d_model)  # Pre-MLP LayerNorm
        else:
            self.ln2 = nn.LayerNorm(d_model)  # Pre-MLP LayerNorm

        # MLP
        if not no_mlp:
            self.mlp = HookedMLP(d_model, d_mlp, dropout)
            self.hook_mlp_in = HookPoint()
            self.hook_mlp_out = HookPoint()

        self.dropout = nn.Dropout(dropout)

        # Hook points (TransformerLens naming)
        self.hook_resid_pre = HookPoint()  # Block input
        self.hook_attn_out = HookPoint()  # Attention output (pre-residual)
        self.hook_resid_mid = HookPoint()  # After attention + residual

        # Additional hook points for attention
        self.hook_q_input = HookPoint()  # Input to Q computation
        self.hook_k_input = HookPoint()  # Input to K computation
        self.hook_v_input = HookPoint()  # Input to V computation

        if block_type == 'decoder':
            self.hook_cross_attn_out = HookPoint()  # Cross-attention output
            self.hook_resid_mid2 = HookPoint()  # After cross-attention + residual

        self.hook_resid_post = HookPoint()  # Block output

    def forward(
            self,
            residual: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            encoder_output: Optional[torch.Tensor] = None,
            cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for unified transformer block.

        Args:
            residual: Input residual stream
            mask: Self-attention mask
            encoder_output: Encoder output (for cross-attention)
            cross_mask: Cross-attention mask
        """
        # Hook point for block input
        residual = self.hook_resid_pre(residual)

        # Self-attention sublayer
        normalized = self.ln1(residual)
        if self.use_split_qkv_input:
            q_input_expanded = repeat_along_head_dimension_tl(normalized, self.n_heads)  # [batch, pos, n_heads, d_model]
            k_input_expanded = repeat_along_head_dimension_tl(normalized, self.n_heads)  # [batch, pos, n_heads, d_model]
            v_input_expanded = repeat_along_head_dimension_tl(normalized, self.n_heads)  # [batch, pos, n_heads, d_model]
            q_input_expanded = self.hook_q_input(q_input_expanded)
            k_input_expanded = self.hook_k_input(k_input_expanded)
            v_input_expanded = self.hook_v_input(v_input_expanded)
            q_input = q_input_expanded[:, :, 0, :]  # [batch, pos, d_model]
            k_input = k_input_expanded[:, :, 0, :]  # [batch, pos, d_model]
            v_input = v_input_expanded[:, :, 0, :]  # [batch, pos, d_model]
        else:
            q_input = self.hook_q_input(normalized)  # Usually same as normalized
            k_input = self.hook_k_input(normalized)  # Usually same as normalized
            v_input = self.hook_v_input(normalized)  # Usually same as normalized

        # q_input_expanded = repeat_along_head_dimension(normalized, self.n_heads)  # [batch, pos, n_heads, d_model]
        # k_input_expanded = repeat_along_head_dimension(normalized, self.n_heads)  # [batch, pos, n_heads, d_model]
        # v_input_expanded = repeat_along_head_dimension(normalized, self.n_heads)  # [batch, pos, n_heads, d_model]
        # q_input_expanded = self.hook_q_input(q_input_expanded)
        # k_input_expanded = self.hook_k_input(k_input_expanded)
        # v_input_expanded = self.hook_v_input(v_input_expanded)
        # q_input = q_input_expanded[:, :, 0, :]  # [batch, pos, d_model]
        # k_input = k_input_expanded[:, :, 0, :]  # [batch, pos, d_model]
        # v_input = v_input_expanded[:, :, 0, :]  # [batch, pos, d_model]
        attn_out = self.attn(q_input, k_input, v_input, mask)
        attn_out = self.hook_attn_out(attn_out) # Hook point for attention output

        # Add residual connection
        residual = residual + self.dropout(attn_out)

        # Hook point after self-attention + residual
        residual = self.hook_resid_mid(residual)

        # Cross-attention sublayer (only for decoder in encoder-decoder)
        if self.block_type == 'decoder':
            if encoder_output is None:
                raise ValueError("encoder_output required for decoder block")

            normalized = self.ln2(residual)
            cross_attn_out = self.cross_attn(normalized, encoder_output, encoder_output, cross_mask)

            # Hook point for cross-attention output
            cross_attn_out = self.hook_cross_attn_out(cross_attn_out)

            # Add residual connection
            residual = residual + self.dropout(cross_attn_out)

            # Hook point after cross-attention + residual
            residual = self.hook_resid_mid2(residual)

            # Use ln3 for MLP in decoder case
            mlp_ln = self.ln3
        else:
            # Use ln2 for MLP in other cases
            mlp_ln = self.ln2

        # MLP sublayer
        if not self.no_mlp:
            normalized = mlp_ln(residual)
            mlp_input = self.hook_mlp_in(normalized)
            mlp_out = self.mlp(mlp_input)

            # Hook point for MLP output
            mlp_out = self.hook_mlp_out(mlp_out)

            # Add residual connection
            residual = residual + self.dropout(mlp_out)

        # Hook point for final block output
        residual = self.hook_resid_post(residual)

        return residual


class HookedEncoderLayer(nn.Module):
    """
    Encoder layer with hook points for interpretability.

    Hook points:
    - hook_resid_pre: Input to the layer (residual stream)
    - hook_attn_out: Attention output (before adding to residual)
    - hook_resid_mid: After attention + residual (input to MLP)
    - hook_mlp_out: MLP output (before adding to residual)
    - hook_resid_post: Final layer output (residual stream)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, no_fnn: bool = False):
        super().__init__()
        self.attn = HookedMultiHeadAttention(d_model, n_heads, dropout)

        if not no_fnn:
            self.feed_forward = HookedFeedForward(d_model, d_ff, dropout)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.feed_forward = None

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Hook points for residual stream tracking
        self.hook_resid_pre = HookPoint()  # Layer input
        self.hook_attn_out = HookPoint()  # Attention output (pre-residual)
        self.hook_resid_mid = HookPoint()  # After attention + residual
        self.hook_mlp_out = HookPoint()  # MLP output (pre-residual)
        self.hook_resid_post = HookPoint()  # Layer output

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Hook point for layer input (residual stream)
        x = self.hook_resid_pre(x)

        # Self-attention
        attn_output, attn_weights = self.attn(x, x, x, mask)

        # Hook point for attention output (before residual)
        attn_output = self.hook_attn_out(attn_output)

        # Add residual connection and normalize
        x = self.norm1(x + self.dropout(attn_output))

        # Hook point after attention + residual (input to MLP)
        x = self.hook_resid_mid(x)

        if self.feed_forward is None:
            # Hook point for final output
            x = self.hook_resid_post(x)
            return x

        # Feed-forward network
        ff_output = self.feed_forward(x)

        # Hook point for MLP output (before residual)
        ff_output = self.hook_mlp_out(ff_output)

        # Add residual connection and normalize
        x = self.norm2(x + self.dropout(ff_output))

        # Hook point for final layer output
        x = self.hook_resid_post(x)

        return x


class HookedDecoderLayer(nn.Module):
    """
    Decoder layer with hook points (for encoder-decoder architecture).

    Hook points:
    - hook_resid_pre: Layer input
    - hook_attn_out: Self-attention output (pre-residual)
    - hook_resid_mid: After self-attention + residual
    - hook_cross_attn_out: Cross-attention output (pre-residual)
    - hook_resid_mid2: After cross-attention + residual
    - hook_mlp_out: MLP output (pre-residual)
    - hook_resid_post: Final layer output
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, no_fnn: bool = False):
        super().__init__()
        self.attn = HookedMultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = HookedMultiHeadAttention(d_model, n_heads, dropout)

        if not no_fnn:
            self.feed_forward = HookedFeedForward(d_model, d_ff, dropout)
            self.norm3 = nn.LayerNorm(d_model)
        else:
            self.feed_forward = None

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Hook points
        self.hook_resid_pre = HookPoint()  # Layer input
        self.hook_attn_out = HookPoint()  # Self-attention output
        self.hook_resid_mid = HookPoint()  # After self-attention + residual
        self.hook_cross_attn_out = HookPoint()  # Cross-attention output
        self.hook_resid_mid2 = HookPoint()  # After cross-attention + residual
        self.hook_mlp_out = HookPoint()  # MLP output
        self.hook_resid_post = HookPoint()  # Final output

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Hook point for layer input
        x = self.hook_resid_pre(x)

        # Self-attention
        attn_output, attn_weights = self.attn(x, x, x, self_mask)

        # Hook point for self-attention output
        attn_output = self.hook_attn_out(attn_output)

        # Add residual and normalize
        x = self.norm1(x + self.dropout(attn_output))

        # Hook point after self-attention + residual
        x = self.hook_resid_mid(x)

        # Cross-attention
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output, cross_mask
        )

        # Hook point for cross-attention output
        cross_attn_output = self.hook_cross_attn_out(cross_attn_output)

        # Add residual and normalize
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Hook point after cross-attention + residual
        x = self.hook_resid_mid2(x)

        if self.feed_forward is None:
            x = self.hook_resid_post(x)
            return x

        # Feed-forward
        ff_output = self.feed_forward(x)

        # Hook point for MLP output
        ff_output = self.hook_mlp_out(ff_output)

        # Add residual and normalize
        x = self.norm3(x + self.dropout(ff_output))

        # Hook point for final output
        x = self.hook_resid_post(x)

        self.last_residual = x  # For debugging
        return x


class HookedDecoderOnlyLayer(nn.Module):
    """
    Decoder-only layer with hook points (GPT-style).

    Hook points:
    - hook_resid_pre: Layer input (residual stream)
    - hook_attn_out: Attention output (pre-residual)
    - hook_resid_mid: After attention + residual (input to MLP)
    - hook_mlp_out: MLP output (pre-residual)
    - hook_resid_post: Final layer output (residual stream)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, no_fnn: bool = False):
        super().__init__()
        self.attn = HookedMultiHeadAttention(d_model, n_heads, dropout)

        if not no_fnn:
            self.feed_forward = HookedFeedForward(d_model, d_ff, dropout)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.feed_forward = None

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.no_ffn = no_fnn

        # Hook points
        self.hook_resid_pre = HookPoint()  # Layer input
        self.hook_attn_out = HookPoint()  # Attention output (pre-residual)
        self.hook_resid_mid = HookPoint()  # After attention + residual
        self.hook_mlp_out = HookPoint()  # MLP output (pre-residual)
        self.hook_resid_post = HookPoint()  # Final output

    def forward(self, x: torch.Tensor, self_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Hook point for layer input (residual stream)
        x = self.hook_resid_pre(x)

        # Self-attention
        attn_output, attn_weights = self.attn(x, x, x, self_mask)

        # Hook point for attention output (before residual)
        attn_output = self.hook_attn_out(attn_output)

        # Add residual and normalize
        x = self.norm1(x + self.dropout(attn_output))

        # Hook point after attention + residual
        x = self.hook_resid_mid(x)

        if self.no_ffn:
            x = self.hook_resid_post(x)
            return x

        # Feed-forward network
        ff_output = self.feed_forward(x)

        # Hook point for MLP output (before residual)
        ff_output = self.hook_mlp_out(ff_output)

        # Add residual and normalize
        x = self.norm2(x + self.dropout(ff_output))

        # Hook point for final output (residual stream)
        x = self.hook_resid_post(x)

        self.last_residual = x  # For debugging
        return x


class HookedEncoderLegacy(nn.Module):
    """
    Transformer encoder with hooked layers.
    """

    def __init__(self, d_model: int, num_layers: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1, no_fnn: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            HookedEncoderLayer(d_model, n_heads, d_ff, dropout, no_fnn)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x

class HookedEncoder(nn.Module):
    """Container for multiple encoder blocks (legacy support)."""
    def __init__(self, d_model: int, num_layers: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1, no_fnn: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            HookedBlock(d_model, n_heads, d_ff, dropout, no_fnn, 'encoder_only')
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x

class HookedDecoderLegacy(nn.Module):
    """
    Transformer decoder with hooked layers.
    """

    def __init__(self, d_model: int, num_layers: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1, decoder_only: bool = False,
                 no_fnn: bool = False):
        super().__init__()
        self.decoder_only = decoder_only

        if decoder_only:
            self.layers = nn.ModuleList([
                HookedDecoderOnlyLayer(d_model, n_heads, d_ff, dropout, no_fnn)
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                HookedDecoderLayer(d_model, n_heads, d_ff, dropout, no_fnn)
                for _ in range(num_layers)
            ])

    def forward(self, x: torch.Tensor, encoder_output: Optional[torch.Tensor] = None,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.decoder_only:
            for layer in self.layers:
                x = layer(x, self_mask)
        else:
            if encoder_output is None:
                raise ValueError("encoder_output cannot be None for encoder-decoder model")
            for layer in self.layers:
                x = layer(x, encoder_output, self_mask, cross_mask)

        return x


class HookedDecoder(nn.Module):
    """Container for multiple decoder blocks (legacy support)."""

    def __init__(self, d_model: int, num_layers: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1, decoder_only: bool = False,
                 no_fnn: bool = False):
        super().__init__()
        self.decoder_only = decoder_only

        block_type = 'decoder_only' if decoder_only else 'decoder'
        self.blocks = nn.ModuleList([
            HookedBlock(d_model, n_heads, d_ff, dropout, no_fnn, block_type)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, encoder_output: Optional[torch.Tensor] = None,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        self_attentions = []  # todo: maybe convert to a tensor
        cross_attentions = []  # todo: maybe convert to a tensor

        for block in self.blocks:
            if self.decoder_only:
                x = block(x, self_mask)
            else:
                if encoder_output is None:
                    raise ValueError("encoder_output cannot be None for encoder-decoder model")
                x = block(x, self_mask, encoder_output, cross_mask)

        return x

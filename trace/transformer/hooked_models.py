"""
Updated Transformer model with TransformerLens-style hooks integration.
This extends trace existing models.py with hook functionality.
"""
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List, Mapping, Any, Callable, Literal
from .config import TransformerConfig
from .components import PositionalEncoding
from .hook_points import HookedRootModule, HookPoint, ActivationCache
from .hooked_components import HookedEncoder, HookedDecoder, HookedBlock

# add a decorator to register the class as a model in TRACE - add it to the transformer class itself
class HookedTransformer(HookedRootModule):
    """
    Your transformer model extended with TransformerLens-style hook functionality.

    This is a replacement for the Transformer class but adds:
    - Hook points at key activations (attention outputs, MLP outputs, residual stream)
    - run_with_hooks() and run_with_cache() capabilities - apart from the forward method
    - Activation caching and intervention support
    """

    def __init__(self, config: TransformerConfig, tokenizer: Optional[Any] = None):
        """
        Initialize hooked transformer model from configuration.

        Args:
            config: TransformerConfig object containing all model parameters
        """
        super().__init__()

        # Store configuration
        self.cfg = config
        self.model_type = config.model_type
        self.vocab_size = config.vocab_size
        self.d_vocab = config.vocab_size if config.d_vocab is None else config.d_vocab
        self.d_model = config.d_model
        self.pad_idx = config.pad_idx
        self.n_heads = config.n_heads
        self.d_heads = config.d_model // config.n_heads if config.d_heads is None else config.d_heads
        if config.model_type == 'encoder_only':
            self.n_layers = config.num_encoder_layers
            self.cfg.n_layers = config.num_encoder_layers
        elif config.model_type == 'decoder_only':
            self.n_layers = config.num_decoder_layers
            self.cfg.n_layers = config.num_decoder_layers
        elif config.model_type == 'encoder_decoder':
            self.n_layers = config.num_encoder_layers + config.num_decoder_layers
            self.cfg.n_layers = config.num_encoder_layers + config.num_decoder_layers
        self.d_ff = config.d_ff if config.d_ff is not None else config.d_model * 4 # d_mlp??

        # Input embedding and positional encoding
        # self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        # self.positional_encoding = PositionalEncoding(config.d_model, config.n_ctx, config.dropout)
        self.pos_embed = PositionalEncoding(config.d_model, config.n_ctx, config.dropout)

        # Hook points for embeddings
        self.hook_embed = HookPoint()  # After embedding lookup #todo: I tried to put this before embedding but it caused issues
        self.hook_pos = HookPoint()  # After positional encoding

        self.set_tokenizer(tokenizer=tokenizer)  # Placeholder for tokenizer setup

        # Initialize encoder and/or decoder based on model_type
        if config.model_type == 'decoder_only':
            self.blocks = nn.ModuleList([
                HookedBlock(
                    config.d_model, config.n_heads, config.d_ff,
                    config.dropout, config.no_fnn,
                    block_type='decoder_only',
                    use_split_qkv_input=config.use_split_qkv_input, use_attn_result=config.use_attn_result
                )
                for _ in range(config.num_decoder_layers)
            ])
        elif config.model_type == 'encoder_only':
            self.blocks = nn.ModuleList([
                HookedBlock(
                    config.d_model, config.n_heads, config.d_ff,
                    config.dropout, config.no_fnn,
                    block_type='encoder_only',
                    use_split_qkv_input=config.use_split_qkv_input, use_attn_result=config.use_attn_result
                )
                for _ in range(config.num_encoder_layers)
            ])
        elif config.model_type == 'encoder_decoder':
            # For encoder-decoder, we'll have separate encoder and decoder blocks
            self.encoder_blocks = nn.ModuleList([
                HookedBlock(
                    config.d_model, config.n_heads, config.d_ff,
                    config.dropout, config.no_fnn,
                    block_type='encoder',
                    use_split_qkv_input=config.use_split_qkv_input, use_attn_result=config.use_attn_result
                )
                for _ in range(config.num_encoder_layers)
            ])
            self.decoder_blocks = nn.ModuleList([
                HookedBlock(
                    config.d_model, config.n_heads, config.d_ff,
                    config.dropout, config.no_fnn,
                    block_type='decoder', use_split_qkv_input=config.use_split_qkv_input, use_attn_result=config.use_attn_result
                )
                for _ in range(config.num_decoder_layers)
            ])
        # # Output projection for decoder-based models
        # if config.model_type in ['decoder_only', 'encoder_decoder']:
        #     self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        #     self.hook_logits = HookPoint()  # Before final output

        # Output projection for generative models
        if config.model_type in ['decoder_only', 'encoder_decoder']:
            self.ln_final = nn.LayerNorm(config.d_model)
            self.hook_normalized = HookPoint()  # After final layer norm

            self.unembed = nn.Linear(config.d_model, config.vocab_size)  # TransformerLens uses 'unembed'
            self.hook_logits = HookPoint()  # Before final output
            # Final layer norm (following TransformerLens convention)


        # Set up hook infrastructure
        self.setup_hooks()
        # Initialize weights
        # self.apply(self._init_weights)
        self.to(config.device)

    def set_tokenizer(self, tokenizer):
        """Set up tokenizer with required attributes"""
        # This will be set when tokenizer is attached
        if tokenizer is None:
            self.tokenizer = None
            return
        # Ensure tokenizer has required attributes
        if not hasattr(tokenizer, 'bos_token_id') or tokenizer.bos_token_id is None or tokenizer.bos_token is None:
            tokenizer.bos_token_id = 1
        if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None or tokenizer.eos_token is None:
            tokenizer.eos_token_id = 2
        if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None or tokenizer.pad_token is None:
            tokenizer.pad_token_id = 0
        if not hasattr(tokenizer, 'unk_token_id') or tokenizer.unk_token_id is None or tokenizer.unk_token is None:
            tokenizer.unk_token_id = 3
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls, config: TransformerConfig) -> 'HookedTransformer':
        """Create hooked transformer model from configuration."""
        return cls(config)

    def generate_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Generate padding mask for attention
        Source: https://medium.com/@swarms/understanding-masking-in-pytorch-for-attention-mechanisms-e725059fd49f
        """
        return (x == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder self-attention """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def to(self, device: torch.device) -> 'HookedTransformer':
        """Move model to specified device."""
        super().to(device)
        self.cfg.device = device
        return self

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,  # TransformerLens uses 'input_ids'
            # Keep trace compatibility
            src: Optional[torch.Tensor] = None,
            tgt: Optional[torch.Tensor] = None,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_type: str = "logits",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with TransformerLens/EAP-IG compatible interface.

        Args:
            input_ids: Input tokens [batch, seq] (TransformerLens style)
            Or use original interface:
            src: Source tokens for encoder
            tgt: Target tokens for decoder
            src_mask: Source sequence padding mask
            tgt_mask: Target sequence causal mask
            memory_mask: Cross-attention mask
            attention_mask: Attention mask (TransformerLens style)
            return_type: 'logits' or 'loss' or 'both'
        """
        # Handle input format conversion
        if input_ids is not None:
            if self.model_type == 'decoder_only':
                tgt = input_ids
            elif self.model_type == 'encoder_only':
                src = input_ids
            elif self.model_type == 'encoder_decoder':
                # For encoder-decoder, assume input_ids is the source
                src = input_ids
                if tgt is None:
                    raise ValueError("encoder_decoder model requires both src and tgt (or specify tgt separately)")
        # Handle attention mask
        if attention_mask is not None:
            if self.model_type == 'decoder_only':
                tgt_mask = attention_mask
            elif self.model_type == 'encoder_only':
                src_mask = attention_mask

        # Route to appropriate forward method
        if self.model_type == 'encoder_only':
            return self._forward_encoder_only(src=src, src_mask=src_mask)
        elif self.model_type == 'decoder_only':
            return self._forward_decoder_only(tgt=tgt, attention_mask=tgt_mask)
        elif self.model_type == 'encoder_decoder':
            return self._forward_encoder_decoder(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _forward_encoder_only(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for encoder-only models."""
        assert src is not None, "Source sequence required for encoder-only model"

        if src.device != self.cfg.device:
            src = src.to(self.cfg.device)
            # print(f"Moved src to model device: {src.device}")

        # Create padding mask if not provided
        if src_mask is None:
            src_mask = self.generate_padding_mask(src)

        # Embed and add positional encoding
        residual = self.embed(src)
        residual = self.hook_embed(residual)  # Hook after embedding

        residual = self.pos_embed(residual)
        residual = self.hook_pos(residual)  # Hook after positional encoding

        # Pass through encoder blocks
        for block in self.blocks: ##todo
            residual = block(residual, src_mask)

        return residual

    def _forward_decoder_only(
            self,
            tgt: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for decoder-only models with TransformerLens naming."""
        assert tgt is not None, "Input tokens required for decoder-only model"

        # Move tgt to model device if needed
        if tgt.device != self.cfg.device:
            tgt = tgt.to(self.cfg.device)
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        # elif attention_mask.device != self.cfg.device:
        #     attention_mask = attention_mask.to(self.cfg.device)
        if attention_mask is not None and attention_mask.device != self.cfg.device:
            attention_mask = attention_mask.to(self.cfg.device)
        # Embed and add positional encoding
        residual = self.embed(tgt)
        residual = self.hook_embed(residual)  # Hook after embedding

        residual = self.pos_embed(residual)
        residual = self.hook_pos(residual)  # Hook after positional encoding

        # Pass through decoder blocks
        for block in self.blocks:
            residual = block(residual, attention_mask)

        # Final layer norm
        out = self.ln_final(residual)
        out = self.hook_normalized(out)

        # Project to vocabulary
        logits = self.unembed(out)
        logits = self.hook_logits(logits)
        # print(f"logits shape: {logits.shape}")
        # print(f"Sample logits: {logits[0, :5, :5]}")  # Print a small sample of logits
        return logits

    def _forward_encoder_decoder(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for encoder-decoder models with TransformerLens naming."""
        assert src is not None, "Source sequence required for encoder-decoder model"
        assert tgt is not None, "Target sequence required for encoder-decoder model"
        model_device = next(self.parameters()).device
        if tgt.device != model_device:
            tgt = tgt.to(model_device)
            # print(f"Moved tgt to model device: {tgt.device}")
        if src.device != model_device:
            src = src.to(model_device)
            # print(f"Moved src to model device: {src.device}")

        # Create masks if not provided
        if src_mask is None:
            src_mask = self.generate_padding_mask(src)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        if memory_mask is None:
            memory_mask = self.generate_padding_mask(src)

        # Encode
        src_residual = self.embed(src)
        src_residual = self.hook_embed(src_residual)
        src_residual = self.pos_embed(src_residual)
        src_residual = self.hook_pos(src_residual)

        for block in self.encoder_blocks:
            src_residual = block(src_residual, src_mask)

        # Decode
        tgt_residual = self.embed(tgt)
        tgt_residual = self.hook_embed(tgt_residual)
        tgt_residual = self.pos_embed(tgt_residual)
        tgt_residual = self.hook_pos(tgt_residual)

        for block in self.decoder_blocks:
            tgt_residual = block(tgt_residual, tgt_mask, src_residual, memory_mask)

        # Final processing
        tgt_residual = self.ln_final(tgt_residual)
        tgt_residual = self.hook_normalized(tgt_residual)

        logits = self.unembed(tgt_residual)
        logits = self.hook_logits(logits)

        return logits

    def get_act_name(self, name: str, layer: Optional[int] = None, layer_type: Optional[str] = None) -> str:
        """
        Generate TransformerLens-style activation names.

        Examples:
        - get_act_name('k', 6) -> 'blocks.6.attn.hook_k'
        - get_act_name('pre', 2) -> 'blocks.2.mlp.hook_pre'
        - get_act_name('embed') -> 'hook_embed'
        """
        if name == 'embed':
            return 'hook_embed'
        elif name == 'pos':
            return 'hook_pos'
        elif name == 'logits':
            return 'hook_logits'
        elif layer is not None:
            if name in ['q', 'k', 'v', 'pattern', 'z', 'result']:
                return f'blocks.{layer}.attn.hook_{name}'
            elif name in ['pre', 'mid', 'post']:
                return f'blocks.{layer}.mlp.hook_{name}'
            elif name in ['resid_pre', 'resid_mid', 'resid_post']:
                return f'blocks.{layer}.hook_{name}'
            elif name in ['attn_out', 'mlp_out']:
                return f'blocks.{layer}.hook_{name}'

        return name  # Return as-is if no pattern matches


    def get_grad(self, include_ffn: bool = True, layers: Optional[List[str]] = None) -> dict:
        """Extract gradients from model parameters."""
        grad = {}
        for name, param in self.named_parameters():
            if not include_ffn and ('feed_forward' in name or 'norm2' in name):
                continue
            if layers is not None and not any(layer in name for layer in layers):
                continue
            elif param.grad is not None:
                grad[name] = param.grad.clone().detach().flatten()
        return grad

    def to_tokens(
            self,
            text: Union[str, List[str]],
            prepend_bos: Optional[Union[bool, None]] = True,
            padding_side: Optional[Union[Literal["left", "right"], None]] = "right",
            truncate: bool = True,
            move_to_device: bool = True,

            **kwargs
    ) -> torch.Tensor:
        """Convert text to tokens using your TRACE tokenizer."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer set. Use model.set_tokenizer(your_tokenizer)")

        # print("Trying tokenizer's built-in methods...")
        temp_result = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )["input_ids"]



        # Move to device if needed
        if move_to_device:
            # result = result.to(self.cfg.device)
            temp_result = temp_result.to(self.cfg.device)

        return temp_result

    def to_string(self, tokens: torch.Tensor) -> List[str]:
        """Convert tokens back to strings (TransformerLens-style)
        todo: check correctness"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call model.set_tokenizer(tokenizer) first.")

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        texts = []
        for token_seq in tokens:
            token_list = token_seq.tolist()
            text = self.tokenizer.decode(token_list, skip_special_tokens=True)
            texts.append(text)

        return texts

    def to_string_tokens(self, tokens: torch.Tensor) -> List[List[str]]:
        """Convert tokens to list of token strings (TransformerLens-style)"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call model.set_tokenizer(tokenizer) first.")

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        token_strs = []
        for token_seq in tokens:
            token_list = token_seq.tolist()
            str_list = [self.tokenizer.decode([token]) for token in token_list]
            token_strs.append(str_list)

        return token_strs

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def initialize_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def freeze_layer(self, layer_idx: int, freeze: bool = True,
                     freezing_components: Optional[List[str]] = None) -> None:
        """Freeze or unfreeze parameters of a specific layer
        todo: I should handle the hook points here as well to lower memory usage
        """
        if freezing_components:
            components = [(name, param) for name, param in self.named_parameters() if
                          any(comp in name for comp in freezing_components)]
        else:
            components = self.named_parameters()
        for name, param in components:
            # if f'layers.{layer_idx}' in name:
            if f'blocks.{layer_idx}' in name:
                param.requires_grad = not freeze

    def freeze_ffn(self, freeze: bool = True) -> None:
        """Freeze or unfreeze feed-forward network parameters"""
        for name, param in self.named_parameters():
            if 'feed_forward' in name or 'ffn' in name or 'mlp' in name:
                param.requires_grad = not freeze

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of parameters in the model"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_model_size(self) -> str:
        """Get model size description"""
        param_count = self.count_parameters(trainable_only=False)

        if param_count >= 1e9:
            return f"{param_count / 1e9:.1f}B parameters"
        elif param_count >= 1e6:
            return f"{param_count / 1e6:.1f}M parameters"
        elif param_count >= 1e3:
            return f"{param_count / 1e3:.1f}K parameters"
        else:
            return f"{param_count} parameters"

    def get_hook_names(self, filter_fn: Optional[Callable[[str], bool]] = None) -> List[str]:
        """Get list of available hook point names."""
        names = list(self.hook_dict.keys())
        if filter_fn is not None:
            names = [name for name in names if filter_fn(name)]
        return names

    def get_activation(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Get activation from a specific hook point."""
        cache = {}

        def store_activation(activation, hook_name):
            cache[hook_name] = activation.detach()
            return activation

        self.run_with_hooks(x, fwd_hooks=[(name, store_activation)])
        return cache.get(name)

    def cache_all(self, x: torch.Tensor) -> ActivationCache:
        """Cache all activations for input x."""
        _, cache = self.run_with_cache(x)
        return cache

    @property
    def W_E(self) -> torch.Tensor:
        """Token embedding weights [vocab, d_model]."""
        return self.embed.weight

    @property
    def W_U(self) -> torch.Tensor:
        """Unembedding weights [d_model, vocab]."""
        if hasattr(self, 'unembed'):
            return self.unembed.weight.T  # Transpose to match TransformerLens convention
        else:
            raise AttributeError("W_U only available for generative models")

    @property
    def b_U(self) -> torch.Tensor:
        """Unembedding bias [vocab]."""
        if hasattr(self, 'unembed'):
            return self.unembed.bias
        else:
            raise AttributeError("b_U only available for generative models")

    # Weight access for attention and MLP layers
    @property
    def W_Q(self) -> torch.Tensor:
        """Query weights [n_layers, n_heads, d_model, d_head]."""
        weights = []
        for block in self.blocks:
            if hasattr(block, 'attn'):
                w_q = block.attn.W_Q  # Will be implemented in updated components
                weights.append(w_q)
        return torch.stack(weights) if weights else None

    @property
    def W_K(self) -> torch.Tensor:
        """Key weights [n_layers, n_heads, d_model, d_head]."""
        weights = []
        for block in self.blocks:
            if hasattr(block, 'attn'):
                w_k = block.attn.W_K
                weights.append(w_k)
        return torch.stack(weights) if weights else None

    @property
    def W_V(self) -> torch.Tensor:
        """Value weights [n_layers, n_heads, d_model, d_head]."""
        weights = []
        for block in self.blocks:
            if hasattr(block, 'attn'):
                w_v = block.attn.W_V
                weights.append(w_v)
        return torch.stack(weights) if weights else None

    @property
    def W_O(self) -> torch.Tensor:
        """Output weights [n_layers, n_heads, d_head, d_model]."""
        weights = []
        for block in self.blocks:
            if hasattr(block, 'attn'):
                w_o = block.attn.W_O
                weights.append(w_o)
        return torch.stack(weights) if weights else None


class HookedEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, no_fnn: bool = False):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x



class HookedDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, no_fnn: bool = False):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


class HookedDecoderOnlyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, no_fnn: bool = False):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, self_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x
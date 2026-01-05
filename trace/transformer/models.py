import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List, Mapping, Any

from torch.nn.modules.module import T

from .config import TransformerConfig
from .components import PositionalEncoding, Encoder, Decoder


class HooksContext:
    """Context manager for TransformerLens-style hooks"""

    def __init__(self, model, fwd_hooks=None, bwd_hooks=None):
        self.model = model
        self.fwd_hooks = fwd_hooks or []
        self.bwd_hooks = bwd_hooks or []
        self.active_handles = []

    def __enter__(self):
        """Enter context - register hooks"""
        # Debug: Print hook formats to understand what EAP-IG is passing
        print(f"Debug: fwd_hooks type: {type(self.fwd_hooks)}")
        if isinstance(self.fwd_hooks, list) and len(self.fwd_hooks) > 0:
            print(f"Debug: First fwd_hook item: {type(self.fwd_hooks[0])} - {self.fwd_hooks[0]}")

        # Handle both dictionary and list formats
        fwd_hooks = self._normalize_hooks(self.fwd_hooks)
        bwd_hooks = self._normalize_hooks(self.bwd_hooks)

        print(f"Debug: Normalized fwd_hooks: {list(fwd_hooks.keys())}")

        # Register forward hooks
        for hook_name, hook_fn in fwd_hooks.items():
            handle = self._register_hook(hook_name, hook_fn)
            if handle:
                self.active_handles.append(handle)

        # Register backward hooks (if needed)
        for hook_name, hook_fn in bwd_hooks.items():
            handle = self._register_hook(hook_name, hook_fn, is_backward=True)
            if handle:
                self.active_handles.append(handle)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - remove hooks"""
        for handle in self.active_handles:
            try:
                handle.remove()
            except:
                pass
        self.active_handles.clear()

    def _normalize_hooks(self, hooks):
        """Convert hooks to dictionary format"""
        if hooks is None:
            return {}
        elif isinstance(hooks, dict):
            return hooks
        elif isinstance(hooks, list):
            # Convert list of tuples/objects to dictionary
            hook_dict = {}
            for i, item in enumerate(hooks):
                try:
                    if isinstance(item, tuple) and len(item) == 2:
                        # Format: [(hook_name, hook_fn), ...]
                        hook_name, hook_fn = item
                        hook_dict[hook_name] = hook_fn
                    elif isinstance(item, tuple) and len(item) >= 2:
                        # Format with more elements: (hook_name, hook_fn, ...)
                        hook_name, hook_fn = item[0], item[1]
                        hook_dict[hook_name] = hook_fn
                    elif hasattr(item, 'name') and hasattr(item, 'hook'):
                        # Format: [HookObject(name=..., hook=...), ...]
                        hook_dict[item.name] = item.hook
                    elif hasattr(item, 'name') and callable(item):
                        # Format: [callable_with_name_attr, ...]
                        hook_dict[item.name] = item
                    elif hasattr(item, '__dict__'):
                        # Try to extract from object attributes
                        obj_dict = item.__dict__
                        name = obj_dict.get('name', obj_dict.get('hook_name', f'hook_{i}'))
                        hook_fn = obj_dict.get('hook', obj_dict.get('hook_fn', obj_dict.get('fn', item)))
                        if callable(hook_fn):
                            hook_dict[name] = hook_fn
                    elif callable(item):
                        # Just a callable function - use index as name
                        hook_dict[f'hook_{i}'] = item
                    else:
                        print(f"Warning: Unknown hook format at index {i}: {type(item)} - {item}")
                        # Try to use the item directly if it looks like a function
                        if callable(item):
                            hook_dict[f'hook_{i}'] = item
                except Exception as e:
                    print(f"Error processing hook at index {i}: {e}")
                    continue

            print(f"Debug: Converted {len(hooks)} hooks to {len(hook_dict)} dictionary entries")
            return hook_dict
        else:
            print(f"Warning: Unknown hooks format: {type(hooks)}")
            return {}

    def _register_hook(self, hook_name, hook_fn, is_backward=False):
        """Register a hook on the appropriate module"""
        try:
            # Map hook names to actual modules
            module = self._get_module_for_hook(hook_name)
            if module is None:
                print(f"Warning: Could not find module for hook '{hook_name}' - trying fallback")
                # Try some common fallback patterns
                module = self._try_fallback_module(hook_name)

            if module is None:
                print(f"Warning: No module found for hook '{hook_name}' - skipping")
                return None

            print(f"Debug: Registering hook '{hook_name}' on {type(module).__name__}")

            if is_backward:
                handle = module.register_backward_hook(hook_fn)
            else:
                handle = module.register_forward_hook(hook_fn)

            return handle

        except Exception as e:
            print(f"Warning: Failed to register hook '{hook_name}': {e}")
            return None

    def _try_fallback_module(self, hook_name):
        """Try fallback strategies to find a module for the hook"""
        try:
            # If it's just a number, try to map to a layer
            if hook_name.isdigit():
                layer_idx = int(hook_name)
                if self.model.config.model_type == "decoder_only":
                    if layer_idx < len(self.model.decoder.layers):
                        return self.model.decoder.layers[layer_idx]
                elif self.model.config.model_type == "encoder_only":
                    if layer_idx < len(self.model.encoder.layers):
                        return self.model.encoder.layers[layer_idx]

            # Try direct attribute access on the model
            parts = hook_name.split('.')
            obj = self.model
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return None

            if hasattr(obj, 'register_forward_hook'):
                return obj

        except:
            pass

        return None

    def _get_module_for_hook(self, hook_name):
        """Map hook name to actual module"""
        # Parse hook names like "blocks.0.attn.hook_result"
        parts = hook_name.split('.')

        try:
            if parts[0] == "blocks" and len(parts) >= 2:
                layer_idx = int(parts[1])

                # Get the appropriate layer
                if self.model.config.model_type == "decoder_only":
                    if layer_idx < len(self.model.decoder.layers):
                        layer = self.model.decoder.layers[layer_idx]
                    else:
                        return None
                elif self.model.config.model_type == "encoder_only":
                    if layer_idx < len(self.model.encoder.layers):
                        layer = self.model.encoder.layers[layer_idx]
                    else:
                        return None
                else:
                    return None

                # Get the specific component
                if len(parts) >= 3:
                    if parts[2] == "attn":
                        if 'hook_q' in parts[3:]:
                            return layer.attn.q_linear
                        elif 'hook_k' in parts[3:]:
                            return layer.attn.k_linear
                        elif 'hook_v' in parts[3:]:
                            return layer.attn.v_linear
                        elif 'hook_attn_scores' in parts[3:]:
                            return layer.attn.output_linear
                        return layer.attn
                    elif parts[2] == "mlp":
                        return layer.feed_forward
                    else:
                        return layer
                else:
                    return layer

            # Handle other hook patterns as needed
            return None

        except (ValueError, IndexError, AttributeError):
            return None

class Transformer(nn.Module):
    """
    Flexible transformer model supporting encoder-only, decoder-only, and encoder-decoder configurations.
    """

    def __init__(self, config: TransformerConfig):
        """
        Initialize transformer model from configuration.

        Args:
            config: TransformerConfig object containing all model parameters
        """
        super().__init__()

        # Store configuration
        self.cfg = config
        self.model_type = config.model_type
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.pad_idx = config.pad_idx
        self.n_heads = config.n_heads
        if config.model_type == 'encoder_only':
            self.n_layers = config.num_encoder_layers
            self.cfg.n_layers = config.num_encoder_layers
        elif config.model_type == 'decoder_only':
            self.n_layers = config.num_decoder_layers
            self.cfg.n_layers = config.num_decoder_layers
        elif config.model_type == 'encoder_decoder':
            self.n_layers = config.num_encoder_layers + config.num_decoder_layers
            self.cfg.n_layers = config.num_encoder_layers + config.num_decoder_layers
        self.d_ff = config.d_ff if config.d_ff is not None else config.d_model * 4
        self._training = True
        # Input embedding and positional encoding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        self.positional_encoding = PositionalEncoding(config.d_model, config.n_ctx, config.dropout)
        self._setup_hooks_system()
        self._setup_tokenizer_compatibility()

        # Initialize encoder and/or decoder based on model_type
        if config.model_type in ['encoder_only', 'encoder_decoder']:
            assert config.num_encoder_layers > 0, "Encoder layers must be > 0 for encoder models"
            self.encoder = Encoder(
                config.d_model,
                config.num_encoder_layers,
                config.n_heads,
                config.d_ff,
                config.dropout,
                config.no_fnn
            )
        else:
            self.encoder = None

        if config.model_type in ['decoder_only', 'encoder_decoder']:
            assert config.num_decoder_layers > 0, "Decoder layers must be > 0 for decoder models"
            is_decoder_only = (config.model_type == 'decoder_only')
            self.decoder = Decoder(
                config.d_model,
                config.num_decoder_layers,
                config.n_heads,
                config.d_ff,
                config.dropout,
                decoder_only=is_decoder_only,
                no_fnn=config.no_fnn
            )
        else:
            self.decoder = None

        # Output projection for decoder-based models
        if config.model_type in ['decoder_only', 'encoder_decoder']:
            self.output_projection = nn.Linear(config.d_model, config.vocab_size)

    @classmethod
    def from_config(cls, config: TransformerConfig) -> 'Transformer':
        """
        Creating transformer model from configuration.

        Args:
            config: TransformerConfig object

        Returns:
            Initialized Transformer model
        """
        return cls(config) # return cls(**config.__dict__) #return cls(**config.__dict__)

    def generate_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate padding mask for attention.
        # Source: https://medium.com/@swarms/understanding-masking-in-pytorch-for-attention-mechanisms-e725059fd49f

        Args:
            x: Input tensor with padding tokens

        Returns:
            Padding mask tensor
        """
        return (x == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for decoder self-attention.
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
            self,
            src: Optional[torch.Tensor] = None,
            tgt: Optional[torch.Tensor] = None,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the transformer.

        Args:
            src: Source sequence tensor [batch_size, src_seq_len]
            tgt: Target sequence tensor [batch_size, tgt_seq_len]
            src_mask: Source sequence padding mask
            tgt_mask: Target sequence causal mask
            memory_mask: Cross-attention mask

        Returns:
            Output tensor(s) depending on model type
        """
        if self.model_type == 'encoder_only':
            return self._forward_encoder_only(src, src_mask)
        elif self.model_type == 'decoder_only':
            return self._forward_decoder_only(tgt, tgt_mask)
        elif self.model_type == 'encoder_decoder':
            return self._forward_encoder_decoder(src, tgt, src_mask, tgt_mask, memory_mask)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _forward_encoder_only(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for encoder-only models."""
        assert src is not None, "Source sequence required for encoder-only model"

        # Create padding mask if not provided
        if src_mask is None:
            src_mask = self.generate_padding_mask(src)

        # Embed source sequence
        src_embedded = self.embedding(src)
        src_embedded = self.positional_encoding(src_embedded)

        # Encode
        encoder_output, encoder_attentions = self.encoder(src_embedded, src_mask)

        return encoder_output

    def _forward_decoder_only(
            self,
            tgt: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for decoder-only models."""
        assert tgt is not None, "Target sequence required for decoder-only model"

        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)

        # Embed target sequence
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        # Decode (without encoder output)
        decoder_output, self_attentions, _ = self.decoder(tgt_embedded, None, tgt_mask, None)

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    def _forward_encoder_decoder(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for encoder-decoder models."""
        assert src is not None, "Source sequence required for encoder-decoder model"
        assert tgt is not None, "Target sequence required for encoder-decoder model"

        # Create masks if not provided
        if src_mask is None:
            src_mask = self.generate_padding_mask(src)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        if memory_mask is None:
            memory_mask = self.generate_padding_mask(src)

        # Embed source and target sequences
        src_embedded = self.embedding(src)
        src_embedded = self.positional_encoding(src_embedded)

        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        # Encode
        encoder_output, encoder_attentions = self.encoder(src_embedded, src_mask)

        # Decode
        decoder_output, self_attentions, cross_attentions = self.decoder(
            tgt_embedded, encoder_output, tgt_mask, memory_mask
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    def get_grad(self, include_ffn: bool = True, layers: Optional[List[str]] = None) -> dict:
        """
        Extract gradients from model parameters.

        Args:
            include_ffn: Whether to include feed-forward network gradients
            layers: Optional list of layer names to filter gradients

        Returns:
            Dictionary mapping parameter names to flattened gradients
        """
        grad = {}
        for name, param in self.named_parameters():
            if not include_ffn and ('feed_forward' in name or 'norm2' in name):
                continue
            if layers is not None and not any(layer in name for layer in layers):
                continue
            elif param.grad is not None:
                grad[name] = param.grad.clone().detach().flatten()
        return grad

    def initialize_weights(self):
        """
        Initialize model weights using Xavier uniform initialization.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def freeze_layer(self, layer_idx: int, freeze: bool = True,
                     freezing_components: Optional[List[str]] = None) -> None:
        """
        Freeze or unfreeze parameters of a specific layer in the transformer model.

        Args:
            layer_idx: Index of the layer to freeze/unfreeze
            freeze: Whether to freeze (True) or unfreeze (False) parameters
        """
        if freezing_components:
            components = [(name, param) for name, param in self.named_parameters() if
                          any(comp in name for comp in freezing_components)]
        else:
            components = self.named_parameters()
        for name, param in components:
            if f'layers.{layer_idx}' in name:
                param.requires_grad = not freeze

    def freeze_ffn(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze (all) feed-forward network parameters in the transformer model.
        """
        for name, param in self.named_parameters():
            if 'feed_forward' in name or 'ffn' in name:
                param.requires_grad = not freeze

    def attach_hooks(self, names: Optional[List[str]] = None) -> None:
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
            names = [name for name, _ in self.named_parameters() if 'embedding' not in name]

        for name, param in self.named_parameters():
            if name in names:
                hook = create_hook(name)
                param.register_hook(hook)

    # def remove_hooks(self, names: Optional[List[str]] = None) -> None:
    #     """
    #     Remove gradient hooks from model parameters.
    #
    #     Args:
    #         names: Optional list of parameter names to remove hooks from. If None, removes all hooks
    #     """
    #     if names is None:
    #         names = [name for name, _ in self.named_parameters() if 'embedding' not in name]
    #
    #     for name, param in self.named_parameters():
    #         if name in names:
    #             param.register_hook(None)

    def setup_tracking(self,
                       track_layers: Optional[List[str]] = None,
                       track_last_token: bool = True,
                       track_interval: int = 0,
                       max_history: int = -1,
                       ema_alpha: float = 0.9,
                       track_save_dir: Optional[str] = None):
        """
        Setup tracking for model states during training.

        Args:
            track_layers: List of layer names to track
            track_last_token: Whether to track last token
            track_interval: Interval for tracking
            max_history: Maximum history to keep
            ema_alpha: EMA alpha for tracking
            track_save_dir: Directory to save tracking data
        """
        # Placeholder for tracking functionality
        pass

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in a model.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_model_size(self) -> str:
        """
        Get an approximate describing model size - HuggingFace style.
        """
        param_count = self.count_parameters(trainable_only=False)

        if param_count >= 1e9:
            return f"{param_count / 1e9:.1f}B parameters"
        elif param_count >= 1e6:
            return f"{param_count / 1e6:.1f}M parameters"
        elif param_count >= 1e3:
            return f"{param_count / 1e3:.1f}K parameters"
        else:
            return f"{param_count} parameters"

    # TransformerLens compatibility methods and properties added for convenience with Mechanistic Interpretability tools
    # def _setup_hooks_system(self):
    #     """Set up TransformerLens-style hooks system"""
    #     self.hooks = {}
    #     self._hook_handles = []
    #     self.hook_dict = {}
    #     self.is_caching = False
    #
    #     # Generate hook points
    #     self.hook_points = self._generate_hook_points()
    def _setup_hooks_system(self):
        """Set up TransformerLens-style hooks system"""
        # Store hook registry (for compatibility)
        self._hook_registry = {}
        self._hook_handles = []
        self.hook_dict = {}
        self.is_caching = False

        # Generate hook points
        self.hook_points = self._generate_hook_points()

    def hooks(self, fwd_hooks=None, bwd_hooks=None): # todo: would Lamda functions work here?
        """Create a hooks context manager (TransformerLens-style)"""
        return HooksContext(self, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks)

    # def _generate_hook_points(self):
    #     """Generate available hook points"""
    #     hook_points = []
    #
    #     if self.cfg.model_type == "decoder_only":
    #         for i in range(self.cfg.num_decoder_layers):
    #             hook_points.extend([ # todo: check names and if any is missing (for sure the residual streams)
    #
    #                 f"blocks.{i}.attn",
    #                 f"blocks.{i}.attn.hook_result",
    #                 f"blocks.{i}.mlp",
    #                 f"blocks.{i}.mlp.hook_post",
    #             ])
    #     elif self.cfg.model_type == "encoder_only":
    #         for i in range(self.cfg.num_encoder_layers):
    #             hook_points.extend([
    #                 f"blocks.{i}.attn",
    #                 f"blocks.{i}.attn.hook_result",
    #                 f"blocks.{i}.mlp",
    #                 f"blocks.{i}.mlp.hook_post",
    #             ])
    #     elif self.cfg.model_type == "encoder_decoder":
    #         for i in range(self.cfg.num_encoder_layers):
    #             hook_points.extend([
    #                 f"encoder.blocks.{i}.attn",
    #                 f"encoder.blocks.{i}.attn.hook_result",
    #                 f"encoder.blocks.{i}.mlp",
    #                 f"encoder.blocks.{i}.mlp.hook_post",
    #             ])
    #         for i in range(self.cfg.num_decoder_layers):
    #             hook_points.extend([
    #                 f"decoder.blocks.{i}.self_attn",
    #                 f"decoder.blocks.{i}.self_attn.hook_result",
    #                 f"decoder.blocks.{i}.cross_attn",
    #                 f"decoder.blocks.{i}.cross_attn.hook_result",
    #                 f"decoder.blocks.{i}.mlp",
    #                 f"decoder.blocks.{i}.mlp.hook_post",
    #             ])
    #
    #     return hook_points
    def _generate_hook_points(self):
        """Generate available hook points"""
        hook_points = []

        if self.cfg.model_type == "decoder_only":
            for i in range(self.cfg.num_decoder_layers):
                hook_points.extend([
                    f"blocks.{i}.attn",
                    f"blocks.{i}.attn.hook_result",
                    f"blocks.{i}.mlp",
                    f"blocks.{i}.mlp.hook_post",
                ])
        elif self.cfg.model_type == "encoder_only":
            for i in range(self.cfg.num_encoder_layers):
                hook_points.extend([
                    f"blocks.{i}.attn",
                    f"blocks.{i}.attn.hook_result",
                    f"blocks.{i}.mlp",
                    f"blocks.{i}.mlp.hook_post",
                ])

        return hook_points
    def _setup_tokenizer_compatibility(self):
        """Set up tokenizer with required attributes"""
        # This will be set when tokenizer is attached
        self.tokenizer = None


    def add_hook(self, name, hook_fn, prepend=False):
        """Add a hook to a specific component"""
        if name not in self._hook_registry:
            self._hook_registry[name] = []

        if prepend:
            self._hook_registry[name].insert(0, hook_fn)
        else:
            self._hook_registry[name].append(hook_fn)

        return len(self._hook_registry[name]) - 1

    def remove_hooks(self, including_permanent=True):
        """Remove all hooks"""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except:
                pass

        self._hook_registry.clear()
        self._hook_handles.clear()
        self.hook_dict.clear()
        self.is_caching = False

    def reset_hooks(self, clear_contexts=True, direction="both"):
        """Reset hooks system"""
        self.remove_hooks()
    def to_tokens(self, inputs, prepend_bos=True, padding_side='right', truncate=None, max_length=None):
        """Convert inputs to tokens (TransformerLens-style)
        Converts a string to a tensor of tokens.
        there are multi versions todo: check correctness
        -> Int[Tensor, 'batch pos']
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call model.set_tokenizer(tokenizer) first.")

        # Handle different input types
        if isinstance(inputs, str):
            inputs = [inputs]
        elif isinstance(inputs, torch.Tensor):
            return inputs

        tokens_list = []
        for text in inputs:
            if isinstance(text, str):
                encoded = self.tokenizer.encode(text)
                tokens = torch.tensor(encoded, dtype=torch.long)
            else:
                tokens = torch.tensor(text, dtype=torch.long)
            tokens_list.append(tokens)

        # Pad sequences
        if len(tokens_list) > 1:
            max_len = max(len(tokens) for tokens in tokens_list)
            if max_length is not None:
                max_len = min(max_len, max_length)

            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
            if pad_token_id is None:
                pad_token_id = 0

            padded_tokens = []
            for tokens in tokens_list:
                if truncate and len(tokens) > max_len:
                    tokens = tokens[:max_len]

                if len(tokens) < max_len:
                    pad_length = max_len - len(tokens)
                    if padding_side == 'right':
                        pad_tokens = torch.full((pad_length,), pad_token_id, dtype=torch.long)
                        tokens = torch.cat([tokens, pad_tokens])
                    else:
                        pad_tokens = torch.full((pad_length,), pad_token_id, dtype=torch.long)
                        tokens = torch.cat([pad_tokens, tokens])

                padded_tokens.append(tokens)

            result = torch.stack(padded_tokens)
        else:
            result = tokens_list[0].unsqueeze(0)

        # Add BOS token if requested
        if prepend_bos:
            bos_token_id = getattr(self.tokenizer, 'bos_token_id', 1)
            if bos_token_id is None:
                bos_token_id = 1

            batch_size, seq_len = result.shape
            bos_tokens = torch.full((batch_size, 1), bos_token_id, dtype=torch.long)
            result = torch.cat([bos_tokens, result], dim=1)

        return result

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer and ensure it has required attributes"""
        self.tokenizer = tokenizer

        # Ensure tokenizer has required attributes
        if not hasattr(tokenizer, 'bos_token_id') or tokenizer.bos_token_id is None or tokenizer.bos_token is None:
            tokenizer.bos_token_id = 1
        if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None or tokenizer.eos_token is None:
            tokenizer.eos_token_id = 2
        if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None or tokenizer.pad_token is None:
            tokenizer.pad_token_id = 0
        if not hasattr(tokenizer, 'unk_token_id') or tokenizer.unk_token_id is None or tokenizer.unk_token is None:
            tokenizer.unk_token_id = 3


    def run_with_cache(self, tokens, **kwargs):
        """Run forward pass and return output with activation cache"""
        cache = {}

        def caching_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    cache[name] = output[0].detach()
                else:
                    cache[name] = output.detach()

            return hook_fn

        # Create hooks dict for caching
        fwd_hooks = {}

        if self.cfg.model_type == "decoder_only":
            for i in range(self.cfg.num_decoder_layers):
                fwd_hooks[f"blocks.{i}.attn.hook_result"] = caching_hook(f"blocks.{i}.attn.hook_result")
                fwd_hooks[f"blocks.{i}.mlp.hook_post"] = caching_hook(f"blocks.{i}.mlp.hook_post")
        elif self.cfg.model_type == "encoder_only":
            for i in range(self.cfg.num_encoder_layers):
                fwd_hooks[f"blocks.{i}.attn.hook_result"] = caching_hook(f"blocks.{i}.attn.hook_result")
                fwd_hooks[f"blocks.{i}.mlp.hook_post"] = caching_hook(f"blocks.{i}.mlp.hook_post")

        # Use hooks context manager
        with self.hooks(fwd_hooks=fwd_hooks):
            # Forward pass
            if self.cfg.model_type == "decoder_only":
                logits = self.forward(tgt=tokens)
            elif self.cfg.model_type == "encoder_only":
                logits = self.forward(src=tokens)
            else:
                logits = self.forward(src=tokens, tgt=tokens)

        return logits, cache

    # Add property for TransformerLens compatibility
    @property
    def model(self):
        """Return self for compatibility"""
        return self

    @property
    def is_categorical(self):
        """Return True for language modeling compatibility"""
        return True

    def training(self):
        """Return training mode"""
        self._training = True
        return self

    def eval(self: T) -> T:
        """Set the module in evaluation mode."""
        self._training = False
        return self






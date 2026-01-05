from dataclasses import dataclass
from typing import Optional
from .utils import get_device
import torch

'''
Inspired by https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py
and https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/HookedTransformerConfig.py 
but simplified for our use case. 
'''
@dataclass
class TransformerConfig:
    """
    Configuration class for transformer models.

    This dataclass contains all hyperparameters and configuration options
    for creating transformer models, following the reference style pattern.
    """

    # Model architecture
    model_type: str = "decoder_only"  # 'encoder_only', 'decoder_only', 'encoder_decoder'
    model_name : str = "custom_transformer"
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    num_encoder_layers: int = 0
    num_decoder_layers: int = 6
    n_layers: Optional[int] = num_encoder_layers + num_decoder_layers  # made to be compatible with TransformerLens
    d_ff: int = 2048
    d_mlp: Optional[int] = d_ff  # If None, d_ff is used
    dtype: torch.dtype = torch.float32
    d_heads : Optional[int] = d_model // n_heads  # Dimension per head, if None d_model // n_heads is used
    # Training parameters
    n_ctx: int = 64 # max_seq_length: int = 5000
    d_vocab : Optional[int] = 32000  # If None, vocab size is used
    dropout: float = 0.1
    pad_idx: int = 0

    # Special options
    no_fnn: bool = False  # Disable feed-forward networks
    freeze_ffn: bool = False  # Freeze feed-forward parameters
    device: torch.device = get_device() # todo: https://docs.pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#create-model-and-dataparallel
    use_attn_result: bool = False  # to compute each head output separately # todo: implement
    use_attn_in: bool = False  # to use attn_in hook get the input to the attention layer # todo: implement
    use_split_qkv_input: bool = False  # to compute each head input separately # todo: implement
    use_hook_mlp_in: bool = True  # to use mlp_in hook get the input to the MLP layer # todo: implement
    n_key_value_heads: Optional[int] = 0  # "The number of groups of heads that use the same key and value matrix. Only for models that use Grouped Query Attention."
    ungroup_grouped_query_attention: bool = True  # "If True, undoes the grouping of heads done by Grouped Query Attention. Only for models that use Grouped Query Attention."
    use_normalization_before_and_after: bool = False  # Whether to use pre-norm and post-norm (True) or just pre-norm (False)
    parallel_attn_mlp: bool = False  # Whether to use parallel attention and MLP (True) or sequential (False)
    normalization_type: str = "LN"  # 'layer_norm' or 'rms_norm'
    act_fn: str = "relu"  # 'relu', 'gelu', 'silu', 'gelu_new', 'swish', 'tanh', 'sigmoid'
    eps: float = 1e-5  # Epsilon for normalization layers


    def __post_init__(self):
        """Validate configuration parameters."""
        if self.model_type not in ['encoder_only', 'decoder_only', 'encoder_decoder']:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        if self.model_type in ['encoder_only', 'encoder_decoder'] and self.num_encoder_layers <= 0:
            raise ValueError("Encoder layers must be > 0 for encoder models")

        if self.model_type in ['decoder_only', 'encoder_decoder'] and self.num_decoder_layers <= 0:
            raise ValueError("Decoder layers must be > 0 for decoder models")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.d_ff is None:
            self.d_ff = self.d_model * 4

    @classmethod
    def encoder_only(cls, **kwargs) -> 'TransformerConfig':
        """Create configuration for encoder-only model."""
        defaults = {
            'model_type': 'encoder_only',
            'num_encoder_layers': 6,
            'num_decoder_layers': 0,
            'n_layers': 6
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def decoder_only(cls, **kwargs) -> 'TransformerConfig':
        """Create configuration for decoder-only model."""
        defaults = {
            'model_type': 'decoder_only',
            'num_encoder_layers': 0,
            'num_decoder_layers': 6,
            'n_layers': 6
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def encoder_decoder(cls, **kwargs) -> 'TransformerConfig':
        """Create configuration for encoder-decoder model."""
        defaults = {
            'model_type': 'encoder_decoder',
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'n_layers': 12
        }
        defaults.update(kwargs)
        return cls(**defaults)
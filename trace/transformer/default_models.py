from .models import Transformer
from .config import TransformerConfig


class TransformerDefault:
    """
    Class for creating transformer models with common configurations.
    """

    @staticmethod
    def create_encoder_only_transformer(
            vocab_size: int,
            d_model: int = 512,
            n_heads: int = 8,
            num_layers: int = 6,
            n_ctx: int = 512,
            dropout: float = 0.01,
            pad_idx: int = 0,
            d_ff: int = 1024,
            no_fnn: bool = False
    ) -> Transformer:
        """
        Create an encoder-only transformer model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of encoder layers
            n_ctx: Maximum sequence length
            dropout: Dropout rate
            pad_idx: Padding token index
            d_ff: Feed-forward dimension
            no_fnn: Whether to disable feed-forward networks

        Returns:
            Configured encoder-only transformer model
        """
        config = TransformerConfig(
            model_type='encoder_only',
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            n_layers=num_layers,
            d_ff=d_ff,
            n_ctx=n_ctx,
            dropout=dropout,
            pad_idx=pad_idx,
            no_fnn=no_fnn
        )
        return Transformer.from_config(config)

    @staticmethod
    def create_decoder_only_transformer(
            vocab_size: int,
            d_model: int = 512,
            n_heads: int = 8,
            num_layers: int = 6,
            n_ctx: int = 512,
            dropout: float = 0.01,
            pad_idx: int = 0,
            d_ff: int = 1024,
            no_fnn: bool = False
    ) -> Transformer:
        """
        Create a decoder-only transformer model (GPT-style).

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of decoder layers
            n_ctx: Maximum sequence length
            dropout: Dropout rate
            pad_idx: Padding token index
            d_ff: Feed-forward dimension
            no_fnn: Whether to disable feed-forward networks

        Returns:
            Configured decoder-only transformer model
        """
        config = TransformerConfig(
            model_type='decoder_only',
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=0,
            num_decoder_layers=num_layers,
            n_layers=num_layers,
            d_ff=d_ff,
            n_ctx=n_ctx,
            dropout=dropout,
            pad_idx=pad_idx,
            no_fnn=no_fnn
        )
        return Transformer.from_config(config)

    @staticmethod
    def create_encoder_decoder_transformer(
            vocab_size: int,
            d_model: int = 512,
            n_heads: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            n_ctx: int = 512,
            dropout: float = 0.01,
            pad_idx: int = 0,
            d_ff: int = 1024,
            no_fnn: bool = False
    ) -> Transformer:
        """
        Create an encoder-decoder transformer model (T5-style).

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            n_ctx: Maximum sequence length
            dropout: Dropout rate
            pad_idx: Padding token index
            d_ff: Feed-forward dimension
            no_fnn: Whether to disable feed-forward networks

        Returns:
            Configured encoder-decoder transformer model
        """
        config = TransformerConfig(
            model_type='encoder_decoder',
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            n_layers=num_encoder_layers + num_decoder_layers,
            d_ff=d_ff,
            n_ctx=n_ctx,
            dropout=dropout,
            pad_idx=pad_idx,
            no_fnn=no_fnn
        )
        return Transformer.from_config(config)

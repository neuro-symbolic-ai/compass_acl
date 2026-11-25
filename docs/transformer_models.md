# Transformer Models

COMPASS includes a transformer implementation for encoder-only, decoder-only and encoder-decoder models. All three are configured with `TransformerConfig`.

The analysis modules work with these models and with TransformerLens `HookedTransformer` models. The model type is detected automatically.

## Configuration

```python
import torch
from trace.transformer import Transformer, TransformerConfig

config = TransformerConfig(
    model_type="decoder_only",   # "encoder_only", "decoder_only" or "encoder_decoder"
    vocab_size=30522,
    d_model=768,                 # model dimension
    n_heads=12,                  # attention heads
    num_encoder_layers=0,        # encoder layers, if applicable
    num_decoder_layers=6,        # decoder layers, if applicable
    d_ff=3072,                   # feed-forward dimension
    n_ctx=512,                   # maximum sequence length
    dropout=0.1,
    device="cpu",
)
```

## Decoder-only models

For autoregressive language modelling:

```python
config = TransformerConfig(
    model_type="decoder_only",
    vocab_size=2000,
    d_model=768,
    n_heads=12,
    num_decoder_layers=12,
    d_ff=3072,
    n_ctx=64,
    dropout=0.1,
    device="cpu",
)
decoder_model = Transformer.from_config(config)

input_ids = torch.randint(0, 2000, (2, 32))   # [batch_size, seq_len]
logits = decoder_model(tgt=input_ids)
print(logits.shape)                           # [2, 32, 2000]
```

## Encoder-only models

For classification and representation learning:

```python
config = TransformerConfig(
    model_type="encoder_only",
    vocab_size=30522,
    d_model=768,
    n_heads=12,
    num_encoder_layers=6,
    d_ff=3072,
    n_ctx=512,
    dropout=0.1,
)
encoder_model = Transformer.from_config(config)

input_ids = torch.randint(0, 30522, (2, 20))  # [batch_size, seq_len]
hidden_states = encoder_model(src=input_ids)
print(hidden_states.shape)                    # [2, 20, 768]
```

## Encoder-decoder models

For sequence-to-sequence tasks:

```python
config = TransformerConfig(
    model_type="encoder_decoder",
    vocab_size=32128,
    d_model=512,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    n_ctx=512,
    dropout=0.1,
)
encoder_decoder_model = Transformer.from_config(config)

src_ids = torch.randint(0, 32128, (2, 15))    # source sequence
tgt_ids = torch.randint(0, 32128, (2, 10))    # target sequence
logits = encoder_decoder_model(src=src_ids, tgt=tgt_ids)
print(logits.shape)                           # [2, 10, 32128]
```

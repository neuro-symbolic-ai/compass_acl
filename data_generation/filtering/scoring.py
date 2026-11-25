"""Next-token scoring of prompts with a causal language model."""

from typing import Dict, List, Sequence

import torch


def load_model(name: str, device: str, half_precision_on_cuda: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if (half_precision_on_cuda and "cuda" in device) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device).eval()
    return tok, model


@torch.no_grad()
def next_token_probs(tok, model, prompts: Sequence[str], candidate_ids: Sequence[int],
                     device: str, batch_size: int = 64) -> List[Dict[int, float]]:
    """Probability of each candidate token after each prompt.

    The logits are read at each prompt's own last token (from the attention
    mask), so the result does not depend on the tokenizer's padding side.
    """
    out = []
    cand = torch.tensor(list(candidate_ids), device=device)
    for i in range(0, len(prompts), batch_size):
        enc = tok(list(prompts[i:i + batch_size]), return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits        # [B, T, V]
        if tok.padding_side == "left":
            last = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=device)
        else:
            last = attn.sum(dim=1) - 1                                          # [B]
        next_logits = logits[torch.arange(input_ids.size(0), device=device), last]  # [B, V]
        probs = torch.softmax(next_logits.float(), dim=-1).index_select(1, cand)    # [B, K]
        for row in probs.tolist():
            out.append({int(c): float(p) for c, p in zip(candidate_ids, row)})
    return out


def space_token_id(tok, word: str) -> int:
    ids = tok.encode(" " + word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"'{word}' is {len(ids)} tokens with leading space")
    return ids[0]


def prefers_target(probs: Dict[int, float], target_id: int, corrupt_ids: Sequence[int]) -> bool:
    """True if the target outranks every corrupt head."""
    p_t = probs.get(target_id, 0.0)
    return all(p_t > probs.get(c, 0.0) for c in corrupt_ids)

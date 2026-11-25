"""Probability-difference metric for the greater-than task, in the EAP metric signature."""

from typing import Sequence

import torch


def make_prob_diff(year_token_ids: Sequence[int]):
    """Return prob_diff(logits, clean_logits, input_length, labels, mean, loss).

    For each example, sums the probability of years above the label minus the
    probability of years at or below it, read at the prompt's last token.
    """
    year_ids = torch.tensor(list(year_token_ids))

    def prob_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor,
                  labels: torch.Tensor, mean: bool = True, loss: bool = False):
        idx = torch.arange(logits.size(0), device=logits.device)
        last = logits[idx, input_length - 1]
        probs = torch.softmax(last, dim=-1)[:, year_ids.to(logits.device)]
        results = torch.stack([p[int(y) + 1:].sum() - p[:int(y) + 1].sum() for p, y in zip(probs, labels)])
        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    return prob_diff

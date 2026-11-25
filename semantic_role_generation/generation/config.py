from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GenerationConfig:
    """Configuration for generating semantic-role next-token datasets."""

    # Where to write <out_dir>/<Role>/{prompts.csv, interventions.csv, dataset.jsonl}
    out_dir: str = "./data/semroles"
    n_per_role: int = 1500
    seed: int = 42
    # Roles to generate; None uses every role in the lexicon.
    roles: Optional[List[str]] = None
    # Give up on a role after n_per_role * patience attempts for computation check
    patience: int = 30

    # Surface variation
    adverb_prob: float = 0.3
    adjective_prob: float = 0.25
    # The "hard" corruption adds a preface clause and may be longer than clean.
    allow_hard_longer: bool = False
    # Number of same-role head swaps written to interventions.csv per example.
    n_interventions: int = 2

    # Corruption rules
    # Reject pairs whose corrupted prompt is identical to the clean prompt.
    # The original generator accepted them (at -> at, "for the" -> "for the").
    skip_identical_corruption: bool = True
    # Accept a same-role foil when no cross-role scaffold of equal length exists.
    allow_same_role_fallback: bool = True

    def __post_init__(self):
        if self.n_per_role <= 0:
            raise ValueError("n_per_role must be positive")
        if not 0 <= self.adverb_prob <= 1 or not 0 <= self.adjective_prob <= 1:
            raise ValueError("adverb_prob and adjective_prob must be in [0, 1]")

    @classmethod
    def default(cls, **kwargs) -> 'GenerationConfig':
        """Paper setting with identical clean/corrupted pairs rejected."""
        defaults = {'n_per_role': 1500, 'seed': 42}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def legacy(cls, **kwargs) -> 'GenerationConfig':
        """Reproduce some of original main.py behaviour, including identical pairs."""
        defaults = {'n_per_role': 1500, 'seed': 1337, 'skip_identical_corruption': False}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def small(cls, **kwargs) -> 'GenerationConfig':
        """Quick smoke-test setting."""
        defaults = {'n_per_role': 50, 'patience': 10}
        defaults.update(kwargs)
        return cls(**defaults)

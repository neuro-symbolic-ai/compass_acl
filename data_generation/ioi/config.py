from dataclasses import dataclass
from typing import Optional


@dataclass
class IOIConfig:
    """Configuration for indirect-object-identification (IOI) minimal pairs."""

    out_path: str = "./data/ioi/ioi_pairs.csv"
    n_examples: int = 1000
    seed: int = 123
    # Template family: BABA, ABBA, mixed, ABC, BAC or "ABC mixed".
    prompt_type: str = "BABA"
    # Use only the first n templates of the family; None uses all.
    n_templates: Optional[int] = None
    # Also emit the A/B-swapped twin of each prompt.
    symmetric: bool = False
    # Corruption of the giver in the second clause:
    #   corrupted      -> a third name (the ABC corruption of Wang et al.)
    #   corrupted_hard -> the indirect object itself, so the cue flips to S
    hard_corruption: bool = True

    def __post_init__(self):
        from .templates import TEMPLATE_SETS
        if self.prompt_type not in TEMPLATE_SETS:
            raise ValueError(f"prompt_type must be one of {list(TEMPLATE_SETS)}, got {self.prompt_type!r}")
        if self.n_examples <= 0:
            raise ValueError("n_examples must be positive")

    @classmethod
    def default(cls, **kwargs) -> 'IOIConfig':
        defaults = {'prompt_type': 'BABA'}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def abc(cls, **kwargs) -> 'IOIConfig':
        """Three-name templates."""
        defaults = {'prompt_type': 'ABC mixed'}
        defaults.update(kwargs)
        return cls(**defaults)

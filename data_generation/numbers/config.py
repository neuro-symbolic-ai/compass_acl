from dataclasses import dataclass, field
from typing import List


@dataclass
class GreaterThanConfig:
    """Configuration for the greater-than task (following EAP-IG Hanna et al., 2023).

    Prompts read "The <noun> lasted from the year XXYY to the year XX"; the
    model should put more probability on years above YY than at or below it.
    """

    out_path: str = "./data/numbers/greater_than.csv"
    n_examples: int = 1000
    seed: int = 0
    # Centuries XX to sample from (11 -> years 11YY).
    centuries: List[int] = field(default_factory=lambda: list(range(11, 18)))
    # Inclusive range of the two-digit start year YY.
    year_min: int = 2
    year_max: int = 98
    # The corrupted prompt replaces YY with this value, so every later year is
    # "greater" and the task carries no signal.
    corrupted_year: int = 1
    template: str = "The {noun} lasted from the year {century:02d}{year:02d} to the year {century:02d}"

    def __post_init__(self):
        if not 0 <= self.year_min <= self.year_max <= 99:
            raise ValueError("need 0 <= year_min <= year_max <= 99")
        if self.n_examples <= 0:
            raise ValueError("n_examples must be positive")

    @classmethod
    def default(cls, **kwargs) -> 'GreaterThanConfig':
        """Matches greater_than_data.csv from the EAP-IG release."""
        defaults = {}
        defaults.update(kwargs)
        return cls(**defaults)

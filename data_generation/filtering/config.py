from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FilterConfig:
    """Configuration for model-based filtering of generated minimal pairs."""

    model_name: str = "EleutherAI/pythia-1b"
    device: str = "cpu"
    batch_size: int = 64
    seed: int = 42
    # Prompt columns to score; each must exist in prompts.csv.
    contexts: List[str] = field(default_factory=lambda: ["clean", "corrupted", "corrupted_hard"])
    # Splits to write: name -> required win pattern over `contexts` (True = model
    # prefers the target over every corrupt head in that context).
    splits: dict = field(default_factory=lambda: {
        "both_correct": [True, True, True],
        "clean_only_correct": [True, False, False],
    })
    # Write <out_dir>/diagnostics/<Role>_scored.csv and all_scored.csv
    save_diagnostics: bool = True
    # Also write one folder per target role inside each split.
    split_by_role: bool = True
    # Use torch.float16 on CUDA devices.
    half_precision_on_cuda: bool = True

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for name, pattern in self.splits.items():
            if len(pattern) != len(self.contexts):
                raise ValueError(f"split {name!r} needs one flag per context ({len(self.contexts)})")

    @classmethod
    def default(cls, **kwargs) -> 'FilterConfig':
        defaults = {}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def clean_only(cls, **kwargs) -> 'FilterConfig':
        """Score only the clean context; keeps every pair the model gets right."""
        defaults = {'contexts': ["clean"], 'splits': {"clean_correct": [True]}}
        defaults.update(kwargs)
        return cls(**defaults)

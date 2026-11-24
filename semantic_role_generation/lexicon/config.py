from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LexiconConfig:
    """Configuration for building a tokenizer-filtered lexicon."""

    tokenizer_name: str = "EleutherAI/pythia-1b"
    # Only keep words that encode as one token with a leading space.
    single_token_only: bool = True
    # Time scaffold family: "1token" (at / on, used in the paper) or "2token"
    # (at the / during the, lets Time swap with other roles).
    scaffold_markers: str = "1token"
    # Restrict to a subset of roles; None keeps all ten.
    roles: Optional[List[str]] = None
    # Names for IOI are filtered as whole words ("Sara"), not role heads.
    include_names: bool = True

    #todo add the number comparision

    def __post_init__(self):
        if self.scaffold_markers not in ("1token", "2token"):
            raise ValueError(f"scaffold_markers must be '1token' or '2token', got {self.scaffold_markers!r}")

    @classmethod
    def default(cls, **kwargs) -> 'LexiconConfig':
        """Paper setting: Pythia tokenizer, one-token Time scaffolds."""
        defaults = {'scaffold_markers': '1token'}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def updated_time(cls, **kwargs) -> 'LexiconConfig':
        """Two-token Time scaffolds so Time can swap with Location and Goal."""
        defaults = {'scaffold_markers': '2token'}
        defaults.update(kwargs)
        return cls(**defaults)

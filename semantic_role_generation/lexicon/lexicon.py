"""Tokenizer-aware lexicon shared by the semantic-role, IOI generators and number comparision."""

from typing import Dict, List, Optional

from .config import LexiconConfig
from . import roles as R
from . import words as W


class Lexicon:
    """Word inventories filtered to single tokens for one tokenizer.

    Example:
        lex = Lexicon(LexiconConfig(tokenizer_name="EleutherAI/pythia-14m"))
        lex.heads["Goal"]            # single-token Goal targets
        lex.scaffolds("Time")        # ["at", "on"]
        lex.token_id("student")      # id of " student"
        lex.names                    # single-token first names for IOI
    """

    def __init__(self, config: Optional[LexiconConfig] = None, tokenizer=None):
        self.config = config or LexiconConfig.default()
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.tokenizer = tokenizer

        self.roles: List[str] = list(self.config.roles or R.ROLES)
        unknown = [r for r in self.roles if r not in R.ROLES]
        if unknown:
            raise ValueError(f"Unknown roles: {unknown}")

        self.agents = self._filter(W.AGENTS)
        self.themes = self._filter(W.THEMES)
        self.heads: Dict[str, List[str]] = {
            role: self._filter(self._raw_heads(role)) for role in self.roles
        }
        self.names = self._filter(W.NAMES) if self.config.include_names else []
        self.ioi_places = self._filter(W.IOI_PLACES)
        self.ioi_objects = self._filter(W.IOI_OBJECTS)

        self.adverbs, self.adj_agent, self.adj_theme = W.ADVERBS, W.ADJ_AGENT, W.ADJ_THEME
        self.prefaces = W.PREFACES

        empty = [k for k, v in self.heads.items() if not v]
        if empty:
            raise ValueError(f"No single-token heads left for roles {empty} "
                             f"with tokenizer {self.config.tokenizer_name}")

    # ---- tokenization ----
    def ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def token_length(self, text: str) -> int:
        return len(self.ids(text))

    def is_single_token(self, word: str) -> bool:
        return len(self.ids(" " + word)) == 1

    def token_id(self, word: str) -> int:
        ids = self.ids(" " + word)
        if len(ids) != 1:
            raise ValueError(f"Not a single token with leading space: {word!r} -> {ids}")
        return ids[0]

    # ---- role inventories ----
    def scaffolds(self, role: str) -> List[str]:
        if role == "Time" and self.config.time_markers == "2token":
            return list(R.TIME_SCAFFOLDS_2TOKEN)
        return list(R.SCAFFOLDS[role])

    def scaffold_token_length(self, role: str) -> int:
        """Token length shared by a role's scaffolds (1 for paper-setting Time, else 2)."""
        return 1 if (role == "Time" and self.config.time_markers == "1token") else 2

    def cue_swap_roles(self, role: str) -> List[str]:
        if role == "Time" and self.config.time_markers == "2token":
            return list(R.TIME_CUE_SWAP_2TOKEN)
        return list(R.ROLE_CUE_SWAP[role])

    def verbs(self, role: str) -> List[str]:
        if role == "Time" and self.config.time_markers == "2token":
            return list(R.TIME_VERBS_2TOKEN)
        return list(R.VERBS[R.FRAME[role]])

    def frame(self, role: str) -> str:
        return R.FRAME[role]

    def has_object(self, role: str) -> bool:
        return role in R.ROLES_WITH_OBJECT

    def all_scaffolds(self) -> List[str]:
        out = []
        for role in self.roles:
            out.extend(self.scaffolds(role))
        return list(dict.fromkeys(out))

    def paraphrases(self, role: str) -> List[str]:
        return list(W.SCAFFOLD_PARAPHRASES[role])

    # ---- helpers ----
    def _raw_heads(self, role: str) -> List[str]:
        if role == "Time" and self.config.time_markers == "2token":
            return W.TIME_HEADS_2TOKEN
        return W.HEADS[role]

    def _filter(self, words: List[str]) -> List[str]:
        if not self.config.single_token_only:
            return list(words)
        return [w for w in words if self.is_single_token(w)]

    def summary(self) -> Dict[str, int]:
        """Counts of surviving words, useful to report tokenizer coverage."""
        out = {"agents": len(self.agents), "themes": len(self.themes), "names": len(self.names)}
        out.update({f"heads/{r}": len(v) for r, v in self.heads.items()})
        return out

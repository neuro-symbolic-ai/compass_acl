"""Builds clean prompts, their corruptions and same-role interventions."""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from ...lexicon import Lexicon
from .config import GenerationConfig
from .utils import build_full_sentence, replace_last_scaffold, target_leaks


@dataclass
class Prompt:
    """A clean prompt and the choices that produced it."""
    text: str
    role: str
    frame: str
    verb: str
    agent: str
    theme: Optional[str]
    scaffold: str
    target: str


@dataclass
class Corruption:
    """A corrupted prompt with the corrupted head and the role its cue points to."""
    text: str
    corrupt: str
    corrupt_role: str


class PromptBuilder:
    """Surface generation for one lexicon and one random generator."""

    def __init__(self, lexicon: Lexicon, config: GenerationConfig, rng: random.Random):
        self.lex = lexicon
        self.config = config
        self.rng = rng

    # ---- clean prompt ----
    def make_clean(self, role: str) -> Prompt:
        lex, rng = self.lex, self.rng
        verb = rng.choice(lex.verbs(role))
        agent = rng.choice(lex.agents)
        theme = rng.choice(lex.themes) if lex.has_object(role) else None
        scaffold = rng.choice(lex.scaffolds(role))
        target = rng.choice(lex.heads[role])

        subject = f"The{self._maybe(lex.adj_agent, self.config.adjective_prob)} {agent}{self._maybe(lex.adverbs, self.config.adverb_prob)}"
        if theme is not None:
            obj = f"the{self._maybe(lex.adj_theme, self.config.adjective_prob)} {theme}"
            text = f"{subject} {verb} {obj} {scaffold}"
        else:
            text = f"{subject} {verb} {scaffold}"

        if target_leaks(text, [target]):
            # Regenerate without modifiers; the caller re-checks leakage.
            agent = rng.choice(lex.agents)
            if theme is not None:
                theme = rng.choice(lex.themes)
                text = f"The {agent} {verb} the {theme} {scaffold}"
            else:
                text = f"The {agent} {verb} {scaffold}"

        return Prompt(text.strip(), role, lex.frame(role), verb, agent, theme, scaffold, target)

    # ---- corruptions ----
    def make_role_cue_corruption(self, prompt: Prompt) -> Optional[Corruption]:
        """Swap the scaffold for another role's scaffold of equal token length.

        Returns None when no candidate exists and same-role fallback is off.
        """
        lex, rng = self.lex, self.rng
        clean_len = lex.token_length(prompt.text)
        candidates = []
        for tgt_role in lex.cue_swap_roles(prompt.role):
            for sc in lex.scaffolds(tgt_role):
                corrupted = replace_last_scaffold(prompt.text, prompt.scaffold, sc)
                if corrupted == prompt.text and self.config.skip_identical_corruption:
                    continue
                if lex.token_length(corrupted) == clean_len:
                    candidates.append((corrupted, tgt_role))
        if candidates:
            corrupted, tgt_role = rng.choice(candidates)
            corrupt_pool = [w for w in lex.heads[tgt_role] if w != prompt.target]
            if corrupt_pool:
                return Corruption(corrupted, rng.choice(corrupt_pool), tgt_role)

        if not self.config.allow_same_role_fallback:
            return None
        corrupt_pool = [w for w in lex.heads[prompt.role] if w != prompt.target]
        if not corrupt_pool:
            return None
        return Corruption(prompt.text, rng.choice(corrupt_pool), prompt.role)


    def make_corrupted_hard(self, prompt: Prompt) -> str:
        """A longer variant with a preface clause; the scaffold is unchanged."""
        if not self.config.allow_hard_longer:
            return prompt.text
        preface = self.rng.choice(self.lex.prefaces)
        return f"{preface} {prompt.text[0].lower()}{prompt.text[1:]}"

    # ---- interventions ----
    def make_interventions(self, prompt: Prompt) -> List[Dict]:
        """Same-role head swaps on the full sentence (sensical by construction)."""
        base_full = build_full_sentence(prompt.text, prompt.target)
        pool = [w for w in self.lex.heads[prompt.role] if w != prompt.target]
        self.rng.shuffle(pool)
        return [{
            "type": "role_swap_sensical",
            "replacement_head": repl,
            "full_sentence": base_full.replace(f" {prompt.target}.", f" {repl}."),
            "sensicality": "sensical",
        } for repl in pool[:self.config.n_interventions]]

    # ---- helpers ----
    def _maybe(self, pool: List[str], prob: float) -> str:
        return " " + self.rng.choice(pool) if pool and self.rng.random() < prob else ""

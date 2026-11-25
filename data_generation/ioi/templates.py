"""IOI sentence templates, after Wang et al. (2023) and the EAP-IG release.

[A] is the indirect object (IO, the answer), [B] the subject (S, the giver),
[C] a third name used only by the ABC templates. [PLACE] and [OBJECT] are
filled from the lexicon.
"""

from typing import List

BABA_TEMPLATES: List[str] = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
    # Additional templates with the same structure.
    "Then, [B] and [A] stopped by the [PLACE]. [B] offered a [OBJECT] to [A]",
    "Then, [B] and [A] visited the [PLACE]. [B] handed a [OBJECT] to [A]",
    "Then, [B] and [A] hurried to the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] relaxed at the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] met at the [PLACE]. [B] planned to give a [OBJECT] to [A]",
    "Then, [B] and [A] spent the afternoon at the [PLACE]. [B] chose to give a [OBJECT] to [A]",
    "Then, [B] and [A] waited at the [PLACE]. [B] promised a [OBJECT] to [A]",
    "Then, [B] and [A] worked late at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] talked at the [PLACE]. [B] said they would give a [OBJECT] to [A]",
]

ABC_TEMPLATES: List[str] = [
    "Then, [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "Afterwards [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "When [A], [B] and [C] arrived at the [PLACE], [B] and [C] gave a [OBJECT] to [A]",
    "Friends [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "Then, [A], [B] and [C] visited the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "Afterwards [A], [B] and [C] stopped by the [PLACE]. [B] and [C] decided to give a [OBJECT] to [A]",
    "When [A], [B] and [C] met at the [PLACE], [B] and [C] offered a [OBJECT] to [A]",
    "Friends [A], [B] and [C] relaxed at the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
]


def swap_first_clause(templates: List[str]) -> List[str]:
    """Swap [A] and [B] in the first clause only (BABA -> ABBA, ABC -> BAC)."""
    out = []
    for t in templates:
        first, sep, rest = t.partition(". ") if ". " in t else t.partition(", ")
        first = first.replace("[A]", "[X]").replace("[B]", "[A]").replace("[X]", "[B]")
        out.append(first + sep + rest)
    return out


ABBA_TEMPLATES: List[str] = swap_first_clause(BABA_TEMPLATES)
BAC_TEMPLATES: List[str] = swap_first_clause(ABC_TEMPLATES)

TEMPLATE_SETS = {
    "BABA": BABA_TEMPLATES,
    "ABBA": ABBA_TEMPLATES,
    "mixed": BABA_TEMPLATES + ABBA_TEMPLATES,
    "ABC": ABC_TEMPLATES,
    "BAC": BAC_TEMPLATES,
    "ABC mixed": ABC_TEMPLATES + BAC_TEMPLATES,
}

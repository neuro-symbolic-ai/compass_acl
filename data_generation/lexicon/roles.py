"""Semantic roles, their prepositional scaffolds, verb frames and cue-swap map.

A scaffold is the phrase that ends a clean prompt and cues the role of the
next token, for example "to the" for Goal. Scaffolds are grouped into
token-parity families so that a corrupted prompt can swap the cue for another
role's cue without changing the prompt's token length.
"""

from typing import Dict, List

ROLES: List[str] = [
    "Goal", "Location", "Source", "Instrument", "Topic",
    "Material", "Path", "Beneficiary", "Manner", "Time",
]

# Scaffolds per role, for the "1token" marker family used in the paper.
# Only Time has one-token scaffolds (at / on); the other roles are two-token.
SCAFFOLDS: Dict[str, List[str]] = {
    "Goal":        ["to the", "for the"],
    "Location":    ["in the", "at the", "near the", "inside the"],
    "Source":      ["from the"],
    "Instrument":  ["with a", "with the", "using a"],
    "Topic":       ["about the"],
    "Material":    ["of the"],
    "Path":        ["through the", "into the"],
    "Beneficiary": ["for the"],
    "Manner":      ["with the"],
    "Time":        ["at", "on"],
}

# Role-cue corruption: for a target role, the roles whose scaffold can replace
# the clean scaffold. Only same-parity swaps are attempted at generation time.
# Time maps to itself because no other role has one-token scaffolds, so its
# corruption is at <-> on. ROLE_CUE_SWAP_2TOKEN fixes that restriction.
ROLE_CUE_SWAP: Dict[str, List[str]] = {
    "Goal":        ["Location", "Topic", "Beneficiary"],
    "Location":    ["Source", "Goal", "Instrument", "Path"],
    "Source":      ["Location", "Material"],
    "Instrument":  ["Location", "Manner"],
    "Topic":       ["Goal", "Location"],
    "Material":    ["Source", "Location"],
    "Path":        ["Location"],
    "Beneficiary": ["Goal"],
    "Manner":      ["Instrument"],
    "Time":        ["Time"],
}

# Overrides for the "2token" marker family (LexiconConfig.scaffold_markers).
# Roles absent from these dicts keep the entries above. Any role can be given
# an alternative inventory here; at present only Time differs.
SCAFFOLDS_2TOKEN: Dict[str, List[str]] = {
    "Time": ["at the", "on the", "during the", "before the", "after the"],
}
ROLE_CUE_SWAP_2TOKEN: Dict[str, List[str]] = {
    "Time": ["Location", "Goal"],
}

# Verb inventories by semantic frame.
VERBS: Dict[str, List[str]] = {
    "giving_prep_dative":       ["gave", "handed", "passed", "offered", "sent"],
    "transitive_with_location": ["put", "placed", "left", "stored", "fixed", "painted"],
    "motion_with_source":       ["carried", "brought", "moved", "took", "fetched", "retrieved"],
    "instrumental_action":      ["cut", "carved", "opened", "drew", "wrote", "marked", "drilled", "shaped"],
    "communication_topic":      ["spoke", "talked", "wrote", "reported"],
    "creation_with_material":   ["made", "built", "formed", "crafted"],
    "motion_along_path":        ["moved", "walked", "ran", "drove"],
    "benefactive_service":      ["prepared", "cooked", "packed", "saved"],
    "basic_intransitive_time":  ["arrived", "started", "stayed", "worked"],
    "manner_with_nominal":      ["handled", "solved", "worked", "spoke"],
}
# Verb overrides for the "2token" family, keyed by role.
VERBS_2TOKEN: Dict[str, List[str]] = {
    "Time": ["arrived", "started", "ended", "worked", "stayed", "met", "left", "waited"],
}

# Semantic frame used by each target role.
FRAME: Dict[str, str] = {
    "Goal":        "giving_prep_dative",
    "Location":    "transitive_with_location",
    "Source":      "motion_with_source",
    "Instrument":  "instrumental_action",
    "Topic":       "communication_topic",
    "Material":    "creation_with_material",
    "Path":        "motion_along_path",
    "Beneficiary": "benefactive_service",
    "Time":        "basic_intransitive_time",
    "Manner":      "manner_with_nominal",
}

# Roles whose prompt contains a direct object ("The chef handed the book to the").
ROLES_WITH_OBJECT: List[str] = ["Goal", "Location", "Source", "Instrument", "Material", "Beneficiary"]

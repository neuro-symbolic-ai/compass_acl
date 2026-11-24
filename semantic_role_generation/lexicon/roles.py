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

# Scaffolds per role. Time has two inventories: the one-token family used in
# the paper (at / on) and a two-token family that lets Time swap with the
# other roles. GenerationConfig.time_markers selects between them.
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
TIME_SCAFFOLDS_2TOKEN: List[str] = ["at the", "on the", "during the", "before the", "after the"]

# Role-cue corruption: for a target role, the roles whose scaffold can replace
# the clean scaffold. Only same-parity swaps are attempted at generation time.
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
TIME_CUE_SWAP_2TOKEN: List[str] = ["Location", "Goal"]

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
TIME_VERBS_2TOKEN: List[str] = ["arrived", "started", "ended", "worked", "stayed", "met", "left", "waited"]

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

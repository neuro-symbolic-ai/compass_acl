"""Raw word lists for prompt generation.

Every list here is unfiltered. The Lexicon class keeps only the words that a
given tokenizer encodes as a single token (with a leading space), so the
next-token target is always one token. New words can be added and filtering checks them.
"""

from typing import Dict, List

# Subjects and direct objects shared by all roles.
AGENTS: List[str] = [
    "teacher", "worker", "builder", "chef", "driver", "farmer", "clerk", "manager",
    "nurse", "guard", "artist", "carpenter", "painter", "writer", "baker", "mover",
    "doctor",
]
THEMES: List[str] = [
    "book", "file", "box", "chair", "shelf", "bread", "letter", "package", "tools", "frame",
    "door", "table", "wall", "cup", "lamp", "cake", "crate", "parts", "list", "plan", "rule",
]

# Target heads per role: the correct next token after the role's scaffold.
HEADS: Dict[str, List[str]] = {
    "Goal":        ["student", "teacher", "worker", "manager", "child", "nurse", "clerk",
                    "driver", "farmer", "guard", "doctor"],
    "Location":    ["office", "kitchen", "garden", "yard", "room", "hall", "park", "school",
                    "garage", "factory", "studio"],
    "Source":      ["store", "barn", "house", "market", "warehouse", "factory", "garage",
                    "pantry", "cellar", "library"],
    "Instrument":  ["knife", "saw", "pen", "brush", "hammer", "drill", "wrench", "axe",
                    "shovel", "ruler"],
    "Topic":       ["plan", "rule", "budget", "law", "deal", "task", "goal", "issue", "risk",
                    "idea", "report"],
    "Material":    ["wood", "steel", "stone", "glass", "clay", "brick", "paper", "plastic",
                    "cotton", "wool"],
    "Path":        ["tunnel", "gate", "field", "yard", "hall", "door", "river", "street",
                    "park", "bridge"],
    "Beneficiary": ["client", "guest", "friend", "child", "buyer", "owner", "visitor",
                    "partner", "team", "group"],
    "Manner":      ["care", "ease", "force", "speed", "grace", "skill", "calm", "focus"],
    "Time":        ["morning", "noon", "evening", "night", "midnight", "dawn", "dusk",
                    "monday", "friday", "saturday"],
}
# Head overrides for the "2token" scaffold family, keyed by role. Time heads
# here read naturally after "at the" / "during the".
HEADS_2TOKEN: Dict[str, List[str]] = {
    "Time": ["start", "end", "break", "shift", "weekend", "holiday", "meeting", "deadline",
             "summer", "winter"],
}

# Optional modifiers for surface variation.
ADVERBS: List[str] = ["quietly", "quickly", "carefully", "slowly", "calmly", "briefly", "softly"]
ADJ_AGENT: List[str] = ["old", "young", "new", "busy", "tired"]
ADJ_THEME: List[str] = ["small", "large", "wooden", "metal", "old", "new"]

# Prefaces for the longer "hard" corruption - used with IOI
PREFACES: List[str] = [
    "After a short delay,", "Before the break,", "Late in the day,",
    "Without much fuss,", "Following the plan,",
]

# Scaffold paraphrases within a role, for sanity-check prompts.
SCAFFOLD_PARAPHRASES: Dict[str, List[str]] = {
    "Goal":        ["to the", "toward the", "towards the"],
    "Location":    ["in the", "at the", "near the", "inside the", "within the"],
    "Source":      ["from the", "out of the"],
    "Instrument":  ["with a", "with the", "using a", "using the"],
    "Topic":       ["about the", "regarding the", "concerning the"],
    "Material":    ["of the", "out of the", "from the"],
    "Path":        ["through the", "into the", "along the", "across the"],
    "Beneficiary": ["for the", "on behalf of the"],
    "Manner":      ["with the", "with great"],
    "Time":        ["at", "on", "during the", "before the", "after the"],
}

# First names for indirect-object-identification (IOI) prompts. Duplicates
# between the two source lists are removed while keeping the original order.
_NAMES_COMMON: List[str] = [
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer", "Joshua",
    "Amanda", "Daniel", "David", "James", "Robert", "John", "Joseph", "Andrew", "Ryan",
    "Brandon", "Jason", "Justin", "Sarah", "William", "Jonathan", "Stephanie", "Brian",
    "Nicole", "Nicholas", "Anthony", "Heather", "Eric", "Elizabeth", "Adam", "Megan",
    "Melissa", "Kevin", "Steven", "Thomas", "Timothy", "Christina", "Kyle", "Rachel",
    "Laura", "Lauren", "Amber", "Brittany", "Danielle", "Richard", "Kimberly", "Jeffrey",
    "Amy", "Crystal", "Michelle", "Tiffany", "Jeremy", "Benjamin", "Mark", "Emily",
    "Aaron", "Charles", "Rebecca", "Jacob", "Stephen", "Patrick", "Sean", "Erin", "Jamie",
    "Kelly", "Samantha", "Nathan", "Sara", "Dustin", "Paul", "Angela", "Tyler", "Scott",
    "Katherine", "Andrea", "Gregory", "Erica", "Mary", "Travis", "Lisa", "Kenneth",
    "Bryan", "Lindsey", "Kristen", "Jose", "Alexander", "Jesse", "Katie", "Lindsay",
    "Shannon", "Vanessa", "Courtney", "Christine", "Alicia", "Cody", "Allison", "Bradley",
    "Samuel",
]
_NAMES_EXTRA: List[str] = [
    "Olivia", "Emma", "Ava", "Sophia", "Isabella", "Mia", "Charlotte", "Amelia", "Harper",
    "Evelyn", "Abigail", "Ella", "Camila", "Luna", "Sofia", "Avery", "Mila", "Aria",
    "Scarlett", "Penelope", "Layla", "Chloe", "Victoria", "Madison", "Eleanor", "Grace",
    "Nora", "Riley", "Zoey", "Hannah", "Hazel", "Lily", "Ellie", "Violet", "Lillian", "Zoe",
    "Stella", "Aurora", "Natalie", "Emilia", "Everly", "Leah", "Aubrey", "Willow", "Addison",
    "Lucy", "Audrey", "Bella",
    "Liam", "Noah", "Oliver", "Elijah", "Lucas", "Henry", "Theodore", "Jack", "Levi",
    "Jackson", "Mateo", "Mason", "Sebastian", "Ethan", "Logan", "Owen", "Asher", "Aiden",
    "Wyatt", "Leo", "Luke", "Julian", "Hudson", "Grayson", "Ezra", "Gabriel", "Carter",
    "Isaac", "Jayden", "Luca", "Dylan", "Lincoln", "Maverick", "Elias", "Josiah",
    "Ivy", "Jade", "Rose", "April", "Summer", "Autumn", "Iris", "Daisy", "Ruby", "Pearl",
    "Sienna", "Clara", "Freya", "Ada", "Mabel", "Esme", "Poppy", "Tessa", "Skye", "Sage",
    "Wren", "Quinn", "Brooke", "Paige", "Blair", "Reese", "Kendall", "Rowan", "Morgan",
    "Miles", "Caleb", "Nolan", "Hugo", "Felix", "Jasper", "Evan", "Finn", "Arthur", "Colin",
    "Gavin", "Parker", "Declan", "Ronan", "Kieran", "Cyrus", "Silas", "Reid", "Micah",
    "Emmett", "Bennett", "Brady", "Grant", "Harvey", "Spencer", "Tristan", "Oscar", "Max",
]
NAMES: List[str] = list(dict.fromkeys(_NAMES_COMMON + _NAMES_EXTRA))

# Places and objects used in IOI templates.
IOI_PLACES: List[str] = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station"]
IOI_OBJECTS: List[str] = ["ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack"]

# Nouns for the greater-than task ("The <noun> lasted from the year 1352 to the year 13").
GREATER_THAN_NOUNS: List[str] = [
    "attack", "campaign", "case", "contact", "contract", "deal", "development", "effect",
    "employment", "existence", "fall", "growth", "impact", "marriage", "operation", "order",
    "plan", "practice", "process", "program", "progress", "project", "quest", "rise",
    "rule", "shift", "study", "test", "testing", "tests", "trial", "trip", "work",
]

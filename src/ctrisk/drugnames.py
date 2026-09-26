"""Normalize drug names so trial interventions and FAERS substances compare equal.

'Cisplatin 75 mg/m2 IV infusion' -> 'cisplatin';  'ONDANSETRON HYDROCHLORIDE' -> 'ondansetron'
"""
import re

_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|kg|ml|l|iu|units?|%)"
                    r"(?:/(?:m2|kg|day|d|h|hr|min|ml|l|dose|wk|week))*(?![a-z])")
_SOLVATE = re.compile(r"\b(?:propylene glycol|acetone|ethanol) solvate\b")
_NOISE = frozenset({
    # salts and counter-ions (hydrates are handled in normalize)
    "hydrochloride", "dihydrochloride", "hcl", "hydrobromide", "sodium", "disodium", "potassium",
    "calcium", "magnesium", "sulfate", "sulphate", "acetate", "maleate", "malate", "mesylate",
    "esylate", "tosylate", "ditosylate", "tartrate", "bitartrate", "citrate", "phosphate",
    "succinate", "besylate", "fumarate", "oxalate", "lactate", "nitrate", "bromide", "chloride",
    "olamine", "meglumine", "tromethamine", "anhydrous",
    # dosage forms, routes, release
    "tablet", "tablets", "capsule", "capsules", "injection", "injectable", "infusion", "oral", "iv",
    "intravenous", "subcutaneous", "intramuscular", "topical", "cream", "gel", "ointment",
    "solution", "suspension", "patch", "spray", "inhaled", "inhalation", "extended", "release",
    "er", "xr", "sr", "dose", "dosing",
    # inert comparators
    "placebo", "saline", "vehicle", "sham", "normal",
})
# catch-all FAERS entries ("vitamin d nos", "unspecified ingredient") that name no specific drug
_VAGUE = frozenset({"nos", "unspecified", "unknown", "ingredient",
                    "vitamins", "minerals", "cosmetics", "probiotics", "antacids"})
# whole substance names that are inert, generic, or fragments of chemical names -- a trial
# "matching" them says nothing about a drug's safety history
_NOT_A_DRUG = frozenset({"protein", "ethyl", "water", "lactose", "dextrose", "glucose", "sucrose",
                         "oxygen", "nitrogen", "air", "alcohol", "honey", "stem cells", "lymphocytes"})
MIN_SINGLE_WORD = 4  # a one-word match shorter than this ("tea") is too likely to be noise


def normalize(name: str) -> str:
    text = _DOSE.sub(" ", name.lower())
    text = _SOLVATE.sub(" ", text)
    text = re.sub(r"\b[rs]-(?=[a-z])", "", text)     # stereo prefix: "s-malate" -> "malate"
    text = re.sub(r"(?<=[a-z])-(?=\d)", "", text)    # keep codes whole: "il-2" -> "il2"
    tokens = re.sub(r"[^a-z0-9]+", " ", text).split()
    return " ".join(t for t in tokens
                    if t not in _NOISE and not t.endswith("hydrate") and not t.isdigit())


def is_specific(name: str) -> bool:
    return name not in _NOT_A_DRUG and not _VAGUE & set(name.split())


def match_substances(name: str, vocab: frozenset[str], max_words: int = 4) -> list[str]:
    """Substances from `vocab` in a normalized name. Scans longest phrase first, no overlaps; returns them sorted."""
    words = name.split()
    found, i = set(), 0
    while i < len(words):
        for n in range(min(max_words, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            if phrase in vocab and (n > 1 or len(phrase) >= MIN_SINGLE_WORD):
                found.add(phrase)
                i += n
                break
        else:
            i += 1
    return sorted(found)

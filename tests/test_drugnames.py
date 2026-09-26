import pytest

from ctrisk.drugnames import is_specific, match_substances, normalize


@pytest.mark.parametrize("raw, expected", [
    ("Metformin 500 mg", "metformin"),
    ("doxorubicin hydrochloride", "doxorubicin"),
    ("ONDANSETRON HYDROCHLORIDE", "ondansetron"),
    ("5-fluorouracil", "fluorouracil"),
    ("Cisplatin 75 mg/m2 IV infusion", "cisplatin"),
    ("Insulin Glargine", "insulin glargine"),
    ("Placebo", ""),
    ("Sodium chloride 0.9%", ""),
    ("Normal saline", ""),
    ("3 lines of therapy", "lines of therapy"),
    ("Emtricitabine 200 mg/Tenofovir 300 mg", "emtricitabine tenofovir"),
    ("Heparin 18 units/kg/h", "heparin"),
    ("IL-2", "il2"),
    ("MK-3475", "mk3475"),
    # salt, hydrate, and solvate forms as FAERS writes them
    ("ATORVASTATIN CALCIUM TRIHYDRATE", "atorvastatin"),
    ("ATORVASTATIN CALCIUM PROPYLENE GLYCOL SOLVATE", "atorvastatin"),
    ("CABOZANTINIB S-MALATE", "cabozantinib"),
    ("ESCITALOPRAM OXALATE", "escitalopram"),
    ("NIRAPARIB TOSYLATE MONOHYDRATE", "niraparib"),
    ("LAPATINIB DITOSYLATE", "lapatinib"),
    ("NINTEDANIB ESYLATE", "nintedanib"),
    ("ELTROMBOPAG OLAMINE", "eltrombopag"),
    ("SUNITINIB MALATE", "sunitinib"),
    ("PANOBINOSTAT LACTATE", "panobinostat"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


VOCAB = frozenset({"pembrolizumab", "insulin", "insulin glargine", "iron", "tea"})


def test_matches_longest_phrase_first():
    assert match_substances("insulin glargine", VOCAB) == ["insulin glargine"]


def test_matches_several_substances_in_one_name():
    assert match_substances("pembrolizumab plus insulin glargine", VOCAB) == [
        "insulin glargine", "pembrolizumab"]


def test_ignores_short_single_words():
    assert match_substances("green tea extract", VOCAB) == []
    assert match_substances("iron", VOCAB) == ["iron"]


def test_no_match():
    assert match_substances("", VOCAB) == []
    assert match_substances("exercise program", VOCAB) == []


@pytest.mark.parametrize("name, expected", [
    ("capreomycin", True),
    ("acetaminophen hydrocodone", True),
    ("vitamin d nos", False),
    ("unspecified ingredient", False),
    ("vitamins", False),
    ("cosmetics", False),
    ("protein", False),         # fragment of verbose chemical names
    ("water", False),           # diluent / placebo vehicle
    ("stem cells", False),      # not one identifiable product
    ("icosapent ethyl", True),  # a real drug that contains a filtered word
    ("dobesilate", True),
])
def test_is_specific(name, expected):
    assert is_specific(name) is expected

"""Tests for the unknown-ingredient screen. Run from the project folder:
    .venv/bin/python -m pytest test/test_screen.py
"""
from regimen.data.normalize import clean_name, ingredient_names
from regimen.safety.knowledge import build_banned_terms
from regimen.safety.screen import Screen, edit_distance, is_close

SYNONYMS = {"aqua": "aqua", "water": "aqua", "parfum (fragrance)": "parfum", "glycerin": "glycerin"}
BANNED = {"linalool": "stated allergen: linalool", "parfum": "stated allergen category: fragrance",
          "alcohol denat": "stated allergen: alcohol denat"}


def screen():
    return Screen(BANNED, SYNONYMS)


def test_edit_distance():
    assert edit_distance("linalol", "linalool") == 1
    assert edit_distance("same", "same") == 0


def test_short_names_must_match_exactly():
    assert not is_close("citrel", "citral")       # 6 letters: no mistakes allowed
    assert is_close("linalol", "linalool")        # 8 letters: one mistake allowed


def test_clean_name_strips_strength_and_label():
    assert clean_name("avobenzone 3%") == "avobenzone"
    assert clean_name("active ingredients: octocrylene (5%)") == "octocrylene"


def test_comma_inside_chemical_name_is_kept():
    assert ingredient_names("water, 1,2-hexanediol, glycerin") == ["water", "1,2-hexanediol", "glycerin"]


def test_exact_match_after_synonym_excludes():
    info = screen().analyze("water, parfum (fragrance)")
    d = screen().decide(info, cutoff=1.0)
    assert d["status"] == "EXCLUDE" and d["reasons"][0].startswith("EXCLUDED: contains parfum")


def test_typo_and_glued_name_are_possible_matches():
    s = screen()
    assert s.decide(s.analyze("water, linalol"), 1.0)["reasons"][0].startswith("POSSIBLE_MATCH")
    assert s.decide(s.analyze("water, peg-8 parfum"), 1.0)["status"] == "EXCLUDE"


def test_generic_word_inside_other_name_is_not_flagged():
    s = screen()
    assert s.decide(s.analyze("water, cetyl alcohol"), 1.0)["status"] == "KEEP"


def test_unrecognized_share_cutoff():
    s = screen()
    info = s.analyze("water, glycerin, mystery one, mystery two")  # 2 of 4 unrecognized = 50%
    assert s.decide(info, 0.50)["status"] == "KEEP"                # not MORE than 50%
    assert s.decide(info, 0.40)["status"] == "UNPARSEABLE"


def test_long_name_and_empty_list_are_unparseable():
    s = screen()
    assert s.decide(s.analyze("water, " + "x" * 61), 1.0)["status"] == "UNPARSEABLE"
    assert s.decide(s.analyze(""), 1.0)["reasons"] == ["EMPTY_LIST"]


def test_banned_terms_from_profile():
    banned = build_banned_terms({"avoid_categories": ["fragrance"], "prescriptions": ["tretinoin"]})
    assert "linalool" in banned and "lactic acid" in banned and "retinol" in banned
    assert "tretinoin conflicts with exfoliating acids" == banned["lactic acid"]

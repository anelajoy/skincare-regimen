"""Tests for the pregnancy gate. Run from the project folder:
    .venv/bin/python -m pytest test/test_gate.py
"""
import pytest

from regimen.safety.gate import check_gate


def gate(status, *drugs):
    return check_gate({"pregnancy_status": status, "prescriptions": list(drugs)})


@pytest.mark.parametrize("status", ["pregnant", "unsure", "trying_to_conceive", "prefer_not_to_say"])
def test_isotretinoin_is_referred_for_every_pregnant_like_answer(status):
    assert gate(status, "isotretinoin")["action"] == "REFER"


@pytest.mark.parametrize("status", ["pregnant", "unsure", "trying_to_conceive", "prefer_not_to_say"])
def test_topical_retinoids_are_flagged(status):
    for drug in ("tretinoin", "adapalene"):
        assert gate(status, drug)["action"] == "FLAG"


def test_refer_beats_flag():
    assert gate("unsure", "tretinoin", "isotretinoin")["action"] == "REFER"


def test_prefer_not_to_say_uses_neutral_message():
    assert "pregnan" not in gate("prefer_not_to_say", "isotretinoin")["message"].lower()
    assert "birth" not in gate("prefer_not_to_say", "isotretinoin")["message"].lower()


def test_breastfeeding_flags_retinoids_only():
    assert gate("breastfeeding", "tretinoin")["action"] == "FLAG"
    assert gate("breastfeeding", "azelaic acid")["action"] == "PROCEED"


def test_not_pregnant_and_non_retinoids_proceed():
    assert gate("not_pregnant", "isotretinoin", "tretinoin")["action"] == "PROCEED"
    assert gate("pregnant", "azelaic acid", "benzoyl peroxide", "clindamycin")["action"] == "PROCEED"


def test_unknown_prescription_is_referred():
    r = gate("not_pregnant", "mystery drug")
    assert r["action"] == "REFER" and r["reasons"][0].startswith("UNKNOWN_PRESCRIPTION")


def test_missing_or_invalid_answer_raises():
    with pytest.raises(ValueError):
        check_gate({"prescriptions": ["tretinoin"]})
    with pytest.raises(ValueError):
        gate("maybe", "tretinoin")

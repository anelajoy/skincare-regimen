"""Pregnancy and high-risk gate. It runs BEFORE any product filtering.

It reads the user's pregnancy answer and prescription list and returns one of:
    REFER    no regimen is built; show a "talk to your clinician" message
    FLAG     the regimen is built, but a clinician-review notice is attached
    PROCEED  normal regimen generation

The answers follow this table:
    pregnant, unsure, trying_to_conceive   isotretinoin -> REFER; topical retinoids -> FLAG
    prefer_not_to_say                      same as pregnant, but with a neutral message
    breastfeeding                          retinoids -> FLAG
    not_pregnant                           PROCEED
"unsure" counts as pregnant. An answer that is missing or not in the table raises an
error, so the app cannot skip this check by accident. A prescription that is not in the
rule table is sent to a clinician (REFER) instead of being guessed at.

This is plain code. The LLM is never involved in this decision.
"""
from regimen.safety.knowledge import load_members, load_rules

TREATED_AS_PREGNANT = {"pregnant", "unsure", "trying_to_conceive", "prefer_not_to_say"}
STATUSES = TREATED_AS_PREGNANT | {"breastfeeding", "not_pregnant"}
RANK = {"PROCEED": 0, "FLAG": 1, "REFER": 2}  # the strongest action wins

MESSAGES = {  # DRAFT wording: review before showing to users
    "REFER": "We can't build a routine for this combination. Please talk to your prescriber "
             "or another clinician first.",
    "REFER_NEUTRAL": "Based on your answers, please review your prescriptions with a clinician "
                     "before we build a routine.",
    "FLAG": "A routine is shown below, but please have a clinician review your prescriptions "
            "before you start it.",
}


def check_gate(profile: dict) -> dict:
    status = profile.get("pregnancy_status")
    if status not in STATUSES:
        raise ValueError(f"pregnancy_status must be one of {sorted(STATUSES)}, got {status!r}")

    rules = load_rules()
    retinoids = set(load_members("ingredient_classes.csv", "class")["retinoids"])
    prescribed_retinoids = lambda d: d in rules and d in retinoids  # noqa: E731

    action, reasons = "PROCEED", []

    def raise_to(new_action, reason):
        nonlocal action
        reasons.append(reason)
        if RANK[new_action] > RANK[action]:
            action = new_action

    for drug in profile.get("prescriptions", []):
        drug = drug.lower()
        if drug not in rules:
            raise_to("REFER", f"UNKNOWN_PRESCRIPTION: {drug} is not in the rule table")
        elif status in TREATED_AS_PREGNANT and drug == "isotretinoin":
            raise_to("REFER", f"HIGH_RISK_REFERRAL: isotretinoin with pregnancy answer '{status}'")
        elif status in TREATED_AS_PREGNANT and prescribed_retinoids(drug):
            raise_to("FLAG", f"CLINICIAN_REVIEW: {drug} with pregnancy answer '{status}'")
        elif status == "breastfeeding" and prescribed_retinoids(drug):
            raise_to("FLAG", f"CLINICIAN_REVIEW: {drug} while breastfeeding")

    if action == "REFER":
        neutral = status == "prefer_not_to_say" and not any(r.startswith("UNKNOWN") for r in reasons)
        message = MESSAGES["REFER_NEUTRAL" if neutral else "REFER"]
    else:
        message = MESSAGES.get(action, "")
    return {"action": action, "reasons": reasons, "message": message}

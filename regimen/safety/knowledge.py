"""Load the files in knowledge/ and turn a user profile into a list of banned terms.

A profile is a plain dictionary, for example:
    {"avoid_ingredients": ["linalool"],        # names the user typed
     "avoid_categories":  ["fragrance"],       # category words the user typed
     "prescriptions":     ["tretinoin"]}       # actives from the prescription list

build_banned_terms returns {banned name: reason}. The reason says WHY the name is
banned, and it becomes the explanation shown to the user.

Rows with match = "contains" (keyword rules such as "extract" or "benzoyl peroxide")
are skipped here: they cannot be turned into a list of names.
"""
import csv
from pathlib import Path

KNOWLEDGE = Path(__file__).resolve().parents[2] / "knowledge"


def _rows(filename):
    with open(KNOWLEDGE / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_synonyms() -> dict:
    """{variant spelling: standard name}"""
    return {r["variant"]: r["standard"] for r in _rows("synonyms.csv")}


def load_members(filename: str, key_column: str) -> dict:
    """{category or class: [ingredient names]}, exact-match rows only."""
    members = {}
    for r in _rows(filename):
        if r["match"] == "exact":
            members.setdefault(r[key_column], []).append(r["ingredient"])
    return members


def load_rules() -> dict:
    """{prescription active: [conflicting class names]}"""
    return {r["active"]: [c.strip() for c in r["conflicting_ingredient_classes"].split(";") if c.strip()]
            for r in _rows("rules.csv")}


def build_banned_terms(profile: dict, synonyms: dict = None) -> dict:
    synonyms = synonyms if synonyms is not None else load_synonyms()
    categories = load_members("allergen_categories.csv", "category")
    classes = load_members("ingredient_classes.csv", "class")
    rules = load_rules()

    banned = {}

    def add(term, reason):
        banned.setdefault(term.lower(), reason)  # first reason wins

    for name in profile.get("avoid_ingredients", []):
        standard = synonyms.get(name.lower(), name.lower())
        add(standard, f"stated allergen: {name}")

    for category in profile.get("avoid_categories", []):
        category = category.lower()
        if category not in categories:  # fail loudly: a typo here would silently skip a safety rule
            raise ValueError(f"unknown allergen category: {category!r}")
        add(category, f"stated allergen category: {category}")  # so a misspelled "fragrence" is caught
        for ingredient in categories[category]:
            add(ingredient, f"stated allergen category: {category}")

    for drug in profile.get("prescriptions", []):
        drug = drug.lower()
        if drug not in rules:
            raise ValueError(f"prescription not in rule table: {drug!r}")
        for cls in rules[drug]:
            for ingredient in classes.get(cls, []):
                add(ingredient, f"{drug} conflicts with {cls}")

    return banned

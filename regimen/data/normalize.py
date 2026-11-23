"""Turn one raw ingredient string into a clean list of ingredient names.

Step 1 is split_ingredients (already written in ingredient_frequency.py): it cuts the
text at commas. Step 2 is clean_name (below): it removes strengths and label words,
so "avobenzone 3%" becomes "avobenzone" and "active ingredients: octocrylene"
becomes "octocrylene".
"""
import re

from regimen.data.ingredient_frequency import split_ingredients

# a number followed by %, with optional brackets, e.g. "3%", "(5 %)", "2.5%"
PERCENT = re.compile(r"\s*\(?\s*\d+(?:\.\d+)?\s*%\s*\)?")
# label words at the start of a name, e.g. "active ingredients :" or "inactive ingredients:"
PREFIX = re.compile(r"^(?:(?:in)?active\s+)?ingr[eé]dients?\s*:\s*|^active\s*:\s*")


def clean_name(name: str) -> str:
    name = PREFIX.sub("", name)
    name = PERCENT.sub("", name)
    return re.sub(r"\s+", " ", name).strip(" .;:")


def ingredient_names(raw_text) -> list[str]:
    """Raw ingredient text -> list of cleaned, lowercase names (empty names dropped)."""
    cleaned = (clean_name(n) for n in split_ingredients(raw_text))
    return [n for n in cleaned if n]

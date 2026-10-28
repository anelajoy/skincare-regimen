"""Filter Open Beauty Facts down to facial skincare products with ingredient text."""
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/obf_products.csv.gz")
OUT = Path("data/processed/products_raw.parquet")

USECOLS = ["code", "product_name", "brands", "categories_tags",
           "countries_tags", "ingredients_text"]

EXCLUDE_TAGS = {
    "en:nail-polish-remover", "en:lip-cosmetics", "en:lip-balms", "en:makeup",
    "en:hair", "en:shampoos", "en:hair-care", "en:hair-conditioners",
    "en:deodorants", "en:toothpastes", "en:toothpaste", "en:perfumes",
    "en:soaps", "en:shower-gels", "en:showers-and-baths", "en:hand-creams",
    "en:body-creams", "en:body-milks", "en:body-oils", "en:body",
}

# Matches at the start of a word (so "shampoos" matches, "ingredient" does not)
EXCLUDE_PREFIX = [
    "shampo", "conditioner", "cabello", "cheveux", "hair", "capelli", "haar",
    "mouth", "tooth", "zahn", "dental", "soap", "savon", "jab[oó]n", "seife",
    "body", "bodi", "corps", "cuerpo", "shower", "douche", "foot", "pied",
    "nail", "ongles", "deodor", "d[eé]odorant", "surface", "towel", "wipe",
    "lingette", "serviette", "ailette", "baby", "b[eé]b[eé]", "diaper",
    "stretch", "vergetures", "massage", "shav", "raser", "d[eé]pilatoire",
    "depil", "perfume", "bubble", "bath", "shine",
]
# Short words that must match as whole words only
EXCLUDE_WORD = ["hand", "hands", "main", "mains", "mano", "lip", "lips",
                "dent", "bain", "pad", "pads"]
EXCLUDE_NAME = re.compile(
    r"\b(?:" + "|".join(EXCLUDE_PREFIX) + r")|\b(?:" + "|".join(EXCLUDE_WORD) + r")\b"
)

FACE_NAME = re.compile(r"face|facial|visage|gesicht|rostro|\bcara\b|\bviso\b|gezicht|"
                       r"\beye\b|yeux|augen|ojos")
FACE_TAGS = {
    "en:face", "en:facial-creams", "en:cleansers", "en:face-lotions",
    "en:day-creams", "en:night-creams", "en:face-masks", "en:face-scrubs",
    "en:face-serum", "en:micellar-water", "en:cleansing-waters",
    "en:cleansing-milks", "en:anti-aging-face-care-products",
    "en:facial-sunscreens", "en:moisturizers", "en:moisturizer",
    "en:anti-wrinkles-creams",
}

TYPE_PATTERNS = {
    "sunscreen": re.compile(r"sun|spf|solaire|solar|sonnen|zonnebrand|protector"),
    "cleanser": re.compile(r"cleans|face wash|facial wash|nettoy|micell|limpiador|"
                           r"gel moussant|reinig"),
    "moisturizer": re.compile(r"moistur|cream|cr[eè]me|crema|lotion|hydrat|day |night |nuit|jour"),
}


def classify(name: str, categories: str):
    """Return (product_type, face_confidence), or (None, None) if out of scope."""
    name = (name or "").strip().lower()
    if not name:
        return None, None
    tags = set((categories or "").split(","))
    if tags & EXCLUDE_TAGS or EXCLUDE_NAME.search(name):
        return None, None
    text = f"{name} {categories}".lower()
    ptype = next((t for t, p in TYPE_PATTERNS.items() if p.search(text)), None)
    if ptype is None:
        return None, None
    face = bool(tags & FACE_TAGS) or bool(FACE_NAME.search(name))
    return ptype, ("high" if face else "medium")


def in_scope(ptype: str, confidence: str) -> bool:
    """Sunscreens and cleansers: any confidence. Moisturizers: face-confident only."""
    return ptype in {"sunscreen", "cleanser"} or confidence == "high"


def main():
    stats = {}
    df = pd.read_csv(RAW, sep="\t", usecols=USECOLS, low_memory=False,
                     on_bad_lines="skip")
    stats["total products"] = len(df)

    df = df[df["ingredients_text"].fillna("").str.strip() != ""].copy()
    stats["with ingredient text"] = len(df)

    results = [classify(n, c) for n, c in zip(df["product_name"].fillna(""),
                                              df["categories_tags"].fillna(""))]
    df["product_type"] = [r[0] for r in results]
    df["face_confidence"] = [r[1] for r in results]
    df = df[df["product_type"].notna()]
    stats["skincare by keyword/tag"] = len(df)

    df = df[[in_scope(t, c) for t, c in zip(df["product_type"], df["face_confidence"])]]
    stats["in scope (facial rule)"] = len(df)

    df["_key"] = (df["brands"].fillna("").str.lower() + "|"
                  + df["product_name"].str.lower().str.strip())
    df = df.drop_duplicates("_key").drop(columns="_key")
    stats["after de-duplication"] = len(df)

    df["us_sold"] = df["countries_tags"].fillna("").str.contains("en:united-states")
    df = df.rename(columns={"ingredients_text": "ingredients_raw", "brands": "brand"})
    df["source"] = "obf"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    for k, v in stats.items():
        print(f"{k:28s} {v:>7,}")
    print(f"{'US-sold':28s} {int(df['us_sold'].sum()):>7,}")
    print(df.groupby(["product_type", "face_confidence"]).size())
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
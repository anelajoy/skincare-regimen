import pandas as pd
import re

df = pd.read_csv(
    "data/raw/obf_products.csv.gz",
    sep="\t",
    low_memory=False,
    on_bad_lines="skip",
)
print(df.shape)
print(df.columns.tolist())

# Which columns matter?
cols = [c for c in df.columns if any(k in c.lower() for k in
        ["product_name", "brands", "ingredients", "countries", "categories", "code"])]
print(cols)

# How complete is the ingredient text?
has_ing = df["ingredients_text"].notna() & (df["ingredients_text"].str.strip() != "")
print("With ingredients:", has_ing.sum(), f"({has_ing.mean():.1%})")

# How many are tagged as sold in the US?
us = df["countries_tags"].fillna("").str.contains("en:united-states")
print("US-tagged:", us.sum(), f"({us.mean():.1%})")
print("US + ingredients:", (us & has_ing).sum())

# Top categories among usable products
usable = df[has_ing]
print(usable["categories_tags"].fillna("").str.split(",").explode().value_counts().head(30))

# A few raw ingredient strings
print(usable["ingredients_text"].sample(10, random_state=1).tolist())

SKIN = {"en:face", "en:facial-creams", "en:body-creams", "en:hand-creams",
        "en:cleansers", "en:suncare", "en:sunscreen", "en:in-sun-protections",
        "en:moisturizers", "en:facial-cleansers", "en:serums"}
NOT_SKIN = {"en:hair", "en:shampoos", "en:hair-care", "en:hair-conditioners",
            "en:deodorants", "en:toothpastes", "en:toothpaste", "en:makeup",
            "en:perfumes", "en:soaps", "en:shower-gels", "en:showers-and-baths"}

tags = df["categories_tags"].fillna("").str.split(",").apply(set)
is_skin = tags.apply(lambda t: bool(t & SKIN) and not (t & NOT_SKIN))

print("Skincare (any country):", is_skin.sum())
print("  with ingredients:", (is_skin & has_ing).sum())
print("  with ingredients + US:", (is_skin & has_ing & us).sum())

# What other tags co-occur? Use this to refine SKIN
sk = df[is_skin & has_ing]
print(sk["categories_tags"].str.split(",").explode().value_counts().head(40))

# Are products with empty categories skincare? Peek at names
print(df[(df["categories_tags"].isna()) & has_ing]["product_name"].sample(25, random_state=1).tolist())

# Does the pre-parsed ingredient column look usable?
print(sk[["ingredients_text", "ingredients_tags"]].head(5).to_dict("records"))

EXCLUDE_TAGS = {"en:nail-polish-remover", "en:lip-cosmetics", "en:lip-balms", "en:makeup",
    "en:hair", "en:shampoos", "en:hair-care", "en:hair-conditioners", "en:deodorants",
    "en:toothpastes", "en:toothpaste", "en:perfumes", "en:soaps", "en:shower-gels",
    "en:showers-and-baths", "en:hand-creams", "en:body-creams", "en:body-milks",
    "en:body-oils", "en:body"}

EXCLUDE_NAME = (r"shampoo|shampoing|conditioner|cabello|cheveux|hair|capelli|haar|"
    r"mouth|dent|tooth|zahn|soap|savon|jabón|seife|body|corps|cuerpo|shower|douche|"
    r"hand|main|mano|mains|lip|lèvres|labios|foot|pied|pies|nail|ongles|deodor|déodorant|"
    r"surface|towel|wipe|lingette|baby|bébé|bebe|diaper|stretch|vergetures|massage|"
    r"shav|raser|perfume|parfum de|bubble|bath|bain")

FACE_NAME = r"face|facial|visage|gesicht|rostro|cara|viso|gezicht|eye|yeux|augen|ojos"
FACE_TAGS = {"en:face", "en:facial-creams", "en:cleansers", "en:face-lotions", "en:day-creams",
    "en:night-creams", "en:face-masks", "en:face-scrubs", "en:face-serum", "en:micellar-water",
    "en:cleansing-waters", "en:cleansing-milks", "en:anti-aging-face-care-products",
    "en:facial-sunscreens", "en:moisturizers", "en:moisturizer", "en:anti-wrinkles-creams"}

TYPE_PATTERNS = {
  "sunscreen":   r"sun|spf|solaire|solar|sonnen|zonnebrand|protector",
  "cleanser":    r"cleans|face wash|facial wash|nettoy|micell|limpiador|gel moussant|reinig",
  "moisturizer": r"moistur|cream|crème|creme|crema|lotion|hydrat|day |night |nuit|jour",
}

def classify(row):
    name = str(row["product_name"]).strip().lower()
    if not name or name == "nan":
        return pd.Series([None, None])
    tags = set(str(row["categories_tags"]).split(","))
    if tags & EXCLUDE_TAGS or re.search(EXCLUDE_NAME, name):
        return pd.Series([None, None])
    text = f"{name} {row['categories_tags']}".lower()
    ptype = next((t for t, p in TYPE_PATTERNS.items() if re.search(p, text)), None)
    if ptype is None:
        return pd.Series([None, None])
    face = bool(tags & FACE_TAGS) or bool(re.search(FACE_NAME, name))
    return pd.Series([ptype, "high" if face else "medium"])

d = df[has_ing].copy()
d[["product_type", "face_confidence"]] = d.apply(classify, axis=1)
d = d[d["product_type"].notna()]

# Remove duplicate products (same brand + name)
d["_key"] = d["brands"].fillna("").str.lower() + "|" + d["product_name"].str.lower().str.strip()
d = d.drop_duplicates("_key")
d["us_sold"] = d["countries_tags"].fillna("").str.contains("en:united-states")

print(len(d))
print(d.groupby(["product_type", "face_confidence"]).size())
print("US-sold:", d["us_sold"].sum())
print(d.sample(40, random_state=3)[["product_name", "product_type", "face_confidence"]].to_string())
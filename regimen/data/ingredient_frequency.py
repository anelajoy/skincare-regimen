"""
ingredient_frequency.py

Counts how many products contain each ingredient, so you know which
ingredients the synonym dictionary and allergen-category map must cover first.

Usage:
    python ingredient_frequency.py path/to/products.csv
    python ingredient_frequency.py path/to/products.csv --ingredients-col ingredients_text --type-col product_type

Outputs (written next to this script):
    ingredient_frequency.csv          one row per raw ingredient name, with counts
    ingredient_frequency_by_type.csv  the same counts split by product type (if a type column exists)
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def load_products(path: Path) -> pd.DataFrame:
    """Load CSV, Parquet, or JSON, depending on the file extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    return pd.read_csv(path)


def split_ingredients(text: str) -> list[str]:
    """
    Turn one INCI string into a list of cleaned ingredient names.
    This is deliberately simple: it lowercases, removes footnote marks and
    "may contain" sections, and splits on commas that are not inside parentheses.
    Real normalization (synonyms) happens later, in the dictionary you build.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.lower()
    # Drop "may contain / +/-" sections, which list optional colorants.
    text = re.split(r"\bmay contain\b|\[\+/-\]|\(\+/-\)|\+/-", text)[0]
    # Drop label prefixes such as "ingredients:" or "inci:".
    text = re.sub(r"^\s*(ingredients|inci)\s*:\s*", "", text)
    # Remove footnote marks like * or ** (often "organic").
    text = text.replace("*", "")

    # Protect commas inside chemical numbers like "1,2-hexanediol" (a digit, a comma,
    # then a digit). Without this the name would be cut into "1" and "2-hexanediol".
    text = re.sub(r"(?<!\w)(\d+),(?=\d)", "\\1\x00", text)

    # Split on commas outside parentheses, so "parfum (fragrance)" stays whole.
    parts, depth, current = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))

    cleaned = []
    for p in parts:
        p = p.replace("\x00", ",")  # put the protected commas back
        p = re.sub(r"\s+", " ", p).strip(" .;:")
        if p:
            cleaned.append(p)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("products", type=Path, help="File produced by ingest.py")
    parser.add_argument("--ingredients-col", default="ingredients_text", help="Column holding the INCI list")
    parser.add_argument("--type-col", default="product_type", help="Column holding cleanser/moisturizer/sunscreen")
    parser.add_argument("--top", type=int, default=50, help="How many top ingredients to print")
    args = parser.parse_args()

    df = load_products(args.products)
    if args.ingredients_col not in df.columns:
        raise SystemExit(f"Column '{args.ingredients_col}' not found. Available columns: {list(df.columns)}")

    df["_ingredients"] = df[args.ingredients_col].apply(split_ingredients)
    n_products = len(df)
    n_empty = int((df["_ingredients"].str.len() == 0).sum())

    # Count each ingredient once per product (document frequency), not per mention.
    counts = Counter()
    for ings in df["_ingredients"]:
        counts.update(set(ings))

    out = pd.DataFrame(counts.most_common(), columns=["ingredient", "n_products"])
    out["pct_products"] = (100 * out["n_products"] / n_products).round(2)
    out["cumulative_pct_mentions"] = (100 * out["n_products"].cumsum() / out["n_products"].sum()).round(2)

    here = Path(__file__).resolve().parent
    out.to_csv(here / "ingredient_frequency.csv", index=False)

    print(f"Products: {n_products}  |  with no parseable ingredients: {n_empty}")
    print(f"Distinct raw ingredient names: {len(out)}")
    for k in (100, 200, 500):
        if len(out) >= k:
            print(f"Top {k} names cover {out['cumulative_pct_mentions'].iloc[k - 1]}% of all ingredient mentions")
    print(f"\nTop {args.top} ingredients:")
    print(out.head(args.top).to_string(index=False))

    if args.type_col in df.columns:
        rows = []
        for ptype, group in df.groupby(args.type_col):
            c = Counter()
            for ings in group["_ingredients"]:
                c.update(set(ings))
            for ing, n in c.items():
                rows.append({"product_type": ptype, "ingredient": ing, "n_products": n,
                             "pct_of_type": round(100 * n / len(group), 2)})
        by_type = pd.DataFrame(rows).sort_values(["product_type", "n_products"], ascending=[True, False])
        by_type.to_csv(here / "ingredient_frequency_by_type.csv", index=False)
        print(f"\nWrote per-type counts for: {sorted(df[args.type_col].unique())}")


if __name__ == "__main__":
    main()
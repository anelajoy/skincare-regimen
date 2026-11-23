"""Report how many products the unknown-ingredient screen excludes at 10%, 20% and 30% cutoffs.

Run from the project folder:
    .venv/bin/python experiments/unknown_screen_report.py
    .venv/bin/python experiments/unknown_screen_report.py --proxy-top 500

--proxy-top N pretends the N most common ingredient names are also "recognized". It is
only a stand-in until the canonical name list exists, so you can see what the cutoffs do
once most common names are known.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so "import regimen" works when running this file directly

import pandas as pd  # noqa: E402

from regimen.data.normalize import clean_name  # noqa: E402
from regimen.safety.knowledge import build_banned_terms, load_synonyms  # noqa: E402
from regimen.safety.screen import Screen  # noqa: E402

CUTOFFS = [0.10, 0.20, 0.30]
# the example user: allergic to fragrance, prescribed tretinoin
PROFILE = {"avoid_categories": ["fragrance"], "prescriptions": ["tretinoin"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-top", type=int, default=0)
    args = parser.parse_args()

    products = pd.read_parquet(ROOT / "data/processed/products_raw.parquet")
    synonyms = load_synonyms()
    extra = set()
    if args.proxy_top:
        freq = pd.read_csv(ROOT / "regimen/data/ingredient_frequency.csv")
        extra = {clean_name(n) for n in freq["ingredient"].head(args.proxy_top)}

    banned = build_banned_terms(PROFILE, synonyms)
    screen = Screen(banned, synonyms, extra_recognized=extra)
    infos = [screen.analyze(t) for t in products["ingredients_raw"]]
    us = products["us_sold"].to_numpy()

    label = f"proxy: top {args.proxy_top} names also recognized" if args.proxy_top else "synonym file only"
    print(f"Scenario: {label}")
    print(f"Example profile: {PROFILE}  ({len(banned)} banned terms)")
    print(f"Products: {len(products)} (US-tagged: {int(us.sum())})")
    print(f"Empty list: {sum(not i['names'] for i in infos)} | "
          f"names over 60 chars: {sum(bool(i['long_names']) for i in infos)} products")
    shares = pd.Series([len(i["unrecognized"]) / len(i["names"]) if i["names"] else 1.0 for i in infos])
    print(f"Share of names unrecognized per product: median {shares.median():.0%}, "
          f"10th-90th percentile {shares.quantile(.1):.0%}-{shares.quantile(.9):.0%}\n")

    rows = []
    for cutoff in CUTOFFS:
        decisions = [screen.decide(i, cutoff) for i in infos]
        for group, mask in (("all", [True] * len(infos)), ("US-tagged", list(us))):
            status = [d["status"] for d, m in zip(decisions, mask) if m]
            reasons = [d["reasons"] for d, m in zip(decisions, mask) if m]
            exact = sum(s == "EXCLUDE" and any(r.startswith("EXCLUDED") for r in rs) for s, rs in zip(status, reasons))
            rows.append({"scenario": label, "cutoff": f"{cutoff:.0%}", "group": group, "products": len(status),
                         "unparseable": status.count("UNPARSEABLE"),
                         "excluded_exact": exact,
                         "excluded_possible_only": status.count("EXCLUDE") - exact,
                         "kept": status.count("KEEP")})
    table = pd.DataFrame(rows)
    print(table.drop(columns="scenario").to_string(index=False))

    # which names triggered POSSIBLE_MATCH at the 20% cutoff? (so you can review false alarms)
    hits = Counter()
    for info, d in zip(infos, (screen.decide(i, 0.20) for i in infos)):
        if d["status"] != "UNPARSEABLE":
            hits.update(info["possible"])
    print(f"\nPOSSIBLE_MATCH names at 20% cutoff ({len(hits)} distinct, top 15):")
    for (name, term), n in hits.most_common(15):
        print(f"  {n:3d}  '{name}' resembles '{term}'")

    out = ROOT / "results" / "unknown_screen_report.csv"
    table.to_csv(out, mode="a", index=False, header=not out.exists())
    print(f"\nAppended summary to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

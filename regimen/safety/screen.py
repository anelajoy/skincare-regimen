"""Unknown-ingredient screen.

The problem: the synonym file does not know every ingredient name. A name we do not
know might still be a banned ingredient in disguise (a typo like "linalol", or a
glued-together name like "linalool tocopherol"). This screen handles that:

  1. A name is RECOGNIZED if it appears in the synonym file (as a variant or standard name).
  2. Every name is first turned into its standard name. If that is a banned term, the
     product is excluded (EXACT match).
  3. An UNRECOGNIZED name is also compared with every banned term in two ways:
        keyword rule: the banned term appears inside it as a whole word
        fuzzy rule:   it is a few typing mistakes away from the banned term
     A hit excludes the product with a POSSIBLE_MATCH reason.
  4. If too many names are unrecognized, the product is UNPARSEABLE (we cannot check it).

Everything here is plain code. No LLM is involved.
"""
import re

from regimen.data.normalize import ingredient_names

DEFAULT_CUTOFF = 0.20    # unparseable if MORE than this share of names is unrecognized
MAX_NAME_LEN = 60        # a longer "name" is probably an unsplit list or label text
MIN_SIMILARITY = 0.85    # fuzzy rule: how alike two names must be (1.0 = identical)
GENERIC_TERMS = {"alcohol"}  # too common inside other names ("cetyl alcohol"): skip the keyword rule


def edit_distance(a: str, b: str) -> int:
    """Number of single-letter inserts, deletes or changes needed to turn a into b."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1,                 # delete a letter
                               current[j - 1] + 1,              # insert a letter
                               previous[j - 1] + (ca != cb)))   # change a letter
        previous = current
    return previous[-1]


def is_close(a: str, b: str) -> bool:
    """True if a and b are at least MIN_SIMILARITY alike.
    Short names (under 7 letters) must match exactly, which avoids false alarms."""
    longest = max(len(a), len(b))
    allowed = int(longest * (1 - MIN_SIMILARITY) + 1e-9)  # most mistakes we tolerate
    if abs(len(a) - len(b)) > allowed:                    # quick reject, saves time
        return False
    return edit_distance(a, b) <= allowed


class Screen:
    def __init__(self, banned: dict, synonyms: dict, extra_recognized=()):
        self.banned = banned
        self.synonyms = synonyms
        self.recognized = set(synonyms) | set(synonyms.values()) | set(extra_recognized)
        # one whole-word pattern per banned term, built once
        self._keyword = [(t, re.compile(rf"(?<![a-z]){re.escape(t)}(?![a-z])"))
                         for t in banned if t not in GENERIC_TERMS]
        self._cache = {}

    def _look_for_banned(self, name: str):
        """Keyword rule, then fuzzy rule. Returns the banned term it resembles, or None."""
        if name not in self._cache:
            found = None
            for term, pattern in self._keyword:
                if pattern.search(name):
                    found = term
                    break
            if found is None:
                found = next((t for t in self.banned if is_close(name, t)), None)
            self._cache[name] = found
        return self._cache[name]

    def analyze(self, raw_text) -> dict:
        """Look at one product. The result does not depend on the cutoff."""
        names = ingredient_names(raw_text)
        exact, possible, unrecognized = [], [], []
        for name in names:
            standard = self.synonyms.get(name, name)
            if standard in self.banned:
                exact.append((name, standard))
            elif name not in self.recognized:
                unrecognized.append(name)
                term = self._look_for_banned(name)
                if term:
                    possible.append((name, term))
        return {"names": names, "unrecognized": unrecognized, "exact": exact, "possible": possible,
                "long_names": [n for n in names if len(n) > MAX_NAME_LEN]}

    def decide(self, info: dict, cutoff: float = DEFAULT_CUTOFF) -> dict:
        """Turn the analysis into a status (KEEP, EXCLUDE or UNPARSEABLE) plus reason codes."""
        if not info["names"]:
            return {"status": "UNPARSEABLE", "reasons": ["EMPTY_LIST"]}
        reasons = []
        if info["long_names"]:
            reasons.append(f"LONG_NAME: {len(info['long_names'])} name(s) over {MAX_NAME_LEN} characters")
        share = len(info["unrecognized"]) / len(info["names"])
        if share > cutoff:
            reasons.append(f"UNRECOGNIZED_SHARE: {share:.0%} of names unrecognized (cutoff {cutoff:.0%})")
        if reasons:
            return {"status": "UNPARSEABLE", "reasons": reasons}
        for name, term in info["exact"]:
            reasons.append(f"EXCLUDED: contains {term} ({self.banned[term]})")
        for name, term in info["possible"]:
            reasons.append(f"POSSIBLE_MATCH: '{name}' resembles '{term}' ({self.banned[term]})")
        return {"status": "EXCLUDE" if reasons else "KEEP", "reasons": reasons}

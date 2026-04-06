"""
parse_additives.py

Bug fixed: the original single-pass regex only captured the FIRST additive
code in a parenthetical group.  Multi-code groups were silently truncated.

Root cause
──────────
Original pattern:
    r'(?:E\s*-?\s*|INS\s*-?\s*|\()(\d{3,4}[A-Z]?)(?:\))?'

The \( branch matches an opening parenthesis and captures the single number
that immediately follows — then stops.  On "(322, 471)" it captures 322
only; the ", 471" segment has no E / INS / ( prefix so the pattern never
fires for it.  On "(501 (ii), 503 (ii), 500(ii))" only 501 is captured:
503 is preceded by ", " (no prefix), and 500 has "(ii)" AFTER it, not
before it.

Fix: two-phase extraction + plausibility filter
───────────────────────────────────────────────
Phase 1  Explicit-prefix forms — E330, E 330, E-330, e330, INS 330, etc.
         Unchanged; authoritative — if the label writer spelled it out with
         an E or INS prefix, we trust it unconditionally.

Phase 2  Parenthetical groups — (422), (322, 471), (501 (ii), 503 (ii), 500(ii)).
         A regex finds each parenthetical group allowing one level of inner
         nesting (to handle the "(ii)"/"(iii)" variant-suffix notation used
         by FSSAI labels).  Every 3–4-digit token inside the group is a
         candidate code, then filtered by _is_valid_additive_code().

         Plausibility filter:
           - Numeric part must be in the real E-number range 100–1525.
           - Letter suffix (if present) must be a–f, the only letters used
             in genuine sub-variant codes (150a, 472e, etc.).
           - Rejects unit strings like "100G" (grams), "500ML", "30%".

Deduplication preserves first-seen order across both phases.
"""

import re
from additives_db import ADDITIVES_DB


# ── Compiled patterns (module-level for efficiency) ──────────────────────────

# Phase 1: explicit E-number or INS-number prefix
# Handles: E330  E 330  E-330  e330  INS330  INS 330
_EXPLICIT_RE = re.compile(
    r'(?:E\s*-?\s*|INS\s*-?\s*)(\d{3,4}[A-Za-z]?)',
    re.IGNORECASE,
)

# Phase 2a: outer parenthetical group, one level of inner nesting allowed.
# Captures the full inner text so "(501 (ii), 503 (ii), 500(ii))" has its
# complete content available for the code scan below.
_PAREN_GROUP_RE = re.compile(
    r'\('
    r'('
    r'[^()]*'                   # text before the first inner paren (if any)
    r'(?:\([^()]*\)[^()]*)*'    # zero or more  (inner)  tail  repetitions
    r')'
    r'\)'
)

# Phase 2b: standalone 3–4-digit token inside a captured group.
_CODE_IN_GROUP_RE = re.compile(
    r'(?<!\d)'           # not preceded by a digit
    r'(\d{3,4}[A-Z]?)'  # 3–4 digit code with optional letter suffix
    r'(?!\d)',           # not followed by another digit
    re.IGNORECASE,
)


# ── Plausibility filter ───────────────────────────────────────────────────────

def _is_valid_additive_code(raw: str) -> bool:
    """
    Return True only if `raw` looks like a genuine E-number code.

    Rules applied to Phase 2 (parenthetical) captures only — Phase 1
    carries an explicit E / INS prefix and is trusted unconditionally.

      1. Numeric part must be in the real E-number range 100–1525.
      2. Letter suffix (if present) must be a–f, the only letters that
         appear in real E-number sub-variants (150a, 150b, 472e, etc.).
         Letters outside a–f are unit abbreviations:
           G → grams,  L → litres,  I → IU (international units)
         and those strings are rejected here.
    """
    m = re.match(r'^(\d{3,4})([a-fA-F]?)$', raw.strip())
    if not m:
        return False
    return 100 <= int(m.group(1)) <= 1525


# ─────────────────────────────────────────────────────────────────────────────

def parse_additives(ingredients_text: str) -> list:
    """
    Scan an ingredients string for E-numbers / INS-numbers.

    Returns a list of dicts: { code, name, safety, note }

    Handles all common label formats:
      E330, E 330, E-330, e330       — explicit E prefix
      INS 330, INS330                — INS prefix
      (330)                          — bare code in brackets
      (322, 471)                     — comma-separated list in brackets
      (501 (ii), 503 (ii), 500(ii))  — variant-suffix list (FSSAI style)
    """
    if not ingredients_text:
        return []

    text = ingredients_text.upper()
    found_codes: list[str] = []

    # ── Phase 1: explicit E / INS prefix forms ────────────────────────────────
    for m in _EXPLICIT_RE.finditer(text):
        found_codes.append(m.group(1))

    # ── Phase 2: parenthetical groups ────────────────────────────────────────
    for paren_m in _PAREN_GROUP_RE.finditer(text):
        group_content = paren_m.group(1)
        for code_m in _CODE_IN_GROUP_RE.finditer(group_content):
            candidate = code_m.group(1)
            if _is_valid_additive_code(candidate):
                found_codes.append(candidate)

    # ── Deduplicate (first-seen order) and look up DB ─────────────────────────
    seen:    set[str]   = set()
    results: list[dict] = []

    for raw_code in found_codes:
        code = raw_code.lower().strip()
        if code in seen:
            continue
        seen.add(code)

        entry = ADDITIVES_DB.get(code)
        if entry:
            results.append({
                "code":   f"E{code.upper()}",
                "name":   entry["name"],
                "safety": entry["safety"],   # "green" | "yellow" | "red"
                "note":   entry["note"],
            })
        else:
            results.append({
                "code":   f"E{code.upper()}",
                "name":   "Unknown Additive",
                "safety": "yellow",
                "note":   "Not found in database; verify independently.",
            })

    return results
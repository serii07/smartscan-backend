import re
from additives_db import ADDITIVES_DB

def parse_additives(ingredients_text: str) -> list:
    """
    Scans an ingredients string for E numbers and INS numbers.
    Returns a list of dicts: { code, name, safety, note }

    Matches patterns like:
      E330, E 330, E-330, e330       → standard E number
      INS 330, INS330                → INS format
      (330), (E330)                  → bracketed form common in Indian labels
    """
    if not ingredients_text:
        return []

    # Normalise to uppercase for matching
    text = ingredients_text.upper()

    # Regex: captures the numeric part (with optional letter suffix like 150a, 472e)
    pattern = r'(?:E\s*-?\s*|INS\s*-?\s*|\()(\d{3,4}[A-Z]?)(?:\))?'

    found_codes = re.findall(pattern, text)

    # Deduplicate while preserving order
    seen = set()
    results = []
    for raw_code in found_codes:
        code = raw_code.lower().strip()
        if code in seen:
            continue
        seen.add(code)

        entry = ADDITIVES_DB.get(code)
        if entry:
            results.append({
                "code": f"E{code.upper()}",
                "name": entry["name"],
                "safety": entry["safety"],   # "green" | "yellow" | "red"
                "note": entry["note"]
            })
        else:
            # Code not in our DB — return it as unknown
            results.append({
                "code": f"E{code.upper()}",
                "name": "Unknown Additive",
                "safety": "yellow",
                "note": "Not found in database; verify independently."
            })

    return results

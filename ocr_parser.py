"""
ocr_parser.py
The core parsing engine. Takes raw OCR text from Vision API and produces
structured, normalized nutrition data and ingredients.

Key challenges addressed:
1. OCR character substitution errors (l→1, O→0, rn→m, etc.)
2. Non-standardized FSSAI label formats
3. Per-serving vs per-100g conversion  ← FIX: column-index-aware, not global guess
4. Unit normalization (kJ→kcal, mg→g) ← FIX: unit inferred from field name too
5. Hindi/transliterated field names
6. Percentage RDA columns (we extract absolute values only) ← FIX: column detection
7. Malformed numbers (e.g. "25.7g" "25 7g" "25,7" all mean 25.7)
8. Ingredient tokenization preserving parenthetical sub-ingredients
9. Non-nutritional metadata lines filtered out  ← FIX: expanded skip patterns
10. Tabular fallback fires when < 3 nutrients found, not just on empty dict ← FIX
"""

import re
from typing import Optional
from difflib import SequenceMatcher
import logging

logger = logging.getLogger("smartscan.ocr_parser")


# ── OCR error correction ──────────────────────────────────────────────────────
# Map common OCR character substitutions in nutrition label context

OCR_CORRECTIONS = {
    # Numeric confusions
    r'\bO\b': '0',           # standalone O → 0
    r'\bl\b': '1',           # standalone l → 1
    r'(?<=\d)O(?=\D|$)': '0',  # digit+O → digit+0
    r'(?<=\d)l(?=\D|$)': '1',  # digit+l → digit+1
    r'(?<=\d)I(?=\D|$)': '1',  # digit+I → digit+1
    r'(?<=\d)S(?=\D|$)': '5',  # digit+S → digit+5 (rare but happens)
    r',(?=\d{1,2}\b)': '.',  # European decimal comma → dot (e.g. 25,7 → 25.7)

    # Common word OCR errors on nutrition labels
    r'Proteln\b': 'Protein',
    r'Protem\b': 'Protein',
    r'Carbohydrotes\b': 'Carbohydrates',
    r'Carbohydrales\b': 'Carbohydrates',
    r'Sodlum\b': 'Sodium',
    r'Calclum\b': 'Calcium',
    r'Calones\b': 'Calories',
    r'Eneray\b': 'Energy',
    r'Saturoled\b': 'Saturated',
    r'Monounsaturoted\b': 'Monounsaturated',
    r'Polyunsaturoted\b': 'Polyunsaturated',
    r'\bFots\b': 'Fats',
    r'\bFat s\b': 'Fats',
    r'Dietory\b': 'Dietary',
    r'Flbre\b': 'Fibre',
    r'\bFlber\b': 'Fiber',
    r'Vltomin\b': 'Vitamin',
    r'Mlnerols\b': 'Minerals',
    r'Tronsfatty\b': 'Trans fatty',
    r'Trens\b': 'Trans',
}

# ── Nutrient field name aliases ───────────────────────────────────────────────
# Maps every known variant (including Hindi transliterations) → canonical key

NUTRIENT_ALIASES = {
    # Energy
    "energy": "energy_kcal",
    "energy value": "energy_kcal",
    "calorific value": "energy_kcal",
    "calories": "energy_kcal",
    "caloric value": "energy_kcal",
    "total energy": "energy_kcal",
    "kcal": "energy_kcal",
    "energy (kcal)": "energy_kcal",
    "energy kcal": "energy_kcal",
    "urja": "energy_kcal",          # Hindi: ऊर्जा
    "urja (kcal)": "energy_kcal",

    # Energy kJ
    "energy (kj)": "energy_kj",
    "energy kj": "energy_kj",
    "kj": "energy_kj",

    # Protein
    "protein": "proteins_100g",
    "proteins": "proteins_100g",
    "total protein": "proteins_100g",
    "crude protein": "proteins_100g",
    "protein content": "proteins_100g",
    "proteen": "proteins_100g",      # transliteration variant
    "pranin": "proteins_100g",       # Hindi: प्रोटीन

    # Carbohydrates
    "carbohydrate": "carbohydrates_100g",
    "carbohydrates": "carbohydrates_100g",
    "total carbohydrate": "carbohydrates_100g",
    "total carbohydrates": "carbohydrates_100g",
    "carbs": "carbohydrates_100g",
    "available carbohydrate": "carbohydrates_100g",
    "karbohaidret": "carbohydrates_100g",
    "karbohaidrets": "carbohydrates_100g",
    "karbohydrate": "carbohydrates_100g",

    # Sugars
    "sugar": "sugars_100g",
    "sugars": "sugars_100g",
    "total sugar": "sugars_100g",
    "total sugars": "sugars_100g",
    "of which sugars": "sugars_100g",
    "of which: sugars": "sugars_100g",
    "chini": "sugars_100g",          # Hindi: चीनी
    "added sugar": "sugars_100g",
    "added sugars": "sugars_100g",

    # Fat
    "fat": "fat_100g",
    "fats": "fat_100g",
    "total fat": "fat_100g",
    "total fats": "fat_100g",
    "fat content": "fat_100g",
    "lipids": "fat_100g",
    "vasa": "fat_100g",              # Hindi: वसा

    # Saturated fat
    "saturated fat": "saturated-fat_100g",
    "saturated fats": "saturated-fat_100g",
    "saturated fatty acids": "saturated-fat_100g",
    "of which saturated": "saturated-fat_100g",
    "of which: saturated": "saturated-fat_100g",
    "saturates": "saturated-fat_100g",
    "sat fat": "saturated-fat_100g",
    "sat. fat": "saturated-fat_100g",

    # Trans fat
    "trans fat": "trans-fat_100g",
    "trans fats": "trans-fat_100g",
    "trans fatty acids": "trans-fat_100g",
    "of which trans": "trans-fat_100g",

    # Fibre
    "dietary fibre": "fiber_100g",
    "dietary fiber": "fiber_100g",
    "fibre": "fiber_100g",
    "fiber": "fiber_100g",
    "total dietary fibre": "fiber_100g",
    "total dietary fiber": "fiber_100g",
    "roughage": "fiber_100g",

    # Sodium / Salt
    "sodium": "sodium_100g",
    "salt": "salt_100g",
    "salt equivalent": "salt_100g",
    "namak": "salt_100g",            # Hindi: नमक

    # Cholesterol
    "cholesterol": "cholesterol_100g",
    "total cholesterol": "cholesterol_100g",

    # Minor nutrients (stored but not displayed in main table)
    "calcium": "calcium_100g",
    "iron": "iron_100g",
    "vitamin c": "vitamin-c_100g",
    "vitamin a": "vitamin-a_100g",
    "vitamin d": "vitamin-d_100g",
    "potassium": "potassium_100g",
    "magnesium": "magnesium_100g",
    "zinc": "zinc_100g",
}

# ── Unit conversion constants ─────────────────────────────────────────────────
KJ_TO_KCAL = 0.239006
MG_TO_G    = 0.001
MCG_TO_G   = 0.000001

# ── FIX #4: Non-nutritional line skip pattern ─────────────────────────────────
# Expanded set of metadata lines that must never be parsed as nutrient rows.
NON_NUTRITIONAL_SKIP = re.compile(
    r'^('
    r'nutrients?|nutrition\s+facts?|nutrition\s+info(rmation)?'
    r'|per\s+100|per\s+serving|per\s+portion|amount\s+per'
    r'|typical\s+values?|as\s+sold|as\s+prepared|as\s+consumed'
    r'|servings?\s+per\s+(pack|container|box|pouch|tin|bottle|can)'
    r'|number\s+of\s+servings?'
    r'|serving\s+size|portion\s+size|serve\s+size'
    r'|%\s*(rda|ri|dv|daily\s+value)|rda\s*%|%\s*ri'
    r'|fssai\s+(lic|license|reg|no|licen)'
    r'|mfg\.?|mfd\.?|mkd\.?|packed\s+by|manufactured\s+by|mkt\.?\s+by'
    r'|best\s+before|expiry|use\s+by|exp\.?'
    r'|store\s+in|storage|keep\s+refrigerated|keep\s+cool'
    r'|country\s+of\s+origin'
    r'|batch|lot\s+no|lic\.?\s*no|b\.?\s*no'
    r'|directions?\s+for\s+use|how\s+to\s+use'
    r'|contains?\s+added'                 # "Contains added vitamins" type lines
    r'|\*+\s*\w'                          # footnote lines starting with *
    r'|†\s*\w'                            # footnote lines starting with †
    r')',
    re.IGNORECASE
)


def _correct_ocr_errors(text: str) -> str:
    """Apply character-level and word-level OCR corrections."""
    for pattern, replacement in OCR_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _fuzzy_match_nutrient(raw_field: str) -> Optional[str]:
    """
    Find the best matching canonical nutrient key for a raw field name.
    Uses exact match first, then normalized match, then fuzzy similarity.
    Returns None if no match above confidence threshold.
    """
    normalized = raw_field.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[*†‡#]', '', normalized)
    normalized = normalized.rstrip(':').strip()

    # Exact match
    if normalized in NUTRIENT_ALIASES:
        return NUTRIENT_ALIASES[normalized]

    # Try without parenthetical content: "Energy (kcal)" → "energy"
    no_paren = re.sub(r'\(.*?\)', '', normalized).strip()
    if no_paren in NUTRIENT_ALIASES:
        return NUTRIENT_ALIASES[no_paren]

    # Try removing units if they crept into the field name
    no_units = re.sub(r'\b(g|mg|mcg|kcal|kj|ml|%)\b', '', normalized).strip()
    if no_units in NUTRIENT_ALIASES:
        return NUTRIENT_ALIASES[no_units]

    # Fuzzy match — only accept if similarity > 0.82
    best_score = 0.0
    best_key   = None
    for alias, canonical in NUTRIENT_ALIASES.items():
        score = SequenceMatcher(None, normalized, alias).ratio()
        if score > best_score:
            best_score = score
            best_key   = canonical

    if best_score >= 0.82:
        return best_key

    return None


def _parse_numeric_value(raw: str) -> Optional[float]:
    """
    Extract a numeric value from messy OCR output.
    Handles: "25.7g", "< 0.5", "Nil", "Trace", "25 7" (OCR space), "N/A"
    """
    raw = raw.strip()

    # Nil / Trace / N/A → 0.0
    if re.match(r'^(nil|none|trace|n\.?a\.?|not detected|nd|-)$', raw, re.IGNORECASE):
        return 0.0

    # Less than / greater than prefix — take the value as-is
    raw = re.sub(r'^[<>≤≥~approx\.]+\s*', '', raw)

    # Remove unit suffixes
    raw = re.sub(r'\s*(g|mg|mcg|μg|kcal|kj|kJ|ml|%|iu|IU)\s*$', '', raw, flags=re.IGNORECASE)

    # OCR sometimes puts spaces in numbers: "25 7" → "25.7" (if < 10 second part)
    raw = re.sub(r'(\d+)\s+(\d{1,2})$', r'\1.\2', raw)

    # Remove all non-numeric except dot
    cleaned = re.sub(r'[^\d.]', '', raw)

    # Handle multiple dots (OCR artifact)
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = parts[0] + '.' + ''.join(parts[1:])

    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _extract_serving_size(text: str) -> Optional[float]:
    """
    Extract the serving size in grams from the label text.
    Needed to normalize per-serving values to per-100g.
    """
    patterns = [
        r'serving\s+size\s*[:\-]?\s*([\d.]+)\s*g',
        r'per\s+serving\s*[:\-]?\s*([\d.]+)\s*g',
        r'per\s+portion\s*[:\-]?\s*([\d.]+)\s*g',
        r'([\d.]+)\s*g\s+per\s+serving',
        r'serve\s+size\s*[:\-]?\s*([\d.]+)\s*g',
        r'portion\s+size\s*[:\-]?\s*([\d.]+)\s*g',
        r'per\s+([\d.]+)\s*g\b',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = _parse_numeric_value(m.group(1))
            if val and 5 <= val <= 500:   # sanity check
                return val
    return None


def _detect_per_unit(text: str) -> str:
    """
    Determine if the table shows values per 100g or per serving.
    Returns 'per_100g' or 'per_serving'.

    NOTE: This is a coarse fallback used only when column-index detection
    cannot be applied (e.g. inline key-value format with a single column).
    For multi-column tables use _detect_per100g_column_index() instead.
    """
    text_lower = text.lower()

    per_100g_signals = [
        'per 100 g', 'per 100g', 'per100g',
        '/100g', '/100 g', 'per 100 ml', 'per 100ml',
        'per 100 gm', 'values per 100'
    ]
    per_serving_signals = [
        'per serving', 'per portion', 'per serve',
        'per pack', 'per packet', 'per sachet',
        'per biscuit', 'per piece', 'per unit'
    ]

    # Check which appears first — usually the primary column header
    first_100g    = min((text_lower.find(s) for s in per_100g_signals   if s in text_lower), default=9999)
    first_serving = min((text_lower.find(s) for s in per_serving_signals if s in text_lower), default=9999)

    if first_100g <= first_serving:
        return 'per_100g'
    return 'per_serving'


# ── FIX #1 & #2: Column-index-aware per-100g detection ───────────────────────

def _detect_per100g_column_index(lines: list) -> int:
    """
    Scan the first 20 lines for a column header row and determine which
    0-based numeric column index corresponds to per-100g values.

    Indian labels typically have columns in one of these orders:
        [per serving]  [per 100g]  [%RDA]   → index 1
        [per 100g]     [per serving]  [%RDA] → index 0
        [per 100g]     [%RDA]         → index 0

    Falls back to 0 (first numeric column) if undetectable.
    """
    per_100g_re  = re.compile(
        r'per\s*100\s*(g|ml|gm)|/\s*100\s*(g|ml)|values?\s*per\s*100',
        re.IGNORECASE
    )
    per_serv_re  = re.compile(
        r'per\s*(serving|portion|serve|pack|sachet|piece|unit|biscuit)',
        re.IGNORECASE
    )
    rda_re = re.compile(r'%\s*(rda|ri|dv)|rda', re.IGNORECASE)

    for line in lines[:20]:
        hits = []
        for label, pattern in [('serving', per_serv_re), ('100g', per_100g_re), ('rda', rda_re)]:
            m = pattern.search(line)
            if m:
                hits.append((label, m.start()))

        if not hits:
            continue

        # Need at least 2 column headers on the same line to determine order
        if len(hits) >= 2:
            hits.sort(key=lambda x: x[1])   # sort by character position in line
            for idx, (label, _) in enumerate(hits):
                if label == '100g':
                    return idx  # 0 = first numeric col, 1 = second, etc.

        # Only per_100g found alone on this header line — it's the only column
        if any(label == '100g' for label, _ in hits):
            return 0

    return 0   # safe default: always take the first numeric value


def _extract_numeric_columns(values_part: str) -> list:
    """
    Extract all (value, unit) pairs from the numeric portion of a line.
    Each match represents one table column.

    E.g. "5.2 g    2.6 g    10%" → [(5.2, 'g'), (2.6, 'g'), (10.0, '%')]

    Returns a list of (float, str) tuples.
    """
    col_pattern = re.compile(
        r'([<>≤≥~]?\s*[\d,.]+(?:\s\d{1,2})?)'    # number (optional OCR-space decimal)
        r'\s*(g|mg|mcg|μg|kcal|kj|kJ|ml|%|iu|IU)?',  # optional unit
        re.IGNORECASE
    )
    results = []
    for m in col_pattern.finditer(values_part):
        raw_num  = m.group(1)
        raw_unit = m.group(2) or ''
        val = _parse_numeric_value(raw_num)
        if val is not None:
            results.append((val, raw_unit.lower().strip()))
    return results


# ── FIX #5: Unit inference from field name ────────────────────────────────────

def _infer_unit_from_field(raw_field: str) -> str:
    """
    When the regex finds no unit after the numeric value, check if the
    unit was embedded in the field name instead.

    Handles patterns like:
        "Sodium (mg)"        → 'mg'
        "Calcium(mcg)"       → 'mcg'
        "Energy (kJ)"        → 'kj'
        "Protein g:"         → 'g'   (less common but seen on some labels)
    """
    # Parenthetical unit: "Sodium (mg)", "Energy (kJ)"
    m = re.search(r'\(\s*(g|mg|mcg|μg|kcal|kj|ml|iu)\s*\)', raw_field, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Unit as a standalone word at the end of the field name (no parens)
    # Only match mg/mcg/kj here — bare 'g' is too ambiguous
    m = re.search(r'\b(mg|mcg|μg|kcal|kj)\b', raw_field, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return ''


def _normalize_to_per_100g(value: float, unit: str, serving_g: Optional[float]) -> float:
    """Convert a per-serving value to per-100g."""
    if serving_g and serving_g > 0:
        value_per_g = value / serving_g
        return round(value_per_g * 100, 2)
    return value


def _convert_units(value: float, unit: str, canonical_key: str) -> float:
    """
    Convert all values to standard units:
    - Energy: always store as kcal (convert from kJ if needed)
    - Minerals: always store as g (convert from mg/mcg)
    """
    unit = unit.lower().strip() if unit else ''

    if canonical_key == 'energy_kj' or unit == 'kj':
        return round(value * KJ_TO_KCAL, 2)

    if unit in ('mg', 'milligrams', 'milligram'):
        return round(value * MG_TO_G, 4)

    if unit in ('mcg', 'μg', 'micrograms', 'microgram', 'ug'):
        return round(value * MCG_TO_G, 6)

    return value


def parse_nutrition_label(raw_text: str) -> dict:
    """
    Main nutrition label parser.
    Takes raw OCR text, returns a structured nutriments dict
    matching the same schema as OpenFoodFacts API responses.

    Output keys use OFF naming convention:
        energy-kcal_100g, proteins_100g, carbohydrates_100g,
        sugars_100g, fat_100g, saturated-fat_100g, fiber_100g,
        sodium_100g, salt_100g, etc.

    Fix summary applied here:
        #1 & #2 — column-index-aware value extraction (multi-column support)
        #3      — tabular fallback fires when < 3 nutrients found
        #4      — NON_NUTRITIONAL_SKIP filters metadata lines
        #5      — unit inferred from field name when not found after value
    """
    text  = _correct_ocr_errors(raw_text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Global per-unit context (used only for single-column inline labels)
    per_unit     = _detect_per_unit(text)
    serving_size = _extract_serving_size(text) if per_unit == 'per_serving' else None

    # FIX #1 & #2: Determine which numeric column holds per-100g values
    per100g_col_idx = _detect_per100g_column_index(lines)

    nutriments = {}

    # ── Strategy 1: line-by-line parsing ─────────────────────────────────────
    # Expected formats:
    #   "Protein    5.2 g    2.6 g    10%"   (multi-column table)
    #   "Energy: 100 kcal"                   (inline key-value)
    #   "Fat (g)    4.5    2.1    8%"        (unit in field name)

    for line in lines:

        # FIX #4: Skip non-nutritional metadata lines
        if NON_NUTRITIONAL_SKIP.match(line):
            continue

        # Skip lines that are only numbers (column separator rows in tables)
        if re.match(r'^[\d\s.,%]+$', line):
            continue

        # ── Split line at the first digit to separate field name from values ──
        first_digit = re.search(r'[<>≤≥~]?\d', line)
        if not first_digit:
            continue

        raw_field   = line[:first_digit.start()].strip()
        values_part = line[first_digit.start():]

        # Clean trailing colon / dash from field name (inline format)
        raw_field = raw_field.rstrip(':–-').strip()

        canonical = _fuzzy_match_nutrient(raw_field)
        if not canonical:
            continue

        # FIX #1 & #2: Extract all numeric columns, pick the right one
        all_cols = _extract_numeric_columns(values_part)
        if not all_cols:
            continue

        # Clamp index to available columns (label may have fewer columns than header)
        col_idx          = min(per100g_col_idx, len(all_cols) - 1)
        raw_value, raw_unit = all_cols[col_idx]

        # FIX #5: If no unit found after the value, check if it's in the field name
        if not raw_unit:
            raw_unit = _infer_unit_from_field(raw_field)

        # Skip RDA-only columns (unit is %) — we never want percentage values
        if raw_unit == '%':
            # Try adjacent columns for a real value
            real_cols = [(v, u) for v, u in all_cols if u != '%']
            if real_cols:
                # Among non-% columns, pick the one at or after per100g_col_idx
                raw_value, raw_unit = real_cols[min(per100g_col_idx, len(real_cols) - 1)]
            else:
                continue

        # Unit conversion (kJ→kcal, mg→g)
        value = _convert_units(raw_value, raw_unit, canonical)

        # Per-serving → per-100g normalization
        # Only apply when global context is per_serving AND we couldn't detect
        # a dedicated per-100g column (i.e. per100g_col_idx == 0 and label is
        # genuinely single-column per-serving)
        if per_unit == 'per_serving' and per100g_col_idx == 0 and canonical != 'energy_kj':
            value = _normalize_to_per_100g(value, raw_unit, serving_size)

        # Remap energy keys to the single canonical OFF key
        if canonical in ('energy_kj', 'energy_kcal'):
            canonical = 'energy-kcal_100g'

        # Sodium: also derive salt
        if canonical == 'sodium_100g':
            nutriments['salt_100g'] = round(value * 2.5, 3)
            nutriments[canonical]   = round(value, 3)
            continue

        nutriments[canonical] = round(value, 2)

    # ── FIX #3: Tabular fallback — fire when < 3 nutrients found, not just empty ─
    MIN_NUTRIENTS_THRESHOLD = 3
    if len(nutriments) < MIN_NUTRIENTS_THRESHOLD:
        tabular_result = _parse_tabular_format(lines, per100g_col_idx)
        # Only replace if tabular strategy found more nutrients
        if len(tabular_result) > len(nutriments):
            nutriments = tabular_result

    # ── Post-processing ───────────────────────────────────────────────────────

    # Derive salt from sodium if salt not present
    if 'sodium_100g' in nutriments and 'salt_100g' not in nutriments:
        nutriments['salt_100g'] = round(nutriments['sodium_100g'] * 2.5, 3)

    # Sanity check: remove obviously wrong values
    nutriments = _sanity_check_nutriments(nutriments)

    return nutriments


def _parse_tabular_format(lines: list, per100g_col_idx: int = 0) -> dict:
    """
    Fallback parser for tabular nutrition labels where field names and
    values appear on separate lines or in wide columns.

    Uses a two-pass approach: first collect field names, then value rows.
    Now accepts per100g_col_idx so it picks the correct column consistently
    with Strategy 1.

    FIX #3: Now also called when Strategy 1 finds fewer than 3 nutrients.
    FIX #1 & #2: Uses per100g_col_idx to pick the right column.
    """
    nutriments  = {}
    field_names = []
    value_groups = []

    for line in lines:
        # FIX #4: Skip metadata lines here too
        if NON_NUTRITIONAL_SKIP.match(line):
            continue

        nums = re.findall(r'[<>~]?[\d,.]+', line)
        if nums:
            # Line has numbers — treat as a value row
            vals = [_parse_numeric_value(n) for n in nums[:4]]  # max 4 columns
            valid_vals = [v for v in vals if v is not None]
            if valid_vals:
                value_groups.append(valid_vals)
            else:
                # Numbers present but all unparseable — insert placeholder so
                # field/value index pairing doesn't drift out of sync
                value_groups.append([None])
        else:
            words = line.strip()
            if len(words) > 2:
                field_names.append(words)

    # Pair field names with value rows (they should interleave in document order)
    for i, field in enumerate(field_names):
        canonical = _fuzzy_match_nutrient(field)
        if not canonical:
            continue
        if i >= len(value_groups):
            break

        row = value_groups[i]
        if row[0] is None:
            continue

        # FIX #1 & #2: Pick the correct column from this row
        col_idx = min(per100g_col_idx, len(row) - 1)
        val = row[col_idx]
        if val is None:
            continue

        # FIX #5: Check field name for embedded unit
        raw_unit = _infer_unit_from_field(field)

        val = _convert_units(val, raw_unit, canonical)

        if canonical in ('energy_kj', 'energy_kcal'):
            canonical = 'energy-kcal_100g'

        nutriments[canonical] = round(val, 2)

    return nutriments


def _sanity_check_nutriments(nutriments: dict) -> dict:
    """
    Remove values that are clearly wrong.
    These bounds are based on physical maximums for food.
    """
    bounds = {
        'energy-kcal_100g':   (0, 900),    # max is pure fat ~900 kcal/100g
        'proteins_100g':      (0, 100),
        'carbohydrates_100g': (0, 100),
        'sugars_100g':        (0, 100),
        'fat_100g':           (0, 100),
        'saturated-fat_100g': (0, 100),
        'trans-fat_100g':     (0, 10),
        'fiber_100g':         (0, 100),
        'sodium_100g':        (0, 40),     # pure NaCl is ~39g sodium/100g
        'salt_100g':          (0, 100),
        'cholesterol_100g':   (0, 5),
    }

    cleaned = {}
    for key, value in nutriments.items():
        if key in bounds:
            lo, hi = bounds[key]
            if lo <= value <= hi:
                cleaned[key] = value
            else:
                print(f"Sanity check failed: {key}={value} (bounds {lo}–{hi}), dropping")
        else:
            cleaned[key] = value  # keep unlisted keys as-is

    # Additional check: sugars <= carbs, sat fat <= fat
    if 'sugars_100g' in cleaned and 'carbohydrates_100g' in cleaned:
        if cleaned['sugars_100g'] > cleaned['carbohydrates_100g']:
            cleaned['sugars_100g'] = cleaned['carbohydrates_100g']

    if 'saturated-fat_100g' in cleaned and 'fat_100g' in cleaned:
        if cleaned['saturated-fat_100g'] > cleaned['fat_100g']:
            cleaned['saturated-fat_100g'] = cleaned['fat_100g']

    return cleaned


# ── Ingredients parser ────────────────────────────────────────────────────────

def parse_ingredients_label(raw_text: str) -> str:
    """
    Parse and clean an ingredients list from OCR text.

    Handles:
    - Removing non-ingredient header text ("INGREDIENTS:", "Contains:")
    - Preserving parenthetical sub-ingredient lists
    - Cleaning up OCR artifacts while preserving E-numbers
    - Normalizing separators
    - Removing allergen declarations (these go in the allergens field)
    - Removing percentage annotations
    """
    text = _correct_ocr_errors(raw_text)

    # ── Step 1: Find where ingredients actually start ─────────────────────────
    # Labels often have brand name, product name, etc. before ingredients
    start_patterns = [
        r'ingredients?\s*[:\-]',
        r'composition\s*[:\-]',
        r'made\s+from\s*[:\-]',
        r'contains?\s*[:\-]',
        r'manufactured\s+from\s*[:\-]',
        r'saamagri\s*[:\-]',    # Hindi: सामग्री
    ]
    for pattern in start_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            text = text[m.end():]
            break

    # ── Step 2: Find where ingredients end ───────────────────────────────────
    end_patterns = [
        r'allergen\s+information',
        r'allergy\s+advice',
        r'contains?\s+allergen',
        r'nutritional\s+information',
        r'nutrition\s+facts',
        r'best\s+before',
        r'manufactured\s+by',
        r'packed\s+by',
        r'fssai\s+lic',
        r'mkd\s+by',
        r'mfd\s+by',
        r'country\s+of\s+origin',
        r'net\s+(wt|weight|qty|quantity)',
        r'storage\s+instructions',
        r'directions\s+for\s+use',
    ]
    for pattern in end_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]

    # ── Step 3: Clean up the text ─────────────────────────────────────────────

    # Collapse newlines into spaces (ingredients often wrap across lines)
    text = re.sub(r'\n+', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove footnote markers that OCR picks up
    text = re.sub(r'\*+[^,)]*', '', text)
    text = re.sub(r'†[^,)]*', '', text)

    # Remove percentage annotations: "(55%)" but KEEP "(E330)" and "(Acidity Regulator)"
    text = re.sub(r'\(\s*\d+\.?\d*\s*%\s*\)', '', text)

    # Remove "May contain traces of..." disclaimers mixed into ingredients
    text = re.sub(r'may\s+contain\s+traces?\s+of[^.]*\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'may\s+contain\s*:?[^.]*\.?', '', text, flags=re.IGNORECASE)

    # ── Step 4: Normalize separators ─────────────────────────────────────────
    # Some labels use semicolons, some use periods between ingredients
    text = re.sub(r';\s*', ', ', text)

    # Periods followed by capital letter are likely sentence starts not separators
    # Periods followed by lowercase or digit are likely OCR artifacts
    text = re.sub(r'\.\s*(?=[a-z0-9])', ', ', text)

    # Clean up multiple commas
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'^\s*,\s*', '', text)   # leading comma
    text = re.sub(r',\s*$', '', text)       # trailing comma

    # Final normalize
    text = text.strip()

    # Capitalize first letter of each ingredient
    ingredients = [i.strip().capitalize() for i in text.split(',') if i.strip()]
    return ', '.join(ingredients)


def _fuzzy_product_name_match(query: str, candidates: list, threshold: float = 0.6) -> list:
    """
    Fuzzy match a query product name against a list of candidate names.
    Returns candidates sorted by similarity score, above threshold.

    Used for the "Did you mean...?" flow.
    """
    results = []
    query_lower = query.lower().strip()

    for candidate in candidates:
        name = candidate.get('product_name', '')
        if not name:
            continue

        name_lower = name.lower()

        # Token overlap score (handles word order differences)
        query_tokens     = set(re.findall(r'\b\w{3,}\b', query_lower))
        candidate_tokens = set(re.findall(r'\b\w{3,}\b', name_lower))
        if query_tokens and candidate_tokens:
            overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens), len(candidate_tokens))
        else:
            overlap = 0

        # String similarity
        sim = SequenceMatcher(None, query_lower, name_lower).ratio()

        # Combined score (token overlap weighted higher for product names)
        score = (overlap * 0.6) + (sim * 0.4)

        if score >= threshold:
            results.append({**candidate, '_match_score': round(score, 3)})

    results.sort(key=lambda x: x['_match_score'], reverse=True)
    return results[:5]   # return top 5 matches max


# ── Full OCR scan processor ───────────────────────────────────────────────────

def process_ocr_scan(raw_text: str, scan_type: str) -> dict:
    """
    Entry point called by the API endpoint.

    scan_type: 'nutrition' | 'ingredients'

    Returns:
    {
        "scan_type": "nutrition" | "ingredients",
        "success": bool,
        "data": dict | str,
        "confidence": float (0–1),
        "raw_text": str,          # for debugging
        "warnings": list[str]
    }
    """
    warnings = []
    logger.info("PARSER: start scan_type=%s raw_len=%s", scan_type, len(raw_text or ""))

    if not raw_text or len(raw_text.strip()) < 10:
        return {
            "scan_type": scan_type,
            "success": False,
            "data": None,
            "confidence": 0.0,
            "raw_text": raw_text or "",
            "warnings": ["OCR returned insufficient text. Ensure good lighting and camera focus."]
        }

    if scan_type == 'nutrition':
        logger.info("PARSER: nutrition branch entered")
        nutriments = parse_nutrition_label(raw_text)
        logger.info("PARSER: nutrition parsed keys=%s", list(nutriments.keys()))

        # Confidence: based on how many key nutrients we found
        key_nutrients = ['energy-kcal_100g', 'proteins_100g', 'carbohydrates_100g',
                         'fat_100g', 'sugars_100g']
        found_key  = sum(1 for k in key_nutrients if k in nutriments)
        confidence = found_key / len(key_nutrients)

        if confidence < 0.4:
            warnings.append(
                "Low confidence — less than half of the key nutrients were detected. "
                "Try rescanning with better lighting."
            )

        if 'energy-kcal_100g' not in nutriments:
            warnings.append("Energy value not detected.")


        logger.info("PARSER: nutrition confidence=%.2f success=%s", confidence, confidence > 0.2)
        return {
            "scan_type": "nutrition",
            "success": confidence > 0.2,
            "data": nutriments,
            "confidence": round(confidence, 2),
            "raw_text": raw_text,
            "warnings": warnings
        }

    elif scan_type == 'ingredients':
        logger.info("PARSER: ingredients branch entered")
        ingredients_text = parse_ingredients_label(raw_text)
        logger.info("PARSER: ingredients parsed len=%s", len(ingredients_text or ""))

        # Confidence: based on whether we found actual ingredient-like content
        has_commas     = ',' in ingredients_text
        reasonable_len = 10 < len(ingredients_text) < 5000
        has_no_junk    = not re.search(r'\d{6,}', ingredients_text)  # no phone/barcode numbers

        confidence = sum([has_commas, reasonable_len, has_no_junk]) / 3.0

        if not has_commas:
            warnings.append(
                "No comma-separated ingredients detected. "
                "The scan may not be pointed at an ingredients list."
            )
        logger.info("PARSER: ingredients confidence=%.2f success=%s", confidence, confidence > 0.3)
        return {
            "scan_type": "ingredients",
            "success": confidence > 0.3,
            "data": ingredients_text,
            "confidence": round(confidence, 2),
            "raw_text": raw_text,
            "warnings": warnings
        }
        logger.warning("PARSER: unknown scan_type=%s", scan_type)
    else:
        return {
            "scan_type": scan_type,
            "success": False,
            "data": None,
            "confidence": 0.0,
            "raw_text": raw_text,
            "warnings": [f"Unknown scan_type: {scan_type}"]
        }
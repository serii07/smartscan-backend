"""
ocr_parser.py
The core parsing engine.  Converts Vision API output → structured nutrition data.

Two-tier architecture
─────────────────────
Tier 1  (geometry-aware) — PRIMARY
    Accepts the word list produced by vision_service.extract_vision_data().
    Each word carries a pixel-accurate bounding box.  The parser:
      1. Groups words into visual rows by y-center proximity.
      2. Clusters numeric words into x-coordinate bands (= table columns).
      3. Identifies which band is "per 100 g" from header-row text.
      4. Pairs each nutrient label with the value in the correct band.
    This completely avoids the column-misalignment and index-drift bugs that
    plague text-only parsing.

Tier 2  (text-based) — FALLBACK
    The original regex / fuzzy-match parser.  Fires when:
      • No bounding-box data is available (legacy callers).
      • Tier 1 extracted fewer than MIN_NUTRIENTS_THRESHOLD nutrients.
    Kept intact so that ingredients parsing and any scan path that still
    provides only flat text continues to work.

Additional improvements over previous version
─────────────────────────────────────────────
  • Digit-merge detection  — "162" that should be "16.2" is caught by
    comparing each value against a per-nutrient expected median.
  • Per-field confidence   — every nutrient gets its own 0-1 score so the
    app can show exactly which values are uncertain.
  • process_ocr_scan()     — now accepts the vision_data dict from
    vision_service.extract_vision_data() as well as a plain string (legacy).
"""

import re
import logging
from typing import Optional, Union
from difflib import SequenceMatcher

logger = logging.getLogger("smartscan.ocr_parser")


# ─────────────────────────────────────────────────────────────────────────────
#  Constants and lookup tables
# ─────────────────────────────────────────────────────────────────────────────

MIN_NUTRIENTS_THRESHOLD = 3   # Tier 1 triggers Tier 2 fallback below this


# ── OCR character-substitution corrections ────────────────────────────────────
OCR_CORRECTIONS = {
    r'\bO\b':                       '0',
    r'\bl\b':                       '1',
    r'(?<=\d)O(?=\D|$)':            '0',
    r'(?<=\d)l(?=\D|$)':            '1',
    r'(?<=\d)I(?=\D|$)':            '1',
    r'(?<=\d)S(?=\D|$)':            '5',
    r',(?=\d{1,2}\b)':              '.',
    r'Proteln\b':                   'Protein',
    r'Protem\b':                    'Protein',
    r'Carbohydrotes\b':             'Carbohydrates',
    r'Carbohydrales\b':             'Carbohydrates',
    r'Sodlum\b':                    'Sodium',
    r'Calclum\b':                   'Calcium',
    r'Calones\b':                   'Calories',
    r'Eneray\b':                    'Energy',
    r'Saturoled\b':                 'Saturated',
    r'Monounsaturoted\b':           'Monounsaturated',
    r'Polyunsaturoted\b':           'Polyunsaturated',
    r'\bFots\b':                    'Fats',
    r'\bFat s\b':                   'Fats',
    r'Dietory\b':                   'Dietary',
    r'Flbre\b':                     'Fibre',
    r'\bFlber\b':                   'Fiber',
    r'Vltomin\b':                   'Vitamin',
    r'Mlnerols\b':                  'Minerals',
    r'Tronsfatty\b':                'Trans fatty',
    r'Trens\b':                     'Trans',
}


# ── Nutrient field-name aliases → canonical OFF key ──────────────────────────
NUTRIENT_ALIASES = {
    "energy": "energy_kcal", "energy value": "energy_kcal",
    "calorific value": "energy_kcal", "calories": "energy_kcal",
    "caloric value": "energy_kcal", "total energy": "energy_kcal",
    "kcal": "energy_kcal", "energy (kcal)": "energy_kcal",
    "energy kcal": "energy_kcal", "urja": "energy_kcal",
    "urja (kcal)": "energy_kcal",
    "energy (kj)": "energy_kj", "energy kj": "energy_kj", "kj": "energy_kj",
    "protein": "proteins_100g", "proteins": "proteins_100g",
    "total protein": "proteins_100g", "crude protein": "proteins_100g",
    "protein content": "proteins_100g", "proteen": "proteins_100g",
    "pranin": "proteins_100g",
    "carbohydrate": "carbohydrates_100g", "carbohydrates": "carbohydrates_100g",
    "total carbohydrate": "carbohydrates_100g",
    "total carbohydrates": "carbohydrates_100g",
    "carbs": "carbohydrates_100g",
    "available carbohydrate": "carbohydrates_100g",
    "karbohaidret": "carbohydrates_100g",
    "karbohaidrets": "carbohydrates_100g",
    "karbohydrate": "carbohydrates_100g",
    "sugar": "sugars_100g", "sugars": "sugars_100g",
    "total sugar": "sugars_100g", "total sugars": "sugars_100g",
    "of which sugars": "sugars_100g", "of which: sugars": "sugars_100g",
    "chini": "sugars_100g", "added sugar": "sugars_100g",
    "added sugars": "sugars_100g",
    "fat": "fat_100g", "fats": "fat_100g", "total fat": "fat_100g",
    "total fats": "fat_100g", "fat content": "fat_100g",
    "lipids": "fat_100g", "vasa": "fat_100g",
    "saturated fat": "saturated-fat_100g",
    "saturated fats": "saturated-fat_100g",
    "saturated fatty acids": "saturated-fat_100g",
    "of which saturated": "saturated-fat_100g",
    "of which: saturated": "saturated-fat_100g",
    "saturates": "saturated-fat_100g",
    "sat fat": "saturated-fat_100g", "sat. fat": "saturated-fat_100g",
    "trans fat": "trans-fat_100g", "trans fats": "trans-fat_100g",
    "trans fatty acids": "trans-fat_100g",
    "of which trans": "trans-fat_100g",
    "dietary fibre": "fiber_100g", "dietary fiber": "fiber_100g",
    "fibre": "fiber_100g", "fiber": "fiber_100g",
    "total dietary fibre": "fiber_100g", "total dietary fiber": "fiber_100g",
    "roughage": "fiber_100g",
    "sodium": "sodium_100g",
    "salt": "salt_100g", "salt equivalent": "salt_100g", "namak": "salt_100g",
    "cholesterol": "cholesterol_100g",
    "total cholesterol": "cholesterol_100g",
    "calcium": "calcium_100g", "iron": "iron_100g",
    "vitamin c": "vitamin-c_100g", "vitamin a": "vitamin-a_100g",
    "vitamin d": "vitamin-d_100g", "potassium": "potassium_100g",
    "magnesium": "magnesium_100g", "zinc": "zinc_100g",
}


# ── Unit conversion ───────────────────────────────────────────────────────────
KJ_TO_KCAL = 0.239006
MG_TO_G    = 0.001
MCG_TO_G   = 0.000001


# ── Non-nutritional line skip pattern ────────────────────────────────────────
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
    r'|contains?\s+added'
    r'|\*+\s*\w'
    r'|†\s*\w'
    r')',
    re.IGNORECASE,
)


# ── Per-nutrient expected medians (used for digit-merge detection) ────────────
_EXPECTED_MEDIANS: dict[str, float] = {
    "energy-kcal_100g":   350.0,
    "proteins_100g":        8.0,
    "carbohydrates_100g":  55.0,
    "sugars_100g":         12.0,
    "fat_100g":            10.0,
    "saturated-fat_100g":   4.0,
    "fiber_100g":           3.0,
    "sodium_100g":          0.5,
    "salt_100g":            1.2,
}


# ── Plausible value ranges for per-field confidence scoring ──────────────────
_PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "energy-kcal_100g":   (50.0,  600.0),
    "proteins_100g":       (0.0,   40.0),
    "carbohydrates_100g":  (0.0,   90.0),
    "sugars_100g":         (0.0,   60.0),
    "fat_100g":            (0.0,   60.0),
    "saturated-fat_100g":  (0.0,   30.0),
    "trans-fat_100g":      (0.0,    5.0),
    "fiber_100g":          (0.0,   20.0),
    "sodium_100g":         (0.0,    3.0),
    "salt_100g":           (0.0,    8.0),
    "cholesterol_100g":    (0.0,    1.0),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Shared utility functions  (used by both Tier 1 and Tier 2)
# ─────────────────────────────────────────────────────────────────────────────

def _correct_ocr_errors(text: str) -> str:
    for pattern, replacement in OCR_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _fuzzy_match_nutrient(raw_field: str) -> Optional[str]:
    """
    Map a raw OCR field name → canonical nutrient key.
    Tries exact match, then stripped match, then fuzzy (≥ 0.82 threshold).
    """
    normalized = re.sub(r'\s+', ' ', raw_field.lower().strip())
    normalized = re.sub(r'[*†‡#]', '', normalized).rstrip(':').strip()

    if normalized in NUTRIENT_ALIASES:
        return NUTRIENT_ALIASES[normalized]

    no_paren = re.sub(r'\(.*?\)', '', normalized).strip()
    if no_paren in NUTRIENT_ALIASES:
        return NUTRIENT_ALIASES[no_paren]

    no_units = re.sub(r'\b(g|mg|mcg|kcal|kj|ml|%)\b', '', normalized).strip()
    if no_units in NUTRIENT_ALIASES:
        return NUTRIENT_ALIASES[no_units]

    best_score, best_key = 0.0, None
    for alias, canonical in NUTRIENT_ALIASES.items():
        score = SequenceMatcher(None, normalized, alias).ratio()
        if score > best_score:
            best_score, best_key = score, canonical

    return best_key if best_score >= 0.82 else None


def _parse_numeric_value(raw: str) -> Optional[float]:
    """
    Extract a float from messy OCR text.
    Handles: "25.7g", "< 0.5", "Nil", "Trace", "25 7", "N/A", "25,7"
    """
    raw = raw.strip()
    if re.match(r'^(nil|none|trace|n\.?a\.?|not detected|nd|-)$', raw, re.IGNORECASE):
        return 0.0

    raw = re.sub(r'^[<>≤≥~approx\.]+\s*', '', raw)
    raw = re.sub(r'\s*(g|mg|mcg|μg|kcal|kj|kJ|ml|%|iu|IU)\s*$', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'(\d+)\s+(\d{1,2})$', r'\1.\2', raw)    # "25 7" → "25.7"
    cleaned = re.sub(r'[^\d.]', '', raw)

    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = parts[0] + '.' + ''.join(parts[1:])

    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _extract_serving_size(text: str) -> Optional[float]:
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
            if val and 5 <= val <= 500:
                return val
    return None


def _detect_per_unit(text: str) -> str:
    text_lower = text.lower()
    per_100g_signals  = ['per 100 g', 'per 100g', 'per100g', '/100g', '/100 g',
                         'per 100 ml', 'per 100ml', 'per 100 gm', 'values per 100']
    per_serv_signals  = ['per serving', 'per portion', 'per serve', 'per pack',
                         'per packet', 'per sachet', 'per biscuit', 'per piece', 'per unit']
    first_100g  = min((text_lower.find(s) for s in per_100g_signals  if s in text_lower), default=9999)
    first_serv  = min((text_lower.find(s) for s in per_serv_signals  if s in text_lower), default=9999)
    return 'per_100g' if first_100g <= first_serv else 'per_serving'


def _infer_unit_from_field(raw_field: str) -> str:
    m = re.search(r'\(\s*(g|mg|mcg|μg|kcal|kj|ml|iu)\s*\)', raw_field, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r'\b(mg|mcg|μg|kcal|kj)\b', raw_field, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return ''


def _convert_units(value: float, unit: str, canonical_key: str) -> float:
    unit = (unit or '').lower().strip()
    if canonical_key == 'energy_kj' or unit == 'kj':
        return round(value * KJ_TO_KCAL, 2)
    if unit in ('mg', 'milligrams', 'milligram'):
        return round(value * MG_TO_G, 4)
    if unit in ('mcg', 'μg', 'micrograms', 'microgram', 'ug'):
        return round(value * MCG_TO_G, 6)
    return value


def _normalize_to_per_100g(value: float, unit: str, serving_g: Optional[float]) -> float:
    if serving_g and serving_g > 0:
        return round((value / serving_g) * 100, 2)
    return value


def _extract_numeric_columns(values_part: str) -> list[tuple[float, str]]:
    """Extract all (value, unit) pairs from the numeric portion of a line."""
    col_pattern = re.compile(
        r'([<>≤≥~]?\s*[\d,.]+(?:\s\d{1,2})?)'
        r'\s*(g|mg|mcg|μg|kcal|kj|kJ|ml|%|iu|IU)?',
        re.IGNORECASE,
    )
    results = []
    for m in col_pattern.finditer(values_part):
        val = _parse_numeric_value(m.group(1))
        if val is not None:
            results.append((val, (m.group(2) or '').lower().strip()))
    return results


def _sanity_check_nutriments(nutriments: dict) -> dict:
    bounds = {
        'energy-kcal_100g':   (0, 900),
        'proteins_100g':      (0, 100),
        'carbohydrates_100g': (0, 100),
        'sugars_100g':        (0, 100),
        'fat_100g':           (0, 100),
        'saturated-fat_100g': (0, 100),
        'trans-fat_100g':     (0, 10),
        'fiber_100g':         (0, 100),
        'sodium_100g':        (0, 40),
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
                logger.warning(
                    "Sanity check failed: %s=%s (bounds %s–%s), dropping",
                    key, value, lo, hi,
                )
        else:
            cleaned[key] = value

    if 'sugars_100g' in cleaned and 'carbohydrates_100g' in cleaned:
        if cleaned['sugars_100g'] > cleaned['carbohydrates_100g']:
            cleaned['sugars_100g'] = cleaned['carbohydrates_100g']

    if 'saturated-fat_100g' in cleaned and 'fat_100g' in cleaned:
        if cleaned['saturated-fat_100g'] > cleaned['fat_100g']:
            cleaned['saturated-fat_100g'] = cleaned['fat_100g']

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
#  New: digit-merge detection and correction
# ─────────────────────────────────────────────────────────────────────────────

def _detect_and_fix_digit_merge(
    nutriments: dict,
    suspicion_factor: float = 8.0,
    plausible_factor: float = 3.0,
) -> tuple[dict, list[str]]:
    """
    Detect and correct OCR digit-merge errors where the decimal point was
    silently dropped (e.g. "16.2 g" OCR-read as "162 g").

    For each nutrient, if:
        value  >  median × suspicion_factor
    then we try inserting a decimal point after each possible split position
    in the integer representation.  The first split that produces a value
    ≤  median × plausible_factor is applied.

    Returns (fixed_nutriments, list_of_warning_strings).
    Both factors are tunable — defaults are deliberately conservative.
    """
    fixed    = dict(nutriments)
    warnings: list[str] = []

    for key, median in _EXPECTED_MEDIANS.items():
        if key not in fixed:
            continue
        val = fixed[key]
        if val <= median * suspicion_factor:
            continue   # looks plausible as-is

        # Try decimal insertion at every position (up to 3 digits from left)
        s = str(int(val))
        for split_pos in range(1, min(len(s), 4)):
            try:
                candidate = float(s[:split_pos] + '.' + s[split_pos:])
            except ValueError:
                continue
            if candidate <= median * plausible_factor:
                msg = (
                    f"Digit-merge corrected: {key}  "
                    f"{val} → {round(candidate, 2)}"
                )
                logger.warning(msg)
                warnings.append(msg)
                fixed[key] = round(candidate, 2)
                break

    return fixed, warnings


# ─────────────────────────────────────────────────────────────────────────────
#  New: per-field confidence scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_per_field_confidence(nutriments: dict) -> dict[str, float]:
    """
    Assign a 0.0–1.0 confidence score per nutrient.

    1.0  — value falls within the expected central range for packaged food.
    0.6  — value is zero for a nutrient that is almost never zero.
            (Zero for trans fat is fine; zero for carbohydrates is suspicious.)
    0.8  — value is zero but the field legitimately can be zero.
    0.4  — value is outside the expected central range.
    """
    _CRITICAL_NONZERO = {"energy-kcal_100g", "carbohydrates_100g", "fat_100g"}
    scores: dict[str, float] = {}

    for key, val in nutriments.items():
        if key not in _PLAUSIBLE_RANGES:
            scores[key] = 1.0
            continue

        lo, hi = _PLAUSIBLE_RANGES[key]

        if val == 0.0:
            scores[key] = 0.6 if key in _CRITICAL_NONZERO else 0.8
        elif lo <= val <= hi:
            scores[key] = 1.0
        else:
            scores[key] = 0.4

    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — Geometry-aware parser
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_line_height(words: list[dict]) -> float:
    """
    Estimate typical text line height from word bounding boxes.
    Uses the median word height — robust against a few very tall or tiny words.
    """
    heights = [w['y_max'] - w['y_min'] for w in words if w['y_max'] > w['y_min']]
    if not heights:
        return 20.0
    heights.sort()
    return heights[len(heights) // 2]


def _group_words_into_lines(
    words: list[dict], y_tolerance: float
) -> list[list[dict]]:
    """
    Group words into visual rows based on y-center proximity.

    Two words belong to the same row when the difference between their
    y-centers is ≤ y_tolerance (usually ~55 % of line height).

    Each row is returned sorted left-to-right by x_min.
    """
    if not words:
        return []

    sorted_words = sorted(
        words,
        key=lambda w: ((w['y_min'] + w['y_max']) / 2, w['x_min']),
    )

    lines: list[list[dict]] = []
    current_line = [sorted_words[0]]
    current_y = (sorted_words[0]['y_min'] + sorted_words[0]['y_max']) / 2

    for word in sorted_words[1:]:
        word_y = (word['y_min'] + word['y_max']) / 2
        if abs(word_y - current_y) <= y_tolerance:
            current_line.append(word)
            # Running mean keeps the reference y-center accurate as we add words
            current_y = (
                sum((w['y_min'] + w['y_max']) / 2 for w in current_line)
                / len(current_line)
            )
        else:
            lines.append(sorted(current_line, key=lambda w: w['x_min']))
            current_line = [word]
            current_y    = word_y

    lines.append(sorted(current_line, key=lambda w: w['x_min']))
    return lines


def _is_numeric_token(text: str) -> bool:
    """
    True for standalone numeric OCR tokens: "5.2", "162", "5.2g", "<0.5",
    "Nil", "Trace".  False for label words that happen to contain digits,
    e.g. "B12", "E330", "100g" (column header — ambiguous, excluded).
    """
    t = text.strip()
    if re.match(r'^(nil|none|trace|n\.?a\.?|nd|-)$', t, re.IGNORECASE):
        return True
    t2 = re.sub(r'^[<>≤≥~]+\s*', '', t)
    t2 = re.sub(r'\s*(g|mg|mcg|μg|kcal|kj|ml|%|iu)\s*$', '', t2, flags=re.IGNORECASE)
    return bool(re.match(r'^\d+([.,]\d+)?$', t2.strip()))


def _find_value_column_bands(
    words: list[dict], gap_threshold: float
) -> list[tuple[float, float]]:
    """
    Find x-coordinate bands that contain numeric tokens (the data columns).

    Only numeric words are used so that the wide label column does not
    pollute the clustering.  Each gap of > gap_threshold pixels between
    adjacent word x-centers starts a new band.

    Returns a list of (x_min, x_max) tuples sorted left-to-right.
    """
    numeric_words = [w for w in words if _is_numeric_token(w['text'])]
    if not numeric_words:
        return []

    x_centers = sorted((w['x_min'] + w['x_max']) / 2 for w in numeric_words)
    bands: list[tuple[float, float]] = []
    band_xs = [x_centers[0]]

    for x in x_centers[1:]:
        if x - max(band_xs) > gap_threshold:
            bands.append((min(band_xs), max(band_xs)))
            band_xs = [x]
        else:
            band_xs.append(x)

    bands.append((min(band_xs), max(band_xs)))
    return bands


def _identify_column_roles(
    lines: list[list[dict]],
    x_bands: list[tuple[float, float]],
) -> dict[int, str]:
    """
    Scan the first 15 lines for column headers.  For each band, reconstruct
    the text of words whose x-center falls inside the band (± padding) and
    match against known header patterns.

    Returns {band_index: role} where role ∈ {'per_100g', 'per_serving', 'rda'}.
    Bands with no recognisable header are omitted from the result.
    """
    PER_100G = re.compile(
        r'per\s*100\s*(g|ml|gm|gram)|/\s*100\s*(g|ml)|values?\s+per\s+100',
        re.IGNORECASE,
    )
    PER_SERV = re.compile(
        r'per\s+(serving|portion|serve|pack|sachet|piece|unit|biscuit)',
        re.IGNORECASE,
    )
    RDA_PAT  = re.compile(
        r'%\s*(rda|ri|dv)|rda\s*%|daily\s+value',
        re.IGNORECASE,
    )

    roles: dict[int, str] = {}
    padding = 25   # pixels — words may sit slightly outside their band

    for line in lines[:15]:
        band_texts: dict[int, list[str]] = {i: [] for i in range(len(x_bands))}

        for word in line:
            wx = (word['x_min'] + word['x_max']) / 2
            for i, (bx_min, bx_max) in enumerate(x_bands):
                if bx_min - padding <= wx <= bx_max + padding:
                    band_texts[i].append(word['text'])
                    break

        for i, word_list in band_texts.items():
            if not word_list or i in roles:
                continue
            band_str = ' '.join(word_list)
            if PER_100G.search(band_str):
                roles[i] = 'per_100g'
            elif PER_SERV.search(band_str):
                roles[i] = 'per_serving'
            elif RDA_PAT.search(band_str):
                roles[i] = 'rda'

    return roles


def _pick_per100g_band_index(
    x_bands: list[tuple[float, float]],
    column_roles: dict[int, str],
) -> int:
    """
    Choose the band index to use for per-100g values.

    Priority:
      1.  Band explicitly labelled 'per_100g' by header detection.
      2.  Heuristic based on number of bands and presence of other known roles:

          1 band  →  index 0  (trivial)
          2 bands, band-0 is 'per_serving'  →  index 1
          2 bands, band-1 is 'rda'          →  index 0
          2 bands, no roles identified       →  index 1 (Indian default: rightmost)
          3+ bands → rightmost non-rda band
    """
    # Explicit header match
    for idx, role in column_roles.items():
        if role == 'per_100g':
            return idx

    n = len(x_bands)
    rda_idx     = next((i for i, r in column_roles.items() if r == 'rda'),         None)
    serving_idx = next((i for i, r in column_roles.items() if r == 'per_serving'), None)

    if n == 1:
        return 0

    if n == 2:
        if serving_idx == 0:
            return 1
        if rda_idx == 1:
            return 0
        return 1   # Indian default: value column is rightmost

    # n >= 3: take rightmost non-rda band
    non_rda = [i for i in range(n) if column_roles.get(i) != 'rda']
    return max(non_rda) if non_rda else n - 2


def _words_to_text(words: list[dict]) -> str:
    """Join words left-to-right with single spaces."""
    return ' '.join(
        w['text'] for w in sorted(words, key=lambda w: w['x_min'])
    )


def parse_nutrition_from_vision_words(
    words: list[dict],
    fallback_text: str = "",
) -> tuple[dict, list[str]]:
    """
    Tier 1 geometry-aware nutrition parser.

    Parameters
    ----------
    words        : list of WordDict objects from vision_service.extract_vision_data()
    fallback_text: flat OCR string (used for serving-size detection and Tier 2 fallback)

    Returns
    -------
    (nutriments_dict, warnings_list)
    """
    warnings: list[str] = []

    if not words:
        logger.info("GEOM: no words supplied, falling to Tier 2")
        result = parse_nutrition_label(fallback_text) if fallback_text else {}
        return result, warnings

    # ── Step 1: Scale estimation ──────────────────────────────────────────────
    line_height = _estimate_line_height(words)
    y_tol = line_height * 0.55   # same-line tolerance
    x_gap = line_height * 2.0    # minimum inter-column whitespace

    # ── Step 2: Group words into visual rows ─────────────────────────────────
    lines = _group_words_into_lines(words, y_tolerance=y_tol)

    # ── Step 3: Find numeric column bands ────────────────────────────────────
    x_bands = _find_value_column_bands(words, gap_threshold=x_gap)

    if not x_bands:
        logger.warning("GEOM: no numeric bands detected, falling to Tier 2")
        warnings.append("No numeric columns detected — used text fallback parser.")
        result = parse_nutrition_label(fallback_text) if fallback_text else {}
        return result, warnings

    # ── Step 4: Identify column roles from headers ───────────────────────────
    column_roles  = _identify_column_roles(lines, x_bands)
    per100g_idx   = _pick_per100g_band_index(x_bands, column_roles)
    per100g_band  = x_bands[per100g_idx]

    logger.info(
        "GEOM: line_h=%.1f  bands=%d  roles=%s  per100g_idx=%d",
        line_height, len(x_bands), column_roles, per100g_idx,
    )

    # ── Step 5: Determine the label / value boundary ──────────────────────────
    # Everything left of the first numeric band is the nutrient-name column.
    label_x_max  = x_bands[0][0] - 10   # 10-px buffer before first band
    band_padding = 22                    # px tolerance for word → band assignment
    bx_min, bx_max = per100g_band

    # ── Step 6: Serving-size context (for labels that only show per-serving) ──
    per_unit     = _detect_per_unit(fallback_text) if fallback_text else 'per_100g'
    serving_size = (
        _extract_serving_size(fallback_text)
        if per_unit == 'per_serving' else None
    )

    # ── Step 7: Parse each visual row ────────────────────────────────────────
    nutriments: dict = {}

    for line in lines:
        line_text = _words_to_text(line)

        # Skip header / metadata lines
        if NON_NUTRITIONAL_SKIP.match(line_text.strip()):
            continue
        # Skip lines that are only numbers (pure-value rows with no label)
        if re.match(r'^[\d\s.,%<>]+$', line_text):
            continue

        # ── Split line into label words and per-100g value words ─────────────
        label_words = [
            w for w in line
            if w['x_max'] <= label_x_max + band_padding
        ]
        if not label_words:
            continue

        label_text = _words_to_text(label_words)

        # Apply OCR corrections to the label before matching
        label_text = _correct_ocr_errors(label_text)

        canonical = _fuzzy_match_nutrient(label_text)
        if canonical is None:
            continue

        # Collect value words in the per-100g band
        value_words = [
            w for w in line
            if (bx_min - band_padding
                <= (w['x_min'] + w['x_max']) / 2
                <= bx_max + band_padding)
            and w['x_min'] > label_x_max   # must be in value zone
        ]
        if not value_words:
            continue

        value_text = _words_to_text(value_words)

        # ── Parse numeric value and unit ──────────────────────────────────────
        all_cols = _extract_numeric_columns(value_text)
        if not all_cols:
            continue

        raw_value, raw_unit = all_cols[0]

        # If no unit after the value, check the label name (e.g. "Sodium (mg)")
        if not raw_unit:
            raw_unit = _infer_unit_from_field(label_text)

        # Skip RDA percentage columns
        if raw_unit == '%':
            continue

        value = _convert_units(raw_value, raw_unit, canonical)

        # Per-serving → per-100g only when the detected column is NOT already
        # a confirmed per-100g column from the header
        if per_unit == 'per_serving' and column_roles.get(per100g_idx) != 'per_100g':
            value = _normalize_to_per_100g(value, raw_unit, serving_size)

        # Normalise energy key
        if canonical in ('energy_kj', 'energy_kcal'):
            canonical = 'energy-kcal_100g'

        # Sodium → also derive salt
        if canonical == 'sodium_100g':
            nutriments['salt_100g'] = round(value * 2.5, 3)
            nutriments[canonical]   = round(value, 3)
            continue

        nutriments[canonical] = round(value, 2)

    # ── Step 8: Tier 2 fallback when Tier 1 found too few nutrients ──────────
    if len(nutriments) < MIN_NUTRIENTS_THRESHOLD and fallback_text:
        logger.info(
            "GEOM: only %d nutrients found — trying Tier 2 text parser",
            len(nutriments),
        )
        tier2 = parse_nutrition_label(fallback_text)
        if len(tier2) > len(nutriments):
            warnings.append(
                f"Geometry parser found only {len(nutriments)} nutrient(s); "
                "text fallback parser was used instead."
            )
            nutriments = tier2

    # ── Step 9: Post-processing ───────────────────────────────────────────────
    if 'sodium_100g' in nutriments and 'salt_100g' not in nutriments:
        nutriments['salt_100g'] = round(nutriments['sodium_100g'] * 2.5, 3)

    nutriments, merge_warnings = _detect_and_fix_digit_merge(nutriments)
    warnings.extend(merge_warnings)

    nutriments = _sanity_check_nutriments(nutriments)

    return nutriments, warnings


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — Text-based parser  (unchanged; kept as fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_per100g_column_index(lines: list) -> int:
    per_100g_re = re.compile(
        r'per\s*100\s*(g|ml|gm)|/\s*100\s*(g|ml)|values?\s*per\s*100',
        re.IGNORECASE,
    )
    per_serv_re = re.compile(
        r'per\s*(serving|portion|serve|pack|sachet|piece|unit|biscuit)',
        re.IGNORECASE,
    )
    rda_re = re.compile(r'%\s*(rda|ri|dv)|rda', re.IGNORECASE)

    for line in lines[:20]:
        hits = []
        for label, pattern in [('serving', per_serv_re), ('100g', per_100g_re), ('rda', rda_re)]:
            m = pattern.search(line)
            if m:
                hits.append((label, m.start()))
        if len(hits) >= 2:
            hits.sort(key=lambda x: x[1])
            for idx, (label, _) in enumerate(hits):
                if label == '100g':
                    return idx
        if any(label == '100g' for label, _ in hits):
            return 0

    return 0


def _extract_numeric_columns_tier2(values_part: str) -> list:
    col_pattern = re.compile(
        r'([<>≤≥~]?\s*[\d,.]+(?:\s\d{1,2})?)'
        r'\s*(g|mg|mcg|μg|kcal|kj|kJ|ml|%|iu|IU)?',
        re.IGNORECASE,
    )
    results = []
    for m in col_pattern.finditer(values_part):
        val = _parse_numeric_value(m.group(1))
        if val is not None:
            results.append((val, (m.group(2) or '').lower().strip()))
    return results


def parse_nutrition_label(raw_text: str) -> dict:
    """
    Tier 2 text-based parser.
    Direct entry point for:
      • Ingredients label parsing (which feeds parse_ingredients_label).
      • Tier 1 fallback when bounding-box data is unavailable or insufficient.
    """
    text  = _correct_ocr_errors(raw_text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    per_unit      = _detect_per_unit(text)
    serving_size  = _extract_serving_size(text) if per_unit == 'per_serving' else None
    per100g_col   = _detect_per100g_column_index(lines)

    nutriments: dict = {}

    for line in lines:
        if NON_NUTRITIONAL_SKIP.match(line):
            continue
        if re.match(r'^[\d\s.,%]+$', line):
            continue

        first_digit = re.search(r'[<>≤≥~]?\d', line)
        if not first_digit:
            continue

        raw_field   = line[:first_digit.start()].strip().rstrip(':–-').strip()
        values_part = line[first_digit.start():]

        canonical = _fuzzy_match_nutrient(raw_field)
        if not canonical:
            continue

        all_cols = _extract_numeric_columns(values_part)
        if not all_cols:
            continue

        col_idx              = min(per100g_col, len(all_cols) - 1)
        raw_value, raw_unit  = all_cols[col_idx]

        if not raw_unit:
            raw_unit = _infer_unit_from_field(raw_field)

        if raw_unit == '%':
            real_cols = [(v, u) for v, u in all_cols if u != '%']
            if real_cols:
                raw_value, raw_unit = real_cols[min(per100g_col, len(real_cols) - 1)]
            else:
                continue

        value = _convert_units(raw_value, raw_unit, canonical)

        if per_unit == 'per_serving' and per100g_col == 0 and canonical != 'energy_kj':
            value = _normalize_to_per_100g(value, raw_unit, serving_size)

        if canonical in ('energy_kj', 'energy_kcal'):
            canonical = 'energy-kcal_100g'

        if canonical == 'sodium_100g':
            nutriments['salt_100g'] = round(value * 2.5, 3)
            nutriments[canonical]   = round(value, 3)
            continue

        nutriments[canonical] = round(value, 2)

    if len(nutriments) < MIN_NUTRIENTS_THRESHOLD:
        tabular = _parse_tabular_format(lines, per100g_col)
        if len(tabular) > len(nutriments):
            nutriments = tabular

    if 'sodium_100g' in nutriments and 'salt_100g' not in nutriments:
        nutriments['salt_100g'] = round(nutriments['sodium_100g'] * 2.5, 3)

    return _sanity_check_nutriments(nutriments)


def _parse_tabular_format(lines: list, per100g_col_idx: int = 0) -> dict:
    nutriments  = {}
    field_names = []
    value_groups: list[list] = []

    for line in lines:
        if NON_NUTRITIONAL_SKIP.match(line):
            continue
        nums = re.findall(r'[<>~]?[\d,.]+', line)
        if nums:
            vals = [_parse_numeric_value(n) for n in nums[:4]]
            valid = [v for v in vals if v is not None]
            value_groups.append(valid if valid else [None])
        else:
            if len(line.strip()) > 2:
                field_names.append(line.strip())

    for i, field in enumerate(field_names):
        canonical = _fuzzy_match_nutrient(field)
        if not canonical or i >= len(value_groups):
            continue
        row = value_groups[i]
        if not row or row[0] is None:
            continue
        col_idx = min(per100g_col_idx, len(row) - 1)
        val     = row[col_idx]
        if val is None:
            continue
        raw_unit = _infer_unit_from_field(field)
        val      = _convert_units(val, raw_unit, canonical)
        if canonical in ('energy_kj', 'energy_kcal'):
            canonical = 'energy-kcal_100g'
        nutriments[canonical] = round(val, 2)

    return nutriments


# ─────────────────────────────────────────────────────────────────────────────
#  Ingredients parser  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def parse_ingredients_label(raw_text: str) -> str:
    text = _correct_ocr_errors(raw_text)

    for pattern in [
        r'ingredients?\s*[:\-]', r'composition\s*[:\-]',
        r'made\s+from\s*[:\-]', r'contains?\s*[:\-]',
        r'manufactured\s+from\s*[:\-]', r'saamagri\s*[:\-]',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            text = text[m.end():]
            break

    for pattern in [
        r'allergen\s+information', r'allergy\s+advice',
        r'contains?\s+allergen', r'nutritional\s+information',
        r'nutrition\s+facts', r'best\s+before', r'manufactured\s+by',
        r'packed\s+by', r'fssai\s+lic', r'mkd\s+by', r'mfd\s+by',
        r'country\s+of\s+origin', r'net\s+(wt|weight|qty|quantity)',
        r'storage\s+instructions', r'directions\s+for\s+use',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\*+[^,)]*', '', text)
    text = re.sub(r'†[^,)]*', '', text)
    text = re.sub(r'\(\s*\d+\.?\d*\s*%\s*\)', '', text)
    text = re.sub(r'may\s+contain\s+traces?\s+of[^.]*\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'may\s+contain\s*:?[^.]*\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r';\s*', ', ', text)
    text = re.sub(r'\.\s*(?=[a-z0-9])', ', ', text)
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'^\s*,\s*', '', text)
    text = re.sub(r',\s*$', '', text)
    text = text.strip()

    ingredients = [i.strip().capitalize() for i in text.split(',') if i.strip()]
    return ', '.join(ingredients)


# ─────────────────────────────────────────────────────────────────────────────
#  Product-name fuzzy matching  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _fuzzy_product_name_match(
    query: str, candidates: list, threshold: float = 0.6
) -> list:
    results = []
    query_lower = query.lower().strip()
    for candidate in candidates:
        name = candidate.get('product_name', '')
        if not name:
            continue
        name_lower    = name.lower()
        q_tokens      = set(re.findall(r'\b\w{3,}\b', query_lower))
        c_tokens      = set(re.findall(r'\b\w{3,}\b', name_lower))
        overlap       = (
            len(q_tokens & c_tokens) / max(len(q_tokens), len(c_tokens))
            if q_tokens and c_tokens else 0
        )
        sim   = SequenceMatcher(None, query_lower, name_lower).ratio()
        score = overlap * 0.6 + sim * 0.4
        if score >= threshold:
            results.append({**candidate, '_match_score': round(score, 3)})
    results.sort(key=lambda x: x['_match_score'], reverse=True)
    return results[:5]


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_ocr_scan(
    vision_data_or_text: Union[dict, str],
    scan_type: str,
) -> dict:
    """
    Orchestrator called by the API endpoint.

    Parameters
    ----------
    vision_data_or_text:
        dict  — output of vision_service.extract_vision_data()
                Enables Tier 1 geometry parser (preferred).
        str   — raw OCR text (legacy path; Tier 2 only).

    scan_type:
        'nutrition'   — run nutrition label parser, return nutriments dict.
        'ingredients' — run ingredients parser, return cleaned string.

    Returns
    -------
    {
        "scan_type"       : str,
        "success"         : bool,
        "data"            : dict | str | None,
        "confidence"      : float,       # 0–1, overall
        "field_confidence": dict,        # per-nutrient scores (nutrition only)
        "raw_text"        : str,
        "warnings"        : list[str],
    }
    """
    warnings: list[str] = []

    # ── Normalise input ───────────────────────────────────────────────────────
    if isinstance(vision_data_or_text, dict):
        vision_data = vision_data_or_text
        raw_text    = vision_data.get('text', '')
        words       = vision_data.get('words', [])
        if not vision_data.get('success', True) and vision_data.get('error'):
            warnings.append(f"Vision API error: {vision_data['error']}")
    else:
        # Legacy: plain string passed directly
        raw_text = vision_data_or_text or ''
        words    = []

    logger.info(
        "PARSER: scan_type=%s  raw_len=%d  words=%d",
        scan_type, len(raw_text), len(words),
    )

    # ── Minimum viability check ───────────────────────────────────────────────
    if not raw_text.strip() and not words:
        return {
            "scan_type":        scan_type,
            "success":          False,
            "data":             None,
            "confidence":       0.0,
            "field_confidence": {},
            "raw_text":         raw_text,
            "warnings":         ["OCR returned no text. Check lighting and camera focus."],
        }

    # ─────────────────────────────────────────────────────────────────────────
    if scan_type == 'nutrition':

        if words:
            # ── Tier 1: geometry-aware ────────────────────────────────────────
            nutriments, tier1_warnings = parse_nutrition_from_vision_words(
                words, fallback_text=raw_text
            )
            warnings.extend(tier1_warnings)
        else:
            # ── Tier 2: text-only (legacy path) ──────────────────────────────
            nutriments = parse_nutrition_label(raw_text)
            warnings.append(
                "No bounding-box data available — text-only parser used. "
                "Update vision_service to extract_vision_data() for better accuracy."
            )

        logger.info("PARSER: nutriments=%s", list(nutriments.keys()))

        # ── Confidence ────────────────────────────────────────────────────────
        KEY_NUTRIENTS = [
            'energy-kcal_100g', 'proteins_100g', 'carbohydrates_100g',
            'fat_100g', 'sugars_100g',
        ]
        found_key       = sum(1 for k in KEY_NUTRIENTS if k in nutriments)
        overall_conf    = found_key / len(KEY_NUTRIENTS)
        field_conf      = _compute_per_field_confidence(nutriments)

        if overall_conf < 0.4:
            warnings.append(
                "Low confidence — fewer than half the key nutrients were "
                "detected. Try rescanning with better lighting or a flatter label."
            )
        if 'energy-kcal_100g' not in nutriments:
            warnings.append("Energy value not detected.")

        logger.info(
            "PARSER: confidence=%.2f  success=%s",
            overall_conf, overall_conf > 0.2,
        )

        return {
            "scan_type":        "nutrition",
            "success":          overall_conf > 0.2,
            "data":             nutriments,
            "confidence":       round(overall_conf, 2),
            "field_confidence": field_conf,
            "raw_text":         raw_text,
            "warnings":         warnings,
        }

    # ─────────────────────────────────────────────────────────────────────────
    elif scan_type == 'ingredients':

        ingredients_text = parse_ingredients_label(raw_text)
        logger.info("PARSER: ingredients len=%d", len(ingredients_text or ''))

        has_commas    = ',' in ingredients_text
        reasonable    = 10 < len(ingredients_text) < 5000
        no_junk       = not re.search(r'\d{6,}', ingredients_text)
        confidence    = sum([has_commas, reasonable, no_junk]) / 3.0

        if not has_commas:
            warnings.append(
                "No comma-separated ingredients detected — "
                "the scan may not be pointing at an ingredients list."
            )

        logger.info(
            "PARSER: ingredients confidence=%.2f  success=%s",
            confidence, confidence > 0.3,
        )

        return {
            "scan_type":        "ingredients",
            "success":          confidence > 0.3,
            "data":             ingredients_text,
            "confidence":       round(confidence, 2),
            "field_confidence": {},
            "raw_text":         raw_text,
            "warnings":         warnings,
        }

    # ─────────────────────────────────────────────────────────────────────────
    else:
        return {
            "scan_type":        scan_type,
            "success":          False,
            "data":             None,
            "confidence":       0.0,
            "field_confidence": {},
            "raw_text":         raw_text,
            "warnings":         [f"Unknown scan_type: {scan_type!r}"],
        }
"""
ocr_parser.py  — PATCHED
Fixes applied (search for "# FIX" to locate every change):

  BUG 1  _pick_per100g_band_index() wrong default
         Old code blindly returned the rightmost band when no header was
         detected ("Indian default").  Many Indian labels are structured as
         [per 100g | %RDA], so the rightmost band is the %RDA column.
         Fix: sample actual numeric values from each candidate band; pick the
         band whose median lands in the per-100g plausible range.

  BUG 2  _identify_column_roles() misses headers
         Old code required each header word's x-centre to fall within 25 px
         of a numeric-data band.  Header rows typically span the full table
         width, so their words never hit that tolerance → roles = {} always.
         Fix: for each candidate header line, join ALL words on the line and
         run the header patterns against the full line string.  Only then
         try to assign the matched role to the nearest band.

  BUG 3  label_x_max computed from band edge, not from actual label words
         x_bands[0][0] − 10 is far too small when band-0 is the per-serving
         column (sits near the centre of the label).  Long nutrient names
         get truncated → fuzzy match fails → row is skipped.
         Fix: compute label_x_max per-row from the actual rightmost x_max
         of words that are clearly label words (left of every numeric band).

  BUG 4  No post-parse plausibility check / column-swap retry
         When the wrong band is chosen the values fail _sanity_check and get
         dropped silently.  The caller never tries the other bands.
         Fix: add _is_result_plausible() and _try_alternate_bands().  After
         the initial parse, if the result looks implausible we iterate over
         every other band index and keep the best-scoring result.

  BUG 5  Tier-2 _detect_per100g_column_index() defaults to 0
         When no header pattern matches the function returns 0, which is the
         per-serving column on the most common [serving | per 100g] layout.
         Fix: if no explicit header is found, sample the first numeric value
         in each column from the first 10 data rows.  The column with the
         larger mean is the per-100g column (energy ~350 vs. ~40 per serve).
"""

import re
import logging
from typing import Optional, Union
from difflib import SequenceMatcher

logger = logging.getLogger("smartscan.ocr_parser")


# ─────────────────────────────────────────────────────────────────────────────
#  Constants and lookup tables  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

MIN_NUTRIENTS_THRESHOLD = 3

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

KJ_TO_KCAL = 0.239006
MG_TO_G    = 0.001
MCG_TO_G   = 0.000001

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

_PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "energy-kcal_100g":   (50.0,  900.0),
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
#  Shared utility functions  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _correct_ocr_errors(text: str) -> str:
    for pattern, replacement in OCR_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _fuzzy_match_nutrient(raw_field: str) -> Optional[str]:
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
    raw = raw.strip()
    if re.match(r'^(nil|none|trace|n\.?a\.?|not detected|nd|-)$', raw, re.IGNORECASE):
        return 0.0

    raw = re.sub(r'^[<>≤≥~approx\.]+\s*', '', raw)
    raw = re.sub(r'\s*(g|mg|mcg|μg|kcal|kj|kJ|ml|%|iu|IU)\s*$', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'(\d+)\s+(\d{1,2})$', r'\1.\2', raw)
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
#  Digit-merge detection  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_and_fix_digit_merge(
    nutriments: dict,
    suspicion_factor: float = 8.0,
    plausible_factor: float = 3.0,
) -> tuple[dict, list[str]]:
    fixed    = dict(nutriments)
    warnings: list[str] = []

    for key, median in _EXPECTED_MEDIANS.items():
        if key not in fixed:
            continue
        val = fixed[key]
        if val <= median * suspicion_factor:
            continue

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
#  Per-field confidence scoring  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_per_field_confidence(nutriments: dict) -> dict[str, float]:
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
#  Tier 1 helpers — geometry  (mostly unchanged; see BUG comments)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_line_height(words: list[dict]) -> float:
    heights = [w['y_max'] - w['y_min'] for w in words if w['y_max'] > w['y_min']]
    if not heights:
        return 20.0
    heights.sort()
    return heights[len(heights) // 2]


def _group_words_into_lines(
    words: list[dict], y_tolerance: float
) -> list[list[dict]]:
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
    t = text.strip()
    if re.match(r'^(nil|none|trace|n\.?a\.?|nd|-)$', t, re.IGNORECASE):
        return True
    t2 = re.sub(r'^[<>≤≥~]+\s*', '', t)
    t2 = re.sub(r'\s*(g|mg|mcg|μg|kcal|kj|ml|%|iu)\s*$', '', t2, flags=re.IGNORECASE)
    return bool(re.match(r'^\d+([.,]\d+)?$', t2.strip()))


def _find_value_column_bands(
    words: list[dict], gap_threshold: float
) -> list[tuple[float, float]]:
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


# ─────────────────────────────────────────────────────────────────────────────
# FIX BUG 2 — _identify_column_roles  (rewritten)
# ─────────────────────────────────────────────────────────────────────────────

def _identify_column_roles(
    lines: list[list[dict]],
    x_bands: list[tuple[float, float]],
) -> dict[int, str]:
    """
    Scan the first 25 lines for column headers.

    FIX: The old approach required each header word's x-centre to fall within
    25 px of a numeric band.  Header rows on Indian labels often span the full
    table width, so their words' x-centres never hit that tolerance.

    New approach:
      1. Reconstruct the full text of each candidate header line.
      2. Run the header regex against the FULL line text.
      3. If a match is found, find the word that triggered the match and use
         ITS x-centre to identify the nearest band.
      4. Fall back to a coarse left/right split when no individual word can be
         matched to a band (e.g. a single header like "Per 100 g / Per Serving").
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

    for line in lines[:25]:
        line_text = ' '.join(w['text'] for w in sorted(line, key=lambda w: w['x_min']))

        for pattern, role in [(PER_100G, 'per_100g'), (PER_SERV, 'per_serving'), (RDA_PAT, 'rda')]:
            m = pattern.search(line_text)
            if not m or role in roles.values():
                continue

            # Find the word closest to the centre of the regex match
            # within the original line.  Use its x-centre to pick the band.
            match_char_mid = (m.start() + m.end()) / 2
            char_offset    = 0
            best_word      = None
            best_dist      = float('inf')
            for word in sorted(line, key=lambda w: w['x_min']):
                word_mid_char = char_offset + len(word['text']) / 2
                dist = abs(word_mid_char - match_char_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_word = word
                char_offset += len(word['text']) + 1  # +1 for the space

            if best_word is None:
                continue

            wx = (best_word['x_min'] + best_word['x_max']) / 2
            # Find the nearest band to this word's x-centre
            nearest_idx = min(
                range(len(x_bands)),
                key=lambda i: abs((x_bands[i][0] + x_bands[i][1]) / 2 - wx),
            )
            # Only accept if within a generous distance (2× band width or 80 px)
            band_centre  = (x_bands[nearest_idx][0] + x_bands[nearest_idx][1]) / 2
            band_width   = x_bands[nearest_idx][1] - x_bands[nearest_idx][0]
            tolerance    = max(80, band_width * 2)
            if abs(band_centre - wx) <= tolerance and nearest_idx not in roles:
                roles[nearest_idx] = role

    return roles


# ─────────────────────────────────────────────────────────────────────────────
# FIX BUG 1 — helpers for value-based band disambiguation
# ─────────────────────────────────────────────────────────────────────────────

def _sample_band_values(
    lines: list[list[dict]],
    band: tuple[float, float],
    all_bands_x_min: float,
    band_padding: float = 22,
    max_lines: int = 40,
) -> list[float]:
    """
    Return up to max_lines numeric values found in the given x-band.
    Only rows that also have a recognisable label (i.e. words left of
    all_bands_x_min) are included, so pure header rows don't pollute the sample.
    """
    bx_min, bx_max = band
    values: list[float] = []

    for line in lines[:max_lines]:
        has_label = any(w['x_max'] <= all_bands_x_min + band_padding for w in line)
        if not has_label:
            continue

        value_words = [
            w for w in line
            if bx_min - band_padding
               <= (w['x_min'] + w['x_max']) / 2
               <= bx_max + band_padding
            and w['x_min'] > all_bands_x_min - band_padding
        ]
        if not value_words:
            continue

        text = ' '.join(w['text'] for w in sorted(value_words, key=lambda w: w['x_min']))
        cols = _extract_numeric_columns(text)
        if cols:
            v, unit = cols[0]
            if unit != '%':
                values.append(v)

    return values


def _score_band_as_per100g(values: list[float]) -> float:
    """
    Score how likely a list of sampled values is to be the per-100g column.

    Heuristic: the per-100g column typically contains energy (50–900 kcal),
    fat/carbs in the 0–90 g range, and protein in the 0–40 g range.
    Per-serving values are the same numbers scaled down by ~4–10×.

    We use the *median* so that a single huge outlier (e.g. a batch number
    that slipped through) doesn't dominate.
    Returns a score in [0, 1]; higher = more likely per-100g.
    """
    if not values:
        return 0.0

    values_sorted = sorted(values)
    median = values_sorted[len(values_sorted) // 2]

    # A median between 5 and 800 is consistent with per-100g macro values
    if 5 <= median <= 800:
        # Prefer medians around the typical packaged-food macro range
        if 10 <= median <= 600:
            return 1.0
        return 0.7
    # Very small median (< 5) → likely per-serving or RDA fraction
    if median < 5:
        return 0.1
    # Very large median (> 800) → might be kJ or a mis-read value
    return 0.3


# ─────────────────────────────────────────────────────────────────────────────
# FIX BUG 1 — _pick_per100g_band_index  (rewritten)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_per100g_band_index(
    x_bands: list[tuple[float, float]],
    column_roles: dict[int, str],
    lines: Optional[list[list[dict]]] = None,
    label_x_max: float = 0.0,
) -> int:
    """
    Choose the band index to use for per-100g values.

    Priority:
      1. Band explicitly labelled 'per_100g' by header detection.
      2. Value-sampling heuristic: sample numeric values from each candidate
         band and pick the one whose median lands in the per-100g range.
         This handles the case where the header detector found no roles.
      3. Structural heuristic as a final fallback (was the only logic before).

    FIX: Removed the unconditional "Indian default: rightmost" fallback that
    was causing systematic column swaps on [per 100g | %RDA] layouts.
    """
    # 1. Explicit header match
    for idx, role in column_roles.items():
        if role == 'per_100g':
            return idx

    # Candidate bands = all bands that are NOT identified as RDA
    rda_indices   = {i for i, r in column_roles.items() if r == 'rda'}
    serv_indices  = {i for i, r in column_roles.items() if r == 'per_serving'}
    candidates    = [i for i in range(len(x_bands)) if i not in rda_indices]

    if not candidates:
        candidates = list(range(len(x_bands)))

    # 2. Value-sampling heuristic (FIX BUG 1)
    if lines and len(candidates) > 1:
        all_bands_x_min = x_bands[0][0]
        scores: dict[int, float] = {}
        for idx in candidates:
            sampled = _sample_band_values(
                lines, x_bands[idx], all_bands_x_min
            )
            scores[idx] = _score_band_as_per100g(sampled)
            logger.debug(
                "BAND %d: sample=%s  score=%.2f", idx, sampled[:5], scores[idx]
            )

        best_idx   = max(candidates, key=lambda i: scores[i])
        best_score = scores[best_idx]

        # Only trust the heuristic if its score is clearly higher than the
        # runner-up; otherwise fall through to structural rules.
        other_scores = [scores[i] for i in candidates if i != best_idx]
        runner_up    = max(other_scores) if other_scores else 0.0
        if best_score > runner_up + 0.15 or best_score >= 0.9:
            logger.info(
                "GEOM: band %d selected by value-sampling (score=%.2f)",
                best_idx, best_score,
            )
            return best_idx

    # 3. Structural fallback
    n = len(x_bands)
    if n == 1:
        return 0

    # If the serving column was explicitly identified, the other one is per-100g
    if len(serv_indices) == 1:
        non_serv = [i for i in candidates if i not in serv_indices]
        if non_serv:
            return non_serv[0]

    # Last resort: prefer the leftmost non-RDA candidate because
    # [per 100g | %RDA] is the more common Indian layout
    return candidates[0]


def _words_to_text(words: list[dict]) -> str:
    return ' '.join(
        w['text'] for w in sorted(words, key=lambda w: w['x_min'])
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIX BUG 4 — post-parse plausibility check + column-swap retry
# ─────────────────────────────────────────────────────────────────────────────

def _is_result_plausible(nutriments: dict) -> bool:
    """
    Return True if the parsed nutriments look like real food data.

    We require at least two of the five key macros to be present AND in their
    plausible per-100g ranges.  A single macro in range is not enough because
    even a wrong column occasionally produces one value that passes.
    """
    KEY_MACROS = [
        'energy-kcal_100g', 'proteins_100g',
        'carbohydrates_100g', 'fat_100g', 'sugars_100g',
    ]
    in_range = 0
    for key in KEY_MACROS:
        if key not in nutriments:
            continue
        lo, hi = _PLAUSIBLE_RANGES.get(key, (0, 1e9))
        if lo < nutriments[key] <= hi:
            in_range += 1

    return in_range >= 2


def _extract_nutriments_for_band(
    lines: list[list[dict]],
    x_bands: list[tuple[float, float]],
    per100g_idx: int,
    per_unit: str,
    serving_size: Optional[float],
    column_roles: dict[int, str],
) -> dict:
    """
    Run the row-extraction loop for a specific per100g_idx.
    Extracted from parse_nutrition_from_vision_words so it can be called
    for each candidate band during the column-swap retry (FIX BUG 4).
    """
    band_padding = 22
    bx_min, bx_max = x_bands[per100g_idx]

    # label_x_max: left edge of the first band minus a small buffer
    # (FIX BUG 3 — see inline comment inside the loop)
    first_band_left = x_bands[0][0]

    nutriments: dict = {}

    for line in lines:
        line_text = _words_to_text(line)
        if NON_NUTRITIONAL_SKIP.match(line_text.strip()):
            continue
        if re.match(r'^[\d\s.,%<>]+$', line_text):
            continue

        # FIX BUG 3: compute label boundary per-row from actual label words.
        # A label word is one whose x_max is clearly left of the first numeric
        # band.  We add a generous buffer so that long names aren't truncated.
        label_words_candidate = [
            w for w in line if w['x_max'] <= first_band_left + 10
        ]
        if not label_words_candidate:
            continue

        # Use the rightmost x_max of candidate label words as the boundary
        computed_label_x_max = max(w['x_max'] for w in label_words_candidate)
        label_text = _words_to_text(label_words_candidate)
        label_text = _correct_ocr_errors(label_text)

        canonical = _fuzzy_match_nutrient(label_text)
        if canonical is None:
            continue

        # Value words: in the target band and to the right of the label
        value_words = [
            w for w in line
            if (bx_min - band_padding
                <= (w['x_min'] + w['x_max']) / 2
                <= bx_max + band_padding)
            and w['x_min'] > computed_label_x_max
        ]
        if not value_words:
            continue

        value_text = _words_to_text(value_words)
        all_cols   = _extract_numeric_columns(value_text)
        if not all_cols:
            continue

        raw_value, raw_unit = all_cols[0]
        if not raw_unit:
            raw_unit = _infer_unit_from_field(label_text)
        if raw_unit == '%':
            continue

        value = _convert_units(raw_value, raw_unit, canonical)

        if per_unit == 'per_serving' and column_roles.get(per100g_idx) != 'per_100g':
            value = _normalize_to_per_100g(value, raw_unit, serving_size)

        if canonical in ('energy_kj', 'energy_kcal'):
            canonical = 'energy-kcal_100g'

        if canonical == 'sodium_100g':
            nutriments['salt_100g'] = round(value * 2.5, 3)
            nutriments[canonical]   = round(value, 3)
            continue

        nutriments[canonical] = round(value, 2)

    return nutriments


def _try_alternate_bands(
    lines: list[list[dict]],
    x_bands: list[tuple[float, float]],
    initial_idx: int,
    initial_nutriments: dict,
    per_unit: str,
    serving_size: Optional[float],
    column_roles: dict[int, str],
) -> tuple[dict, int, list[str]]:
    """
    FIX BUG 4: If initial result is implausible, try every other band index
    and return the best-scoring result.

    'Best' = passes _is_result_plausible() AND has the most plausible nutrients.
    """
    warnings: list[str] = []

    if _is_result_plausible(initial_nutriments):
        return initial_nutriments, initial_idx, warnings

    best_nutriments = initial_nutriments
    best_idx        = initial_idx
    best_count      = len(initial_nutriments)

    for try_idx in range(len(x_bands)):
        if try_idx == initial_idx:
            continue

        candidate = _extract_nutriments_for_band(
            lines, x_bands, try_idx, per_unit, serving_size, column_roles
        )
        candidate = _sanity_check_nutriments(candidate)

        if _is_result_plausible(candidate) and len(candidate) > best_count:
            best_nutriments = candidate
            best_idx        = try_idx
            best_count      = len(candidate)

    if best_idx != initial_idx:
        msg = (
            f"Column-swap: band {initial_idx} gave implausible values; "
            f"switched to band {best_idx}."
        )
        logger.warning(msg)
        warnings.append(msg)

    return best_nutriments, best_idx, warnings


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — Geometry-aware parser  (updated to use fixed helpers above)
# ─────────────────────────────────────────────────────────────────────────────

def parse_nutrition_from_vision_words(
    words: list[dict],
    fallback_text: str = "",
) -> tuple[dict, list[str]]:
    warnings: list[str] = []

    if not words:
        logger.info("GEOM: no words supplied, falling to Tier 2")
        result = parse_nutrition_label(fallback_text) if fallback_text else {}
        return result, warnings

    line_height = _estimate_line_height(words)
    y_tol = line_height * 0.55
    x_gap = line_height * 2.0

    lines   = _group_words_into_lines(words, y_tolerance=y_tol)
    x_bands = _find_value_column_bands(words, gap_threshold=x_gap)

    if not x_bands:
        logger.warning("GEOM: no numeric bands detected, falling to Tier 2")
        warnings.append("No numeric columns detected — used text fallback parser.")
        result = parse_nutrition_label(fallback_text) if fallback_text else {}
        return result, warnings

    # FIX BUG 2: use updated _identify_column_roles
    column_roles = _identify_column_roles(lines, x_bands)

    # FIX BUG 1: pass lines so _pick_per100g_band_index can sample values
    first_band_left = x_bands[0][0]
    per100g_idx     = _pick_per100g_band_index(
        x_bands, column_roles,
        lines=lines, label_x_max=first_band_left,
    )

    logger.info(
        "GEOM: line_h=%.1f  bands=%d  roles=%s  per100g_idx=%d",
        line_height, len(x_bands), column_roles, per100g_idx,
    )

    per_unit     = _detect_per_unit(fallback_text) if fallback_text else 'per_100g'
    serving_size = (
        _extract_serving_size(fallback_text)
        if per_unit == 'per_serving' else None
    )

    # FIX BUG 3: row-level label_x_max computed inside _extract_nutriments_for_band
    nutriments = _extract_nutriments_for_band(
        lines, x_bands, per100g_idx, per_unit, serving_size, column_roles
    )

    # FIX BUG 4: validate result; try other bands if implausible
    nutriments = _sanity_check_nutriments(nutriments)
    nutriments, per100g_idx, swap_warnings = _try_alternate_bands(
        lines, x_bands, per100g_idx, nutriments,
        per_unit, serving_size, column_roles,
    )
    warnings.extend(swap_warnings)

    # Tier 2 fallback when Tier 1 still found too few nutrients
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

    if 'sodium_100g' in nutriments and 'salt_100g' not in nutriments:
        nutriments['salt_100g'] = round(nutriments['sodium_100g'] * 2.5, 3)

    nutriments, merge_warnings = _detect_and_fix_digit_merge(nutriments)
    warnings.extend(merge_warnings)

    nutriments = _sanity_check_nutriments(nutriments)

    return nutriments, warnings


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — Text-based parser
# ─────────────────────────────────────────────────────────────────────────────

# FIX BUG 5 — _detect_per100g_column_index: smarter default when no header found
def _detect_per100g_column_index(lines: list) -> int:
    """
    Detect which column (0-based) in the Tier-2 split text holds per-100g data.

    FIX BUG 5: The old code returned 0 when no header was found.  That is
    wrong for the very common [per serving | per 100g] layout where the
    per-100g data is in column 1.

    New strategy:
      1. Scan header lines for explicit column labels (unchanged).
      2. If no header is found, sample the first numeric value in each column
         from up to 10 data rows.  The column with the larger mean is more
         likely the per-100g column (energy ~350 kcal vs. ~35 kcal per serve).
    """
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

    # FIX BUG 5: value-sampling heuristic when no header was found
    col_sums: dict[int, float] = {}
    col_counts: dict[int, int] = {}
    data_rows_checked = 0

    for line in lines:
        if NON_NUTRITIONAL_SKIP.match(line.strip()):
            continue
        first_digit = re.search(r'[<>≤≥~]?\d', line)
        if not first_digit:
            continue

        values_part = line[first_digit.start():]
        cols = _extract_numeric_columns(values_part)
        if len(cols) < 2:
            continue

        for col_idx, (val, unit) in enumerate(cols[:4]):
            if unit == '%':
                continue
            col_sums[col_idx]   = col_sums.get(col_idx, 0.0)   + val
            col_counts[col_idx] = col_counts.get(col_idx, 0)    + 1

        data_rows_checked += 1
        if data_rows_checked >= 10:
            break

    if col_counts:
        col_means = {
            idx: col_sums[idx] / col_counts[idx]
            for idx in col_counts
        }
        # The column with the higher mean is the per-100g column
        best_col = max(col_means, key=lambda i: col_means[i])
        logger.info(
            "TIER2: no header found; column means=%s → using col %d",
            col_means, best_col,
        )
        return best_col

    # True last-resort default: 0 (unchanged from original)
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
    text  = _correct_ocr_errors(raw_text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    per_unit      = _detect_per_unit(text)
    serving_size  = _extract_serving_size(text) if per_unit == 'per_serving' else None
    per100g_col   = _detect_per100g_column_index(lines)   # FIX BUG 5 applied here

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
#  Main entry point  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def process_ocr_scan(
    vision_data_or_text: Union[dict, str],
    scan_type: str,
) -> dict:
    warnings: list[str] = []

    if isinstance(vision_data_or_text, dict):
        vision_data = vision_data_or_text
        raw_text    = vision_data.get('text', '')
        words       = vision_data.get('words', [])
        if not vision_data.get('success', True) and vision_data.get('error'):
            warnings.append(f"Vision API error: {vision_data['error']}")
    else:
        raw_text = vision_data_or_text or ''
        words    = []

    logger.info(
        "PARSER: scan_type=%s  raw_len=%d  words=%d",
        scan_type, len(raw_text), len(words),
    )

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

    if scan_type == 'nutrition':

        if words:
            nutriments, tier1_warnings = parse_nutrition_from_vision_words(
                words, fallback_text=raw_text
            )
            warnings.extend(tier1_warnings)
        else:
            nutriments = parse_nutrition_label(raw_text)
            warnings.append(
                "No bounding-box data available — text-only parser used. "
                "Update vision_service to extract_vision_data() for better accuracy."
            )

        logger.info("PARSER: nutriments=%s", list(nutriments.keys()))

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
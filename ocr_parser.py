"""ocr_parser.py

Updated OCR parsing pipeline for nutrition labels and ingredients.

Design goals:
- Use Vision OCR only as the text engine.
- Reconstruct table structure from coordinates.
- Prefer line/column geometry over raw flattened text.
- Validate values with nutrient rules and confidence scoring.
- Keep ingredients parsing simpler and safer.
- Fall back to text-only parsing when structured OCR is unavailable.

Expected OCR input:
- Google Vision full response dict (preferred), or
- raw OCR text string (fallback compatibility).

Main entrypoint:
- process_ocr_scan(raw_ocr, scan_type)

Output schema:
{
    "scan_type": "nutrition" | "ingredients",
    "success": bool,
    "data": dict | str | None,
    "confidence": float,
    "raw_text": str,
    "warnings": list[str]
}
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("smartscan.ocr_parser")

# ---------------------------------------------------------------------------
# OCR corrections
# ---------------------------------------------------------------------------

OCR_CORRECTIONS = {
    # Numeric confusions
    r"\bO\b": "0",
    r"\bl\b": "1",
    r"(?<=\d)O(?=\D|$)": "0",
    r"(?<=\d)l(?=\D|$)": "1",
    r"(?<=\d)I(?=\D|$)": "1",
    r"(?<=\d)S(?=\D|$)": "5",
    r",(?=\d{1,2}\b)": ".",

    # Common label OCR mistakes
    r"Proteln\b": "Protein",
    r"Protem\b": "Protein",
    r"Carbohydrotes\b": "Carbohydrates",
    r"Carbohydrales\b": "Carbohydrates",
    r"Sodlum\b": "Sodium",
    r"Calclum\b": "Calcium",
    r"Calones\b": "Calories",
    r"Eneray\b": "Energy",
    r"Saturoled\b": "Saturated",
    r"Monounsaturoted\b": "Monounsaturated",
    r"Polyunsaturoted\b": "Polyunsaturated",
    r"\bFots\b": "Fats",
    r"\bFat s\b": "Fats",
    r"Dietory\b": "Dietary",
    r"Flbre\b": "Fibre",
    r"\bFlber\b": "Fiber",
    r"Vltomin\b": "Vitamin",
    r"Mlnerols\b": "Minerals",
    r"Tronsfatty\b": "Trans fatty",
    r"Trens\b": "Trans",
}

# ---------------------------------------------------------------------------
# Nutrient aliases
# ---------------------------------------------------------------------------

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
    "urja": "energy_kcal",
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
    "proteen": "proteins_100g",
    "pranin": "proteins_100g",

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
    "chini": "sugars_100g",
    "added sugar": "sugars_100g",
    "added sugars": "sugars_100g",

    # Fat
    "fat": "fat_100g",
    "fats": "fat_100g",
    "total fat": "fat_100g",
    "total fats": "fat_100g",
    "fat content": "fat_100g",
    "lipids": "fat_100g",
    "vasa": "fat_100g",

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

    # Fibre / fiber
    "dietary fibre": "fiber_100g",
    "dietary fiber": "fiber_100g",
    "fibre": "fiber_100g",
    "fiber": "fiber_100g",
    "total dietary fibre": "fiber_100g",
    "total dietary fiber": "fiber_100g",
    "roughage": "fiber_100g",

    # Sodium / salt
    "sodium": "sodium_100g",
    "salt": "salt_100g",
    "salt equivalent": "salt_100g",
    "namak": "salt_100g",

    # Cholesterol
    "cholesterol": "cholesterol_100g",
    "total cholesterol": "cholesterol_100g",

    # Other nutrients
    "calcium": "calcium_100g",
    "iron": "iron_100g",
    "vitamin c": "vitamin-c_100g",
    "vitamin a": "vitamin-a_100g",
    "vitamin d": "vitamin-d_100g",
    "potassium": "potassium_100g",
    "magnesium": "magnesium_100g",
    "zinc": "zinc_100g",
}

# ---------------------------------------------------------------------------
# Constants / regex patterns
# ---------------------------------------------------------------------------

KJ_TO_KCAL = 0.239006
MG_TO_G = 0.001
MCG_TO_G = 0.000001

MIN_NUTRIENTS_THRESHOLD = 3
KEY_NUTRIENTS = [
    "energy-kcal_100g",
    "proteins_100g",
    "carbohydrates_100g",
    "fat_100g",
    "sugars_100g",
]

NON_NUTRITIONAL_SKIP = re.compile(
    r"^("
    r"nutrients?|nutrition\s+facts?|nutrition\s+info(rmation)?"
    r"|per\s+100|per\s+serving|per\s+portion|amount\s+per"
    r"|typical\s+values?|as\s+sold|as\s+prepared|as\s+consumed"
    r"|servings?\s+per\s+(pack|container|box|pouch|tin|bottle|can)"
    r"|number\s+of\s+servings?"
    r"|serving\s+size|portion\s+size|serve\s+size"
    r"|%\s*(rda|ri|dv|daily\s+value)|rda\s*%|%\s*ri"
    r"|fssai\s+(lic|license|reg|no|licen)"
    r"|mfg\.?|mfd\.?|mkd\.?|packed\s+by|manufactured\s+by|mkt\.?\s+by"
    r"|best\s+before|expiry|use\s+by|exp\.?"
    r"|store\s+in|storage|keep\s+refrigerated|keep\s+cool"
    r"|country\s+of\s+origin"
    r"|batch|lot\s+no|lic\.?\s*no|b\.?\s*no"
    r"|directions?\s+for\s+use|how\s+to\s+use"
    r"|contains?\s+added"
    r"|\*+\s*\w"
    r"|†\s*\w"
    r")",
    re.IGNORECASE,
)

HEADER_HINT_RE = re.compile(
    r"(per\s*100\s*(g|gm|ml)|/\s*100\s*(g|gm|ml)|per\s*(serving|portion|serve)|%\s*(rda|ri|dv)|rda)",
    re.IGNORECASE,
)

START_INGREDIENT_PATTERNS = [
    r"ingredients?\s*[:\-]",
    r"composition\s*[:\-]",
    r"made\s+from\s*[:\-]",
    r"contains?\s*[:\-]",
    r"manufactured\s+from\s*[:\-]",
    r"saamagri\s*[:\-]",
]

END_INGREDIENT_PATTERNS = [
    r"allergen\s+information",
    r"allergy\s+advice",
    r"contains?\s+allergen",
    r"nutritional\s+information",
    r"nutrition\s+facts",
    r"best\s+before",
    r"manufactured\s+by",
    r"packed\s+by",
    r"fssai\s+lic",
    r"mkd\s+by",
    r"mfd\s+by",
    r"country\s+of\s+origin",
    r"net\s+(wt|weight|qty|quantity)",
    r"storage\s+instructions",
    r"directions\s+for\s+use",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OCRToken:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    cx: float
    cy: float
    line_hint: int = -1

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _apply_ocr_corrections(text: str) -> str:
    for pattern, replacement in OCR_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _strip_noise(text: str) -> str:
    text = text.replace("\u200b", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_numeric_like(text: str) -> bool:
    return bool(re.search(r"[\d]", text))


def _contains_percentage_only(text: str) -> bool:
    t = text.strip()
    return bool(re.fullmatch(r"[<>≤≥~\s\d.,]+\s*%", t))


def _flatten_ocr_payload(raw_ocr: Union[str, dict, None]) -> str:
    if raw_ocr is None:
        return ""

    if isinstance(raw_ocr, str):
        return raw_ocr

    if not isinstance(raw_ocr, dict):
        return str(raw_ocr)

    full = raw_ocr.get("fullTextAnnotation", {}) if isinstance(raw_ocr, dict) else {}
    text = full.get("text", "") if isinstance(full, dict) else ""
    if text:
        return text

    # Fallback: reconstruct text from words
    tokens = _vision_payload_to_tokens(raw_ocr)
    if not tokens:
        return ""

    lines = _cluster_tokens_into_lines(tokens)
    return "\n".join(_line_to_text(line) for line in lines)


def _safe_float(value: Optional[float], default: float = 0.0) -> float:
    return default if value is None or math.isnan(value) else float(value)


# ---------------------------------------------------------------------------
# Vision response parsing
# ---------------------------------------------------------------------------


def _vision_payload_to_tokens(payload: dict) -> List[OCRToken]:
    """Extract OCR tokens with boxes from a Google Vision full response."""
    tokens: List[OCRToken] = []

    pages = payload.get("fullTextAnnotation", {}).get("pages", [])
    if not pages:
        return tokens

    for page_idx, page in enumerate(pages):
        blocks = page.get("blocks", []) or []
        for block in blocks:
            paragraphs = block.get("paragraphs", []) or []
            for para in paragraphs:
                words = para.get("words", []) or []
                for word in words:
                    symbols = word.get("symbols", []) or []
                    text = "".join(sym.get("text", "") for sym in symbols).strip()
                    if not text:
                        continue

                    box = word.get("boundingBox", {}).get("vertices", []) or []
                    if len(box) < 4:
                        continue

                    xs = [float(v.get("x", 0.0)) for v in box]
                    ys = [float(v.get("y", 0.0)) for v in box]
                    x0, x1 = min(xs), max(xs)
                    y0, y1 = min(ys), max(ys)
                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0

                    tokens.append(
                        OCRToken(
                            text=text,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            cx=cx,
                            cy=cy,
                            line_hint=page_idx,
                        )
                    )

    return tokens


# ---------------------------------------------------------------------------
# Layout reconstruction
# ---------------------------------------------------------------------------


def _cluster_tokens_into_lines(tokens: Sequence[OCRToken]) -> List[List[OCRToken]]:
    if not tokens:
        return []

    ordered = sorted(tokens, key=lambda t: (t.cy, t.cx))
    heights = [t.h for t in ordered if t.h > 0]
    y_tolerance = max(6.0, (median(heights) * 0.65) if heights else 10.0)

    lines: List[List[OCRToken]] = []
    line_centers: List[float] = []

    for tok in ordered:
        placed = False
        for i, center in enumerate(line_centers):
            if abs(tok.cy - center) <= y_tolerance:
                lines[i].append(tok)
                new_center = sum(t.cy for t in lines[i]) / len(lines[i])
                line_centers[i] = new_center
                placed = True
                break
        if not placed:
            lines.append([tok])
            line_centers.append(tok.cy)

    for line in lines:
        line.sort(key=lambda t: t.cx)

    # Merge tiny adjacent lines if needed
    merged: List[List[OCRToken]] = []
    for line in sorted(lines, key=lambda ln: sum(t.cy for t in ln) / len(ln)):
        if not merged:
            merged.append(line)
            continue
        prev = merged[-1]
        prev_y = sum(t.cy for t in prev) / len(prev)
        cur_y = sum(t.cy for t in line) / len(line)
        prev_h = median([t.h for t in prev if t.h > 0]) if prev else 0
        cur_h = median([t.h for t in line if t.h > 0]) if line else 0
        merge_gap = max(8.0, min(prev_h, cur_h) * 0.55 if prev_h and cur_h else 8.0)
        if abs(cur_y - prev_y) <= merge_gap:
            merged[-1] = sorted(prev + line, key=lambda t: t.cx)
        else:
            merged.append(line)

    return merged


def _line_to_text(line: Sequence[OCRToken]) -> str:
    return " ".join(tok.text for tok in line).strip()


def _line_bounds(line: Sequence[OCRToken]) -> Tuple[float, float, float, float]:
    x0 = min(t.x0 for t in line)
    y0 = min(t.y0 for t in line)
    x1 = max(t.x1 for t in line)
    y1 = max(t.y1 for t in line)
    return x0, y0, x1, y1


def _split_line_into_segments(line: Sequence[OCRToken]) -> List[List[OCRToken]]:
    if not line:
        return []

    ordered = sorted(line, key=lambda t: t.x0)
    widths = [t.w for t in ordered if t.w > 0]
    median_width = median(widths) if widths else 18.0
    gap_threshold = max(18.0, median_width * 1.85)

    segments: List[List[OCRToken]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        gap = cur.x0 - prev.x1
        if gap > gap_threshold:
            segments.append([cur])
        else:
            segments[-1].append(cur)
    return segments


def _segment_text(seg: Sequence[OCRToken]) -> str:
    return " ".join(tok.text for tok in seg).strip()


def _segment_has_digits(seg: Sequence[OCRToken]) -> bool:
    return any(_is_numeric_like(tok.text) for tok in seg)


def _segment_cx(seg: Sequence[OCRToken]) -> float:
    return sum(t.cx for t in seg) / len(seg)


def _cluster_1d(values: Sequence[float], threshold: float = 40.0) -> List[float]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return []

    clusters: List[List[float]] = [[values[0]]]
    for v in values[1:]:
        if abs(v - clusters[-1][-1]) > threshold:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return [sum(c) / len(c) for c in clusters]


# ---------------------------------------------------------------------------
# Text normalization / matching
# ---------------------------------------------------------------------------


def _normalize_text_for_match(text: str) -> str:
    text = _apply_ocr_corrections(text)
    text = text.lower().strip()
    text = re.sub(r"[\*†‡#]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[:\-–]+$", "", text).strip()
    text = re.sub(r"\(.*?\)", "", text).strip()
    text = re.sub(r"\b(g|mg|mcg|μg|ug|kcal|kj|ml|%)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fuzzy_match_nutrient(raw_field: str) -> Optional[str]:
    normalized = _normalize_text_for_match(raw_field)

    if normalized in NUTRIENT_ALIASES:
        return NUTRIENT_ALIASES[normalized]

    no_paren = re.sub(r"\(.*?\)", "", normalized).strip()
    if no_paren in NUTRIENT_ALIASES:
        return NUTRIENT_ALIASES[no_paren]

    no_units = re.sub(r"\b(g|mg|mcg|μg|ug|kcal|kj|ml|%)\b", "", normalized).strip()
    if no_units in NUTRIENT_ALIASES:
        return NUTRIENT_ALIASES[no_units]

    best_score = 0.0
    best_key: Optional[str] = None
    for alias, canonical in NUTRIENT_ALIASES.items():
        score = SequenceMatcher(None, normalized, alias).ratio()
        if score > best_score:
            best_score = score
            best_key = canonical

    return best_key if best_score >= 0.82 else None


# ---------------------------------------------------------------------------
# Numeric parsing / unit conversion
# ---------------------------------------------------------------------------


def _parse_numeric_value(raw: str) -> Optional[float]:
    raw = raw.strip()

    if re.match(r"^(nil|none|trace|n\.?a\.?|not detected|nd|-)$", raw, re.IGNORECASE):
        return 0.0

    raw = re.sub(r"^[<>≤≥~≈]\\s*", "", raw)
    raw = re.sub(r"\s*(g|mg|mcg|μg|ug|kcal|kj|kJ|ml|%|iu|IU)\s*$", "", raw, flags=re.IGNORECASE)

    # Fix "25 7" -> "25.7" when the second part looks like decimal digits.
    raw = re.sub(r"(\d+)\s+(\d{1,2})$", r"\1.\2", raw)

    cleaned = re.sub(r"[^\d.]", "", raw)
    if not cleaned:
        return None

    parts = cleaned.split(".")
    if len(parts) > 2:
        cleaned = parts[0] + "." + "".join(parts[1:])

    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_numeric_value_and_unit(text: str) -> Tuple[Optional[float], str]:
    """Return the first numeric value and attached unit from a token/segment."""
    text = text.strip()
    text = _apply_ocr_corrections(text)

    m = re.search(
        r"([<>≤≥~≈]?[\s\d.,]+(?:\s\d{1,2})?)\s*(g|mg|mcg|μg|ug|kcal|kj|kJ|ml|%|iu|IU)?",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None, ""

    value = _parse_numeric_value(m.group(1))
    unit = (m.group(2) or "").lower().strip()
    return value, unit


def _infer_unit_from_field(raw_field: str) -> str:
    m = re.search(r"\(\s*(g|mg|mcg|μg|ug|kcal|kj|ml|iu)\s*\)", raw_field, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    m = re.search(r"\b(mg|mcg|μg|ug|kcal|kj)\b", raw_field, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return ""


def _convert_units(value: float, unit: str, canonical_key: str) -> float:
    unit = unit.lower().strip() if unit else ""

    if canonical_key == "energy_kj" or unit == "kj":
        return round(value * KJ_TO_KCAL, 2)

    if unit in ("mg", "milligrams", "milligram"):
        return round(value * MG_TO_G, 6)

    if unit in ("mcg", "μg", "ug", "micrograms", "microgram"):
        return round(value * MCG_TO_G, 8)

    return value


def _normalize_to_per_100g(value: float, serving_g: Optional[float]) -> float:
    if serving_g and serving_g > 0:
        return round((value / serving_g) * 100.0, 2)
    return value


# ---------------------------------------------------------------------------
# Column / layout detection
# ---------------------------------------------------------------------------


def _detect_header_roles(lines: Sequence[Sequence[OCRToken]]) -> Dict[str, int]:
    """Try to infer which visual column corresponds to which role."""
    best: Dict[str, Tuple[float, int]] = {}

    for line in lines[:15]:
        segments = _split_line_into_segments(line)
        for idx, seg in enumerate(segments):
            seg_text = _normalize_text_for_match(_segment_text(seg))
            if not seg_text:
                continue

            role = None
            if re.search(r"per\s*100\s*(g|gm|ml)|/\s*100\s*(g|gm|ml)|values?\s*per\s*100", seg_text, re.IGNORECASE):
                role = "per_100g"
            elif re.search(r"per\s*(serving|portion|serve|pack|sachet|piece|unit|biscuit)", seg_text, re.IGNORECASE):
                role = "per_serving"
            elif re.search(r"%\s*(rda|ri|dv)|rda", seg_text, re.IGNORECASE):
                role = "rda"

            if role is not None:
                x = _segment_cx(seg)
                current = best.get(role)
                if current is None or x < current[0]:
                    best[role] = (x, idx)

    # Fallback if header role text is not found but columns are visually clear.
    if not best:
        return {}

    # Convert to index map ordered by x position.
    ordered = sorted(((role, v[0], v[1]) for role, v in best.items()), key=lambda item: item[1])
    role_to_index: Dict[str, int] = {}
    for pos, (role, _, idx) in enumerate(ordered):
        role_to_index[role] = idx
    return role_to_index


def _detect_numeric_column_anchors(lines: Sequence[Sequence[OCRToken]]) -> List[float]:
    """Cluster x positions of numeric columns from the top part of the label."""
    xs: List[float] = []

    for line in lines[:18]:
        segments = _split_line_into_segments(line)
        for seg in segments:
            seg_text = _segment_text(seg)
            if _segment_has_digits(seg) or _contains_percentage_only(seg_text):
                xs.append(_segment_cx(seg))

    # Also inspect lines that look like column headers.
    for line in lines[:12]:
        text = _normalize_text_for_match(_line_to_text(line))
        if HEADER_HINT_RE.search(text):
            segments = _split_line_into_segments(line)
            for seg in segments:
                seg_text = _normalize_text_for_match(_segment_text(seg))
                if HEADER_HINT_RE.search(seg_text):
                    xs.append(_segment_cx(seg))

    if not xs:
        return []

    # Threshold should be wide enough to group table columns but not rows.
    spread = max(xs) - min(xs) if len(xs) > 1 else 0
    threshold = max(30.0, min(70.0, spread / 6.0 if spread else 40.0))
    return _cluster_1d(xs, threshold=threshold)


# ---------------------------------------------------------------------------
# Nutrition parsing
# ---------------------------------------------------------------------------


def _extract_serving_size_from_text(text: str) -> Optional[float]:
    patterns = [
        r"serving\s+size\s*[:\-]?\s*([\d.]+)\s*g",
        r"per\s+serving\s*[:\-]?\s*([\d.]+)\s*g",
        r"per\s+portion\s*[:\-]?\s*([\d.]+)\s*g",
        r"([\d.]+)\s*g\s+per\s+serving",
        r"serve\s+size\s*[:\-]?\s*([\d.]+)\s*g",
        r"portion\s+size\s*[:\-]?\s*([\d.]+)\s*g",
        r"per\s+([\d.]+)\s*g\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            value = _parse_numeric_value(m.group(1))
            if value is not None and 5 <= value <= 500:
                return value
    return None


def _choose_numeric_candidate(
    candidates: List[Tuple[float, str, float]],
    anchors: Sequence[float],
    per100g_idx: int,
) -> Optional[Tuple[float, str, float]]:
    """Pick the best numeric candidate for the per-100g column."""
    if not candidates:
        return None

    # If there are visual anchors, assign each candidate to the nearest anchor.
    if anchors:
        annotated = []
        for value, unit, x in candidates:
            nearest_idx = min(range(len(anchors)), key=lambda i: abs(x - anchors[i]))
            annotated.append((nearest_idx, value, unit, x))

        # Prefer the requested per-100g column, but ignore % only candidates.
        filtered = [item for item in annotated if item[2] != "%"]
        if not filtered:
            return None

        by_idx = [item for item in filtered if item[0] == per100g_idx]
        if by_idx:
            # If multiple candidates fall in same column, choose the first reliable one.
            return sorted(by_idx, key=lambda t: t[3])[0][1:]

        # Otherwise fall back to left-to-right order.
        return sorted(filtered, key=lambda t: (t[0], t[3]))[0][1:]

    # No anchors: use the first non-% candidate.
    for value, unit, _ in candidates:
        if unit != "%":
            return value, unit, _
    return None


def _row_candidates_from_line(line: Sequence[OCRToken]) -> List[Tuple[float, str, float]]:
    candidates: List[Tuple[float, str, float]] = []
    for tok in line:
        if not _is_numeric_like(tok.text):
            continue
        value, unit = _extract_numeric_value_and_unit(tok.text)
        if value is None:
            continue
        candidates.append((value, unit, tok.cx))
    return candidates


def _parse_nutrition_structured(lines: Sequence[Sequence[OCRToken]], raw_text: str) -> Tuple[Dict[str, float], List[str], float]:
    nutriments: Dict[str, float] = {}
    warnings: List[str] = []
    parsed_rows = 0
    matched_rows = 0

    role_map = _detect_header_roles(lines)
    anchors = _detect_numeric_column_anchors(lines)
    per100g_idx = role_map.get("per_100g", 0)
    serving_size = _extract_serving_size_from_text(raw_text)

    # If we have a serving size but no explicit per-100g column, we may need
    # to normalize per-serving single column values.
    explicit_per_serving = "per_serving" in role_map
    explicit_rda = "rda" in role_map

    pending_label: Optional[str] = None
    pending_raw_line: Optional[str] = None

    for line in lines:
        line_text = _line_to_text(line)
        norm_text = _normalize_text_for_match(line_text)

        if not norm_text:
            continue

        if NON_NUTRITIONAL_SKIP.match(norm_text):
            continue

        # Ignore pure header rows.
        if HEADER_HINT_RE.search(norm_text) and not _is_numeric_like(norm_text):
            continue

        if re.fullmatch(r"[\d\s.,%]+", norm_text):
            continue

        parsed_rows += 1

        # Split the line into label and value zone using the first numeric token.
        first_numeric_idx = None
        for idx, tok in enumerate(line):
            if _is_numeric_like(tok.text):
                first_numeric_idx = idx
                break

        if first_numeric_idx is None:
            # If this is a label-only row, retain it as a pending label.
            canonical = _fuzzy_match_nutrient(line_text)
            if canonical:
                pending_label = canonical
                pending_raw_line = line_text
                continue
            else:
                pending_label = None
                pending_raw_line = None
                continue

        label_tokens = line[:first_numeric_idx]
        value_tokens = line[first_numeric_idx:]
        raw_field = _line_to_text(label_tokens).strip().rstrip(":–-")

        canonical = _fuzzy_match_nutrient(raw_field)
        if not canonical and pending_label:
            canonical = pending_label
            raw_field = pending_raw_line or raw_field
            pending_label = None
            pending_raw_line = None

        if not canonical:
            # Sometimes the OCR starts with a number in a wrapped table row.
            # Skip those lines instead of forcing a wrong mapping.
            continue

        candidates = _row_candidates_from_line(value_tokens)
        if not candidates and pending_label:
            # Pair a numeric-only line with the last pending label.
            candidates = _row_candidates_from_line(line)

        chosen = _choose_numeric_candidate(candidates, anchors, per100g_idx)
        if chosen is None:
            # Keep pending label in case the value spills to the next line.
            pending_label = canonical
            pending_raw_line = raw_field
            continue

        raw_value, raw_unit, raw_x = chosen
        if raw_unit == "%":
            # Skip RDA-only rows unless there is also a usable value.
            non_pct = [c for c in candidates if c[1] != "%"]
            if non_pct:
                raw_value, raw_unit, raw_x = non_pct[0]
            else:
                continue

        if not raw_unit:
            raw_unit = _infer_unit_from_field(raw_field)

        value = _convert_units(raw_value, raw_unit, canonical)

        # Per-serving normalization is only safe if we know the serving size
        # and the row is not already from an explicit per-100g column.
        if serving_size and explicit_per_serving and not explicit_rda and per100g_idx == 0:
            if canonical not in ("energy_kj",):
                value = _normalize_to_per_100g(value, serving_size)

        if canonical in ("energy_kj", "energy_kcal"):
            canonical = "energy-kcal_100g"

        if canonical == "sodium_100g":
            nutriments["sodium_100g"] = round(value, 3)
            nutriments["salt_100g"] = round(value * 2.5, 3)
            matched_rows += 1
            continue

        # Prefer the first reliable value, but allow better structured rows to overwrite
        # earlier low-confidence guesses.
        if canonical not in nutriments:
            nutriments[canonical] = round(value, 2)
        else:
            # If the existing value is missing or absurd, replace it.
            if not isinstance(nutriments[canonical], (int, float)):
                nutriments[canonical] = round(value, 2)

        matched_rows += 1

    if len(nutriments) < MIN_NUTRIENTS_THRESHOLD:
        warnings.append("Fewer than 3 nutrients found from structured parsing; label may be partial or badly cropped.")

    confidence = _nutrition_confidence(nutriments, parsed_rows, matched_rows, role_map, anchors)
    return nutriments, warnings, confidence


def _parse_nutrition_text_only(raw_text: str) -> Tuple[Dict[str, float], List[str], float]:
    """Fallback for legacy OCR output that only contains flattened text."""
    nutriments: Dict[str, float] = {}
    warnings: List[str] = []
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    serving_size = _extract_serving_size_from_text(raw_text)

    per_unit = _detect_per_unit_text(raw_text)
    per100g_col_idx = 0

    for line in lines:
        norm = _normalize_text_for_match(line)
        if NON_NUTRITIONAL_SKIP.match(norm):
            continue
        if re.fullmatch(r"[\d\s.,%]+", norm):
            continue

        first_digit = re.search(r"[<>≤≥~≈]?\d", line)
        if not first_digit:
            continue

        raw_field = line[: first_digit.start()].strip().rstrip(":–-")
        canonical = _fuzzy_match_nutrient(raw_field)
        if not canonical:
            continue

        values_part = line[first_digit.start() :]
        candidates: List[Tuple[float, str, float]] = []
        for match in re.finditer(r"([<>≤≥~≈]?[\s\d.,]+(?:\s\d{1,2})?)\s*(g|mg|mcg|μg|ug|kcal|kj|kJ|ml|%|iu|IU)?", values_part, re.IGNORECASE):
            value = _parse_numeric_value(match.group(1))
            if value is None:
                continue
            unit = (match.group(2) or "").lower().strip()
            candidates.append((value, unit, float(match.start())))

        chosen = _choose_numeric_candidate(candidates, [], per100g_col_idx)
        if chosen is None:
            continue

        raw_value, raw_unit, _ = chosen
        if not raw_unit:
            raw_unit = _infer_unit_from_field(raw_field)

        value = _convert_units(raw_value, raw_unit, canonical)
        if per_unit == "per_serving" and serving_size and canonical not in ("energy_kj",):
            value = _normalize_to_per_100g(value, serving_size)

        if canonical in ("energy_kj", "energy_kcal"):
            canonical = "energy-kcal_100g"

        if canonical == "sodium_100g":
            nutriments["sodium_100g"] = round(value, 3)
            nutriments["salt_100g"] = round(value * 2.5, 3)
        else:
            nutriments[canonical] = round(value, 2)

    if len(nutriments) < MIN_NUTRIENTS_THRESHOLD:
        warnings.append("Text-only parsing found too few nutrients; structured OCR would be more reliable.")

    confidence = _nutrition_confidence(nutriments, len(lines), len(nutriments), {}, [])
    return nutriments, warnings, confidence


def _detect_per_unit_text(text: str) -> str:
    text_lower = text.lower()

    per_100g_signals = [
        "per 100 g", "per 100g", "per100g", "/100g", "/100 g", "per 100 ml", "per 100ml", "per 100 gm", "values per 100",
    ]
    per_serving_signals = [
        "per serving", "per portion", "per serve", "per pack", "per packet", "per sachet", "per biscuit", "per piece", "per unit",
    ]

    first_100g = min((text_lower.find(s) for s in per_100g_signals if s in text_lower), default=9999)
    first_serving = min((text_lower.find(s) for s in per_serving_signals if s in text_lower), default=9999)
    return "per_100g" if first_100g <= first_serving else "per_serving"


# ---------------------------------------------------------------------------
# Nutrition post-processing / confidence
# ---------------------------------------------------------------------------


def _sanity_check_nutriments(nutriments: Dict[str, float]) -> Dict[str, float]:
    bounds = {
        "energy-kcal_100g": (0, 900),
        "proteins_100g": (0, 100),
        "carbohydrates_100g": (0, 100),
        "sugars_100g": (0, 100),
        "fat_100g": (0, 100),
        "saturated-fat_100g": (0, 100),
        "trans-fat_100g": (0, 20),
        "fiber_100g": (0, 100),
        "sodium_100g": (0, 40),
        "salt_100g": (0, 100),
        "cholesterol_100g": (0, 5),
    }

    cleaned: Dict[str, float] = {}
    for key, value in nutriments.items():
        if key in bounds:
            lo, hi = bounds[key]
            if lo <= value <= hi:
                cleaned[key] = value
            else:
                logger.debug("Dropping out-of-range nutrient %s=%s", key, value)
        else:
            cleaned[key] = value

    if "sugars_100g" in cleaned and "carbohydrates_100g" in cleaned:
        cleaned["sugars_100g"] = min(cleaned["sugars_100g"], cleaned["carbohydrates_100g"])

    if "saturated-fat_100g" in cleaned and "fat_100g" in cleaned:
        cleaned["saturated-fat_100g"] = min(cleaned["saturated-fat_100g"], cleaned["fat_100g"])

    if "sodium_100g" in cleaned and "salt_100g" not in cleaned:
        cleaned["salt_100g"] = round(cleaned["sodium_100g"] * 2.5, 3)

    return cleaned


def _nutrition_confidence(
    nutriments: Dict[str, float],
    parsed_rows: int,
    matched_rows: int,
    role_map: Dict[str, int],
    anchors: Sequence[float],
) -> float:
    key_found = sum(1 for k in KEY_NUTRIENTS if k in nutriments)
    key_score = key_found / len(KEY_NUTRIENTS)

    row_score = min(1.0, matched_rows / max(1, parsed_rows))
    structure_score = 0.0
    if anchors:
        structure_score += 0.4
    if role_map:
        structure_score += 0.35
    if len(nutriments) >= MIN_NUTRIENTS_THRESHOLD:
        structure_score += 0.25
    structure_score = min(1.0, structure_score)

    confidence = (key_score * 0.45) + (row_score * 0.35) + (structure_score * 0.20)
    return max(0.0, min(1.0, round(confidence, 3)))


# ---------------------------------------------------------------------------
# Ingredients parsing
# ---------------------------------------------------------------------------


def _extract_ingredients_block_from_text(text: str) -> str:
    if not text:
        return ""

    start_idx = None
    for pattern in START_INGREDIENT_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            start_idx = m.end()
            break

    if start_idx is None:
        return text

    cut_text = text[start_idx:]

    end_idx = len(cut_text)
    for pattern in END_INGREDIENT_PATTERNS:
        m = re.search(pattern, cut_text, re.IGNORECASE)
        if m:
            end_idx = min(end_idx, m.start())

    return cut_text[:end_idx]


def parse_ingredients_label(raw_ocr: Union[str, dict, None]) -> str:
    text = _flatten_ocr_payload(raw_ocr)
    if not text:
        return ""

    text = _apply_ocr_corrections(text)
    text = _extract_ingredients_block_from_text(text)

    # Keep line order, but collapse wrapped lines.
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove footnote markers and percent annotations.
    text = re.sub(r"\*+[^,)]*", "", text)
    text = re.sub(r"†[^,)]*", "", text)
    text = re.sub(r"\(\s*\d+\.?\d*\s*%\s*\)", "", text)

    # Remove allergen / marketing disclaimers mixed into ingredient text.
    text = re.sub(r"may\s+contain\s+traces?\s+of[^.]*\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"may\s+contain\s*:?[^.]*\.?", "", text, flags=re.IGNORECASE)

    # Normalize separators, but preserve parentheses and E-numbers.
    text = re.sub(r";\s*", ", ", text)
    text = re.sub(r"\s*/\s*", ", ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Clean leading/trailing commas.
    text = re.sub(r"^\s*,\s*", "", text)
    text = re.sub(r",\s*$", "", text)

    return text


def _ingredients_confidence(text: str) -> float:
    if not text:
        return 0.0
    has_anchor = bool(re.search(r"ingredients?|composition|made from|contains", text, re.IGNORECASE))
    has_separators = "," in text or ";" in text
    reasonable_len = 10 < len(text) < 5000
    has_no_junk = not re.search(r"\b\d{6,}\b", text)
    confidence = (0.35 if has_anchor else 0.0) + (0.25 if has_separators else 0.0) + (0.25 if reasonable_len else 0.0) + (0.15 if has_no_junk else 0.0)
    return round(min(1.0, confidence), 3)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def parse_nutrition_label(raw_ocr: Union[str, dict, None]) -> Dict[str, float]:
    """Compatibility wrapper that returns only the nutriments dictionary."""
    result = process_ocr_scan(raw_ocr, "nutrition")
    return result.get("data") or {}


def process_ocr_scan(raw_ocr: Union[str, dict, None], scan_type: str) -> dict:
    """
    Entry point called by the API endpoint.

    scan_type: 'nutrition' | 'ingredients'
    """
    warnings: List[str] = []
    raw_text = _flatten_ocr_payload(raw_ocr)
    raw_text = _apply_ocr_corrections(raw_text)
    raw_text = _strip_noise(raw_text)

    logger.info("PARSER: start scan_type=%s raw_len=%s", scan_type, len(raw_text or ""))

    if not raw_text or len(raw_text.strip()) < 10:
        return {
            "scan_type": scan_type,
            "success": False,
            "data": None,
            "confidence": 0.0,
            "raw_text": raw_text or "",
            "warnings": ["OCR returned insufficient text. Ensure good lighting and focus."],
        }

    if scan_type == "nutrition":
        structured_tokens = _vision_payload_to_tokens(raw_ocr) if isinstance(raw_ocr, dict) else []
        if structured_tokens:
            lines = _cluster_tokens_into_lines(structured_tokens)
            nutriments, parse_warnings, confidence = _parse_nutrition_structured(lines, raw_text)
        else:
            nutriments, parse_warnings, confidence = _parse_nutrition_text_only(raw_text)

        warnings.extend(parse_warnings)
        nutriments = _sanity_check_nutriments(nutriments)

        if confidence < 0.4:
            warnings.append(
                "Low confidence. The crop may include unrelated text or the nutrition table may be partially visible."
            )
        if "energy-kcal_100g" not in nutriments:
            warnings.append("Energy value not detected.")

        success = confidence > 0.2 and len(nutriments) > 0
        return {
            "scan_type": "nutrition",
            "success": success,
            "data": nutriments,
            "confidence": round(confidence, 2),
            "raw_text": raw_text,
            "warnings": warnings,
        }

    if scan_type == "ingredients":
        ingredients_text = parse_ingredients_label(raw_ocr)
        confidence = _ingredients_confidence(ingredients_text)

        if not ingredients_text:
            warnings.append("No ingredient text detected.")
        if confidence < 0.35:
            warnings.append("Ingredient scan confidence is low. The crop may not be centered on the ingredient block.")

        return {
            "scan_type": "ingredients",
            "success": confidence > 0.3,
            "data": ingredients_text,
            "confidence": round(confidence, 2),
            "raw_text": raw_text,
            "warnings": warnings,
        }

    return {
        "scan_type": scan_type,
        "success": False,
        "data": None,
        "confidence": 0.0,
        "raw_text": raw_text,
        "warnings": [f"Unknown scan_type: {scan_type}"],
    }


# ---------------------------------------------------------------------------
# Optional helpers for debugging / review
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fuzzy product name matcher (restored for compatibility)
# ---------------------------------------------------------------------------

def _fuzzy_product_name_match(query: str, candidates: list, threshold: float = 0.6) -> list:
    results = []
    query_lower = query.lower().strip()

    for candidate in candidates:
        name = candidate.get('product_name', '')
        if not name:
            continue

        name_lower = name.lower()

        query_tokens = set(re.findall(r'\b\w{3,}\b', query_lower))
        candidate_tokens = set(re.findall(r'\b\w{3,}\b', name_lower))

        if query_tokens and candidate_tokens:
            overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens), len(candidate_tokens))
        else:
            overlap = 0

        sim = SequenceMatcher(None, query_lower, name_lower).ratio()
        score = (overlap * 0.6) + (sim * 0.4)

        if score >= threshold:
            results.append({**candidate, '_match_score': round(score, 3)})

    results.sort(key=lambda x: x['_match_score'], reverse=True)
    return results[:5]
"""
vision_service.py  — PATCHED
Fixes applied (search "# FIX" to locate every change):

  BUG VS-1  Word text reconstruction corrupts numeric tokens
            The old loop appended SPACE / EOL break characters into word_text
            during symbol iteration.  strip() only removes LEADING / TRAILING
            whitespace — an internal space ("1 7.4g") survives and is later
            mis-parsed as 1.74 instead of 17.4 by _parse_numeric_value's
            "25 7 → 25.7" heuristic.
            Fix: reconstruct word text purely from symbol characters — break
            types are only needed to rebuild flat text, which Vision API
            already provides as fullTextAnnotation.text.

  BUG VS-2  No coordinate normalisation for rotated images
            When a user photographs a label with the phone held sideways (90°),
            Vision API still reads the text correctly but returns bounding boxes
            in the ORIGINAL image coordinate space — so x and y are
            effectively swapped.  The y-centre clustering in
            _group_words_into_lines then groups words from DIFFERENT table rows
            together, causing label-value row mismatches (e.g. protein label
            paired with carbohydrate value).
            Fix: detect the overall text orientation from the first block's
            bounding-polygon vertex ordering, then transform all word
            coordinates into a normalised upright frame before returning.

  BUG VS-3  block_idx / para_idx extracted but discarded
            These fields were stored in each WordDict but the parser never
            used them — wasted potential for disambiguating rows.
            Fix: expose a new 'line_key' field (block_idx, para_idx) so the
            parser can use Vision API's own grouping as a tie-breaker when
            the y-centre geometry is ambiguous.
"""

import os
import logging
import math
from typing import Optional

import requests

logger = logging.getLogger("smartscan.vision")

VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"

_EMPTY_FAIL: dict = {
    "success":      False,
    "text":         "",
    "words":        [],
    "raw_response": {},
    "error":        None,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_vertices(vertices: list[dict]) -> tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) from a Vision API vertices list."""
    xs = [v.get("x", 0) for v in vertices]
    ys = [v.get("y", 0) for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


# ─────────────────────────────────────────────────────────────────────────────
# FIX BUG VS-2  — rotation detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_orientation(response_data: dict) -> tuple[int, int, int]:
    """
    Detect the text orientation angle (0 / 90 / 180 / 270 degrees clockwise)
    from the first block's bounding polygon vertex ordering, and return the
    canvas dimensions (page_width, page_height) from the Vision API response.

    Returns (angle_degrees, page_width, page_height).

    For a correctly-oriented (0°) page the Vision API lists vertices in
    clockwise order starting from top-left:
        v0=TL  v1=TR  v2=BR  v3=BL
    The top edge v0→v1 therefore points rightward (dx > 0, dy ≈ 0).

    For a 90° clockwise rotation (phone held sideways, landscape label
    photographed as portrait) the top edge points downward (dx ≈ 0, dy > 0).
    """
    responses = response_data.get("responses", [])
    if not responses:
        return 0, 0, 0

    pages = responses[0].get("fullTextAnnotation", {}).get("pages", [])
    if not pages:
        return 0, 0, 0

    page       = pages[0]
    page_w     = page.get("width",  0)
    page_h     = page.get("height", 0)
    angle      = 0

    # Use the first few blocks to vote on orientation
    angle_votes: dict[int, int] = {0: 0, 90: 0, 180: 0, 270: 0}

    for block in page.get("blocks", [])[:5]:
        verts = block.get("boundingBox", {}).get("vertices", [])
        if len(verts) < 2:
            continue

        v0 = verts[0]
        v1 = verts[1]
        dx = v1.get("x", 0) - v0.get("x", 0)
        dy = v1.get("y", 0) - v0.get("y", 0)

        if abs(dx) >= abs(dy):
            # Top edge is horizontal
            if dx >= 0:
                angle_votes[0]   += 1   # normal
            else:
                angle_votes[180] += 1   # upside-down
        else:
            # Top edge is vertical
            if dy >= 0:
                angle_votes[90]  += 1   # 90° clockwise
            else:
                angle_votes[270] += 1   # 90° counter-clockwise

    angle = max(angle_votes, key=lambda k: angle_votes[k])

    logger.info(
        "VISION: orientation votes=%s  detected=%d°  page=%dx%d",
        angle_votes, angle, page_w, page_h,
    )
    return angle, page_w, page_h


def _normalise_coordinates(
    words: list[dict],
    angle: int,
    page_w: int,
    page_h: int,
) -> list[dict]:
    """
    Transform word bounding-box coordinates so that the text reads
    left-to-right, top-to-bottom regardless of how the image was captured.

    Transformation formulae (pixel coordinates):
        0°   → identity
        90°  → (x, y)  →  (y,        page_w − x)   [clockwise rotation]
        180° → (x, y)  →  (page_w−x, page_h − y)
        270° → (x, y)  →  (page_h−y, x)             [counter-clockwise]

    We operate on axis-aligned bounding boxes (x_min, y_min, x_max, y_max)
    by transforming all four corners and recomputing the axis-aligned box
    from the results.
    """
    if angle == 0 or not page_w or not page_h:
        return words

    def _transform_point(x: int, y: int) -> tuple[int, int]:
        if angle == 90:
            return y, page_w - x
        elif angle == 180:
            return page_w - x, page_h - y
        elif angle == 270:
            return page_h - y, x
        return x, y

    normalised = []
    for w in words:
        corners = [
            (w['x_min'], w['y_min']),
            (w['x_max'], w['y_min']),
            (w['x_max'], w['y_max']),
            (w['x_min'], w['y_max']),
        ]
        transformed = [_transform_point(cx, cy) for cx, cy in corners]
        txs = [p[0] for p in transformed]
        tys = [p[1] for p in transformed]
        normalised.append({
            **w,
            'x_min': min(txs),
            'y_min': min(tys),
            'x_max': max(txs),
            'y_max': max(tys),
        })

    return normalised


# ─────────────────────────────────────────────────────────────────────────────
# FIX BUG VS-1  — word text reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_words_with_boxes(response_data: dict) -> list[dict]:
    """
    Walk the Vision API fullTextAnnotation hierarchy:
        page → block → paragraph → word → symbol

    FIX BUG VS-1: The previous implementation appended SPACE / EOL break
    characters to word_text inside the symbol loop.  strip() only removes
    LEADING and TRAILING whitespace — an internal space caused by a break
    marker on a non-final symbol (OCR noise) would survive and corrupt the
    value, e.g. "17.4g" → "17.4 g" → parsed as 17.4 (OK), but "17.4g" where
    the "." symbol carries a SPACE break → "17. 4g" → strip → "17. 4g" →
    _parse_numeric_value drops the trailing "4g" → 17.0 instead of 17.4.

    Fix: reconstruct word_text purely as the concatenation of symbol.text
    characters.  The Vision API already provides the flat document text via
    fullTextAnnotation.text; we do not need to rebuild it from symbols here.

    FIX BUG VS-3: Add 'line_key' = (block_idx, para_idx) to each WordDict
    so the geometry parser can use Vision API's own grouping as a tie-breaker
    when y-centre proximity is ambiguous.
    """
    words: list[dict] = []
    responses = response_data.get("responses", [])
    if not responses:
        return words

    full_text_annotation = responses[0].get("fullTextAnnotation", {})

    for page in full_text_annotation.get("pages", []):
        for block_idx, block in enumerate(page.get("blocks", [])):
            for para_idx, paragraph in enumerate(block.get("paragraphs", [])):
                for word_data in paragraph.get("words", []):

                    # ── FIX VS-1: concatenate symbol texts only ───────────────
                    # Do NOT append break characters here.  Break types signal
                    # inter-word / inter-line gaps in the flat document, not
                    # intra-word structure.  Appending them inside the loop
                    # inserts spaces into word_text when a non-final symbol
                    # has an unexpected break type (common OCR noise on numbers).
                    word_text = "".join(
                        symbol.get("text", "")
                        for symbol in word_data.get("symbols", [])
                    ).strip()

                    if not word_text:
                        continue

                    # ── Extract bounding box ──────────────────────────────────
                    vertices = (
                        word_data
                        .get("boundingBox", {})
                        .get("vertices", [])
                    )
                    if len(vertices) < 4:
                        continue

                    x_min, y_min, x_max, y_max = _parse_vertices(vertices)

                    words.append({
                        "text":      word_text,
                        "x_min":     x_min,
                        "y_min":     y_min,
                        "x_max":     x_max,
                        "y_max":     y_max,
                        # FIX VS-3: expose Vision API grouping for tie-breaking
                        "block_idx": block_idx,
                        "para_idx":  para_idx,
                        # Convenience composite key for the parser
                        "line_key":  (block_idx, para_idx),
                    })

    return words


def _extract_flat_text(response_data: dict) -> str:
    """
    Extract the flat text string from a Vision API response.
    Tries fullTextAnnotation first; falls back to textAnnotations[0].
    """
    responses = response_data.get("responses", [])
    if not responses:
        return ""

    r    = responses[0]
    text = r.get("fullTextAnnotation", {}).get("text", "")
    if not text:
        ta = r.get("textAnnotations", [])
        if ta:
            text = ta[0].get("description", "")
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_vision_data(image_base64: str) -> dict:
    """
    Call Vision API and return a structured result containing both the flat
    text string (for Tier 2 / ingredients parsing) and per-word bounding
    boxes (for Tier 1 geometry-aware nutrition parsing).

    Bounding boxes are normalised to an upright coordinate frame so that the
    geometry parser works correctly regardless of how the user's phone was
    oriented when photographing the label.
    """
    if not VISION_API_KEY:
        logger.error("VISION: GOOGLE_VISION_API_KEY not set")
        return {**_EMPTY_FAIL, "error": "API key not configured"}

    payload = {
        "requests": [
            {
                "image": {"content": image_base64},
                "features": [
                    {"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 1}
                ],
                "imageContext": {
                    "languageHints": ["en", "hi"],
                },
            }
        ]
    }

    try:
        logger.info("VISION: sending request")
        resp = requests.post(
            f"{VISION_API_URL}?key={VISION_API_KEY}",
            json=payload,
            timeout=15,
        )
        logger.info(
            "VISION: response status=%d  size=%d chars",
            resp.status_code, len(resp.text),
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.Timeout:
        logger.error("VISION: request timed out")
        return {**_EMPTY_FAIL, "error": "Vision API request timed out"}
    except requests.RequestException as exc:
        logger.error("VISION: HTTP error: %s", exc)
        return {**_EMPTY_FAIL, "error": f"HTTP error: {exc}"}
    except ValueError as exc:
        logger.error("VISION: JSON decode error: %s", exc)
        return {**_EMPTY_FAIL, "error": f"JSON decode error: {exc}"}

    # ── Validate response structure ───────────────────────────────────────────
    responses_list = data.get("responses", [])
    if not responses_list:
        logger.warning("VISION: empty responses array")
        return {**_EMPTY_FAIL, "error": "Empty Vision API responses"}

    r0 = responses_list[0]
    if "error" in r0:
        msg = r0["error"].get("message", "Unknown Vision API error")
        logger.error("VISION: API returned error: %s", msg)
        return {**_EMPTY_FAIL, "error": msg}

    # ── Extract text and words ────────────────────────────────────────────────
    text  = _extract_flat_text(data)
    words = _extract_words_with_boxes(data)       # FIX VS-1 applied inside

    # ── FIX VS-2: normalise coordinates for rotated images ───────────────────
    angle, page_w, page_h = _detect_orientation(data)
    if angle != 0:
        words = _normalise_coordinates(words, angle, page_w, page_h)
        logger.info(
            "VISION: rotated %d° — coordinates normalised (canvas %dx%d)",
            angle, page_w, page_h,
        )

    logger.info(
        "VISION: text_len=%d  words=%d  orientation=%d°",
        len(text), len(words), angle,
    )

    return {
        "success":      bool(text or words),
        "text":         text,
        "words":        words,
        "raw_response": data,
        "error":        None,
        "orientation":  angle,   # exposed for debugging
    }


def extract_text_from_image(image_base64: str) -> Optional[str]:
    """
    Deprecated shim — returns only the flat text string.
    Existing callers will continue to work; new callers should use
    extract_vision_data() to get bounding-box data as well.
    """
    result = extract_vision_data(image_base64)
    return result["text"] if result["success"] else None
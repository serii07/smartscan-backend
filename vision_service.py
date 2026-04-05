"""
vision_service.py
Wraps Google Cloud Vision API — DOCUMENT_TEXT_DETECTION mode.

Key change from previous version:
  extract_text_from_image() only returned the flat text string, discarding
  every word's bounding-box coordinates. That destroyed the spatial structure
  the OCR parser needs for reliable column detection.

  Now extract_vision_data() returns a structured dict that carries:
    - text   : flat string (Tier 2 / ingredients parser fallback)
    - words  : list of word objects, each with pixel-accurate bounding box
    - raw_response : full Vision JSON (for debugging / future use)

  The old extract_text_from_image() is kept as a thin shim for any callers
  that have not yet been updated.
"""

import os
import logging
from typing import Optional, Union
import requests

logger = logging.getLogger("smartscan.vision")

VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"

# ── Structured result schema ──────────────────────────────────────────────────
# {
#   "success"      : bool
#   "text"         : str                 — flat OCR string (Tier 2 fallback)
#   "words"        : list[WordDict]      — bounding-box word objects (Tier 1)
#   "raw_response" : dict                — full Vision API JSON
#   "error"        : str | None
# }
#
# WordDict:
# {
#   "text"      : str    — word text (trailing whitespace stripped)
#   "x_min"     : int    — left edge of bounding box (pixels)
#   "y_min"     : int    — top edge
#   "x_max"     : int    — right edge
#   "y_max"     : int    — bottom edge
#   "block_idx" : int    — Vision API block index (0-based)
#   "para_idx"  : int    — Vision API paragraph index within block
# }


_EMPTY_FAIL: dict = {
    "success": False,
    "text": "",
    "words": [],
    "raw_response": {},
    "error": None,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_vertices(vertices: list[dict]) -> tuple[int, int, int, int]:
    """
    Return (x_min, y_min, x_max, y_max) from a Vision API vertices list.
    Handles missing x/y keys by defaulting to 0.
    """
    xs = [v.get("x", 0) for v in vertices]
    ys = [v.get("y", 0) for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def _extract_words_with_boxes(response_data: dict) -> list[dict]:
    """
    Walk the Vision API fullTextAnnotation hierarchy:
        page → block → paragraph → word → symbol

    Each word's text is reconstructed from its symbols, respecting the
    DetectedBreak type that Vision API stores after each symbol.

    Break types → appended characters:
        SPACE / SURE_SPACE              → single space
        EOL_SURE_SPACE / LINE_BREAK     → newline (stripped before storage)
        HYPHEN / UNKNOWN                → nothing

    Returns a flat list of WordDict objects (empty words are filtered out).
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

                    # ── Reconstruct word text from symbols ────────────────────
                    word_text = ""
                    for symbol in word_data.get("symbols", []):
                        word_text += symbol.get("text", "")

                        prop = symbol.get("property", {})
                        brk = prop.get("detectedBreak", {})
                        brk_type = brk.get("type", "")

                        if brk_type in ("SPACE", "SURE_SPACE"):
                            word_text += " "
                        elif brk_type in ("EOL_SURE_SPACE", "LINE_BREAK"):
                            word_text += "\n"
                        # HYPHEN / UNKNOWN → append nothing

                    word_text = word_text.strip()
                    if not word_text:
                        continue

                    # ── Extract bounding box ──────────────────────────────────
                    vertices = word_data.get("boundingBox", {}).get("vertices", [])
                    if len(vertices) < 4:
                        # Vision API always returns 4 vertices for valid words;
                        # skip if malformed.
                        continue

                    x_min, y_min, x_max, y_max = _parse_vertices(vertices)

                    words.append({
                        "text":      word_text,
                        "x_min":     x_min,
                        "y_min":     y_min,
                        "x_max":     x_max,
                        "y_max":     y_max,
                        "block_idx": block_idx,
                        "para_idx":  para_idx,
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

    r = responses[0]
    text = r.get("fullTextAnnotation", {}).get("text", "")
    if not text:
        ta = r.get("textAnnotations", [])
        if ta:
            text = ta[0].get("description", "")
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def extract_vision_data(image_base64: str) -> dict:
    """
    Call Vision API and return a structured result containing both the flat
    text string (for Tier 2 / ingredients parsing) and per-word bounding
    boxes (for Tier 1 geometry-aware nutrition parsing).

    Parameters
    ----------
    image_base64 : str
        Base64-encoded image content (JPEG or PNG).

    Returns
    -------
    dict  matching the schema described at the top of this file.
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
                    "languageHints": ["en", "hi"]
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
            "VISION: response status=%d size=%d chars",
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
    words = _extract_words_with_boxes(data)

    logger.info(
        "VISION: extracted text_len=%d words=%d",
        len(text), len(words),
    )

    return {
        "success":      bool(text or words),
        "text":         text,
        "words":        words,
        "raw_response": data,
        "error":        None,
    }


def extract_text_from_image(image_base64: str) -> Optional[str]:
    """
    Deprecated shim — returns only the flat text string.
    Existing callers will continue to work; new callers should use
    extract_vision_data() to get bounding-box data as well.
    """
    result = extract_vision_data(image_base64)
    return result["text"] if result["success"] else None
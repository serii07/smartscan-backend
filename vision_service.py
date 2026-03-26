"""
vision_service.py
Wraps Google Cloud Vision API — DOCUMENT_TEXT_DETECTION mode.
This mode is specifically optimised for dense printed text (nutrition labels, ingredient lists)
as opposed to TEXT_DETECTION which is for sparse natural scene text.
"""

import os
import base64
import requests
from typing import Optional

VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


def extract_text_from_image(image_base64: str) -> Optional[str]:
    """
    Send a base64-encoded image to Google Vision API.
    Returns the full text annotation as a single string, or None on failure.

    DOCUMENT_TEXT_DETECTION is used because:
    - Preserves paragraph/line structure better than TEXT_DETECTION
    - Handles small fonts better (nutrition label text is often 6-8pt)
    - Better with structured tabular content (nutrition facts table)
    - Handles multilingual text (English + Hindi on Indian labels)
    """
    if not VISION_API_KEY:
        print("WARNING: GOOGLE_VISION_API_KEY not set")
        return None

    payload = {
        "requests": [
            {
                "image": {
                    "content": image_base64
                },
                "features": [
                    {
                        "type": "DOCUMENT_TEXT_DETECTION",
                        "maxResults": 1
                    }
                ],
                "imageContext": {
                    # Hint Vision API to expect English + Hindi
                    # This significantly improves accuracy on Indian labels
                    "languageHints": ["en", "hi"]
                }
            }
        ]
    }

    try:
        response = requests.post(
            f"{VISION_API_URL}?key={VISION_API_KEY}",
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        responses = data.get("responses", [])
        if not responses:
            return None

        full_text_annotation = responses[0].get("fullTextAnnotation", {})
        text = full_text_annotation.get("text", "")

        if not text:
            # Fall back to textAnnotations
            text_annotations = responses[0].get("textAnnotations", [])
            if text_annotations:
                text = text_annotations[0].get("description", "")

        return text.strip() if text else None

    except requests.RequestException as e:
        print(f"Vision API error: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Vision API response parse error: {e}")
        return None

"""
vision_service.py
Wraps Google Cloud Vision API — DOCUMENT_TEXT_DETECTION mode.
"""

import os
import requests
from typing import Optional

VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


def extract_text_from_image(image_base64: str) -> Optional[str]:
    if not VISION_API_KEY:
        print("VISION: GOOGLE_VISION_API_KEY not set", flush=True)
        return None

    payload = {
        "requests": [
            {
                "image": {"content": image_base64},
                "features": [
                    {
                        "type": "DOCUMENT_TEXT_DETECTION",
                        "maxResults": 1
                    }
                ],
                "imageContext": {
                    "languageHints": ["en", "hi"]
                }
            }
        ]
    }

    try:
        print("VISION: sending request", flush=True)

        response = requests.post(
            f"{VISION_API_URL}?key={VISION_API_KEY}",
            json=payload,
            timeout=15
        )

        print(f"VISION: response received status={response.status_code}", flush=True)
        print(f"VISION: response size={len(response.text)} chars", flush=True)
        response.raise_for_status()
        data = response.json()

        print("VISION: json parsed", flush=True)

        responses = data.get("responses", [])
        if not responses:
            print("VISION: empty responses", flush=True)
            return None

        full_text_annotation = responses[0].get("fullTextAnnotation", {})
        text = full_text_annotation.get("text", "")

        if not text:
            text_annotations = responses[0].get("textAnnotations", [])
            if text_annotations:
                text = text_annotations[0].get("description", "")

        print(f"VISION: extracted text length={len(text.strip()) if text else 0}", flush=True)

        return responses[0]

    except requests.RequestException as e:
        print(f"VISION ERROR: {e}", flush=True)
        return None
    except (KeyError, IndexError) as e:
        print(f"VISION PARSE ERROR: {e}", flush=True)
        return None
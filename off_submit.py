"""
off_submit.py
Submits user-contributed product data to OpenFoodFacts via their write API.
This runs asynchronously — we serve the user from our local cache immediately,
then submit to OFF in the background.

OFF Write API docs: https://wiki.openfoodfacts.org/API/Write
"""

import os
import requests
import json
from typing import Optional

# OFF credentials — register free account at https://world.openfoodfacts.org/cgi/user.pl
OFF_USERNAME = os.environ.get("OFF_USERNAME", "")
OFF_PASSWORD = os.environ.get("OFF_PASSWORD", "")

OFF_WRITE_URL = "https://world.openfoodfacts.org/cgi/product_jqm2.pl"
OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
OFF_IN_SEARCH_URL = "https://in.openfoodfacts.org/cgi/search.pl"


def submit_product_to_off(
    barcode: str,
    product_name: Optional[str],
    nutriments: Optional[dict],
    ingredients_text: Optional[str],
    brands: Optional[str] = None,
    quantity: Optional[str] = None
) -> dict:
    """
    Submit a product contribution to OpenFoodFacts.

    OFF write API uses multipart form data.
    Returns {"success": bool, "message": str, "status_verbose": str}
    """
    if not OFF_USERNAME or not OFF_PASSWORD:
        return {
            "success": False,
            "message": "OFF credentials not configured",
            "status_verbose": "skipped"
        }

    payload = {
        "code":     barcode,
        "user_id":  OFF_USERNAME,
        "password": OFF_PASSWORD,
        "comment":  "SmartScan app user contribution",
        "app_name": "SmartScan",
        "app_version": "1.0",
        "lang":     "en",
        "countries": "India",
    }

    if product_name:
        payload["product_name"] = product_name
        payload["product_name_en"] = product_name

    if brands:
        payload["brands"] = brands

    if quantity:
        payload["quantity"] = quantity

    if ingredients_text:
        payload["ingredients_text"] = ingredients_text
        payload["ingredients_text_en"] = ingredients_text

    # OFF expects nutriments with specific naming
    if nutriments:
        # Map our internal keys to OFF write API keys
        off_nutriment_map = {
            "energy-kcal_100g":    ("energy-kcal", "kcal"),
            "proteins_100g":       ("proteins",     "g"),
            "carbohydrates_100g":  ("carbohydrates","g"),
            "sugars_100g":         ("sugars",        "g"),
            "fat_100g":            ("fat",           "g"),
            "saturated-fat_100g":  ("saturated-fat", "g"),
            "trans-fat_100g":      ("trans-fat",     "g"),
            "fiber_100g":          ("fiber",         "g"),
            "sodium_100g":         ("sodium",        "g"),
            "salt_100g":           ("salt",          "g"),
            "cholesterol_100g":    ("cholesterol",   "g"),
            "calcium_100g":        ("calcium",       "g"),
            "iron_100g":           ("iron",          "g"),
        }
        for internal_key, value in nutriments.items():
            if internal_key in off_nutriment_map:
                off_key, unit = off_nutriment_map[internal_key]
                payload[f"nutriment_{off_key}"]      = str(value)
                payload[f"nutriment_{off_key}_unit"]  = unit
                payload[f"nutriment_{off_key}_100g"]  = str(value)

        payload["nutrition_data_per"] = "100g"
        payload["nutrition_grade_fr"] = ""  # let OFF compute this

    try:
        response = requests.post(
            OFF_WRITE_URL,
            data=payload,
            timeout=15
        )
        response.raise_for_status()
        result = response.json()

        return {
            "success": result.get("status") == 1,
            "message": result.get("status_verbose", ""),
            "status_verbose": result.get("status_verbose", ""),
            "product_id": result.get("code", barcode)
        }

    except requests.RequestException as e:
        print(f"OFF submission error: {e}")
        return {
            "success": False,
            "message": str(e),
            "status_verbose": "network_error"
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"OFF response parse error: {e}")
        return {
            "success": False,
            "message": "Invalid response from OFF",
            "status_verbose": "parse_error"
        }


def fuzzy_search_off(query: str, barcode_prefix: Optional[str] = None) -> list:
    """
    Search OpenFoodFacts by product name for fuzzy matching.
    Tries India-specific subdomain first, then global.

    barcode_prefix: first 3 digits of barcode (e.g. "890" for India)
    Used to pre-filter results for more accurate matching.
    """
    results = []

    search_params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 20,
        "fields": "code,product_name,brands,image_url,nutrition_grades,nova_group"
    }

    if barcode_prefix:
        search_params["tagtype_0"] = "countries"
        search_params["tag_0"] = "india"

    # Try India subdomain first
    for base_url in [OFF_IN_SEARCH_URL, OFF_SEARCH_URL]:
        try:
            response = requests.get(base_url, params=search_params, timeout=8)
            if response.status_code == 200:
                data = response.json()
                products = data.get("products", [])
                if products:
                    results = products
                    break
        except requests.RequestException:
            continue

    return results

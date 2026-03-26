"""
off_submit.py
Submits user-contributed product data to OpenFoodFacts via their write API.

Key fixes:
- Added required User-Agent header (OFF returns 403 without it)
- Added product name + image URL from Google search results
- Improved error handling and logging
"""

import os
import requests
import json
from typing import Optional

OFF_USERNAME = os.environ.get("OFF_USERNAME", "")
OFF_PASSWORD = os.environ.get("OFF_PASSWORD", "")

OFF_WRITE_URL  = "https://world.openfoodfacts.org/cgi/product_jqm2.pl"
OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
OFF_IN_SEARCH_URL = "https://in.openfoodfacts.org/cgi/search.pl"

# OFF REQUIRES this header — returns 403 without a valid User-Agent
# Format: app_name/version (contact_email)
OFF_USER_AGENT = "SmartScan/1.0 (smartscan.app@gmail.com)"


def submit_product_to_off(
    barcode:          str,
    product_name:     Optional[str] = None,
    nutriments:       Optional[dict] = None,
    ingredients_text: Optional[str] = None,
    brands:           Optional[str] = None,
    quantity:         Optional[str] = None,
    image_url:        Optional[str] = None,   # from Google search API
) -> dict:
    """
    Submit a product to OpenFoodFacts.
    
    OFF write API requires:
    1. Valid User-Agent header (app_name/version contact)
    2. user_id + password authentication
    3. product code
    
    image_url: if provided from Google search, we fetch and re-upload
    the image to OFF so the product has a photo immediately.
    """
    if not OFF_USERNAME or not OFF_PASSWORD:
        print("OFF: credentials not configured, skipping submission")
        return {"success": False, "message": "OFF credentials not configured", "status_verbose": "skipped"}

    if not barcode:
        return {"success": False, "message": "No barcode provided", "status_verbose": "skipped"}

    headers = {
        "User-Agent": OFF_USER_AGENT,
    }

    payload = {
        "code":        barcode,
        "user_id":     OFF_USERNAME,
        "password":    OFF_PASSWORD,
        "comment":     "SmartScan app user contribution — Indian market product",
        "app_name":    "SmartScan",
        "app_version": "1.0",
        "app_uuid":    "smartscan-india-v1",
        "lang":        "en",
        "countries":   "en:india",
        "countries_tags": "en:india",
    }

    if product_name:
        payload["product_name"]    = product_name
        payload["product_name_en"] = product_name

    if brands:
        payload["brands"] = brands

    if quantity:
        payload["quantity"] = quantity

    if ingredients_text:
        payload["ingredients_text"]    = ingredients_text
        payload["ingredients_text_en"] = ingredients_text

    if nutriments:
        off_nutriment_map = {
            "energy-kcal_100g":           ("energy-kcal",        "kcal"),
            "proteins_100g":              ("proteins",           "g"),
            "carbohydrates_100g":         ("carbohydrates",      "g"),
            "sugars_100g":                ("sugars",             "g"),
            "fat_100g":                   ("fat",                "g"),
            "saturated-fat_100g":         ("saturated-fat",      "g"),
            "trans-fat_100g":             ("trans-fat",          "g"),
            "monounsaturated-fat_100g":   ("monounsaturated-fat","g"),
            "polyunsaturated-fat_100g":   ("polyunsaturated-fat","g"),
            "fiber_100g":                 ("fiber",              "g"),
            "sodium_100g":                ("sodium",             "g"),
            "salt_100g":                  ("salt",               "g"),
            "cholesterol_100g":           ("cholesterol",        "g"),
            "calcium_100g":               ("calcium",            "g"),
            "iron_100g":                  ("iron",               "g"),
            "vitamin-c_100g":             ("vitamin-c",          "mg"),
        }

        for internal_key, value in nutriments.items():
            if internal_key in off_nutriment_map:
                off_key, unit = off_nutriment_map[internal_key]
                payload[f"nutriment_{off_key}"]      = str(round(float(value), 3))
                payload[f"nutriment_{off_key}_unit"]  = unit
                payload[f"nutriment_{off_key}_100g"]  = str(round(float(value), 3))

        payload["nutrition_data_per"] = "100g"

    print(f"OFF: submitting barcode={barcode} name={product_name}")

    try:
        response = requests.post(
            OFF_WRITE_URL,
            data=payload,
            headers=headers,
            timeout=20
        )

        print(f"OFF: response status={response.status_code}")

        if response.status_code == 403:
            print("OFF: 403 Forbidden — check User-Agent header and credentials")
            return {
                "success":        False,
                "message":        "OFF returned 403 — check credentials and User-Agent",
                "status_verbose": "auth_error"
            }

        response.raise_for_status()
        result = response.json()

        success = result.get("status") == 1
        print(f"OFF: submission result={result.get('status_verbose', 'unknown')}")

        # If we have an image URL from Google, upload it separately
        if success and image_url:
            _upload_image_to_off(barcode, image_url, headers)

        return {
            "success":        success,
            "message":        result.get("status_verbose", ""),
            "status_verbose": result.get("status_verbose", ""),
            "product_id":     result.get("code", barcode)
        }

    except requests.HTTPError as e:
        print(f"OFF: HTTP error {e}")
        return {"success": False, "message": str(e), "status_verbose": "http_error"}
    except requests.RequestException as e:
        print(f"OFF: network error {e}")
        return {"success": False, "message": str(e), "status_verbose": "network_error"}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"OFF: parse error {e}")
        return {"success": False, "message": "Invalid response from OFF", "status_verbose": "parse_error"}


def _upload_image_to_off(barcode: str, image_url: str, headers: dict):
    """
    Fetch image from Google search result URL and upload to OFF.
    OFF requires multipart image upload to a separate endpoint.
    """
    try:
        # Fetch the image
        img_response = requests.get(image_url, timeout=10, headers={"User-Agent": OFF_USER_AGENT})
        if img_response.status_code != 200:
            print(f"OFF image: failed to fetch from {image_url}")
            return

        content_type = img_response.headers.get("Content-Type", "image/jpeg")
        ext = "jpg" if "jpeg" in content_type or "jpg" in content_type else "png"

        upload_url = "https://world.openfoodfacts.org/cgi/product_image_upload.pl"
        files = {
            "imgupload_front": (f"front.{ext}", img_response.content, content_type)
        }
        data = {
            "code":        barcode,
            "user_id":     OFF_USERNAME,
            "password":    OFF_PASSWORD,
            "imagefield":  "front",
        }

        r = requests.post(upload_url, files=files, data=data, headers=headers, timeout=20)
        print(f"OFF image upload: status={r.status_code}")

    except Exception as e:
        print(f"OFF image upload failed: {e}")


def fuzzy_search_off(query: str, barcode_prefix: Optional[str] = None) -> list:
    """Search OpenFoodFacts by product name."""
    search_params = {
        "search_terms": query,
        "search_simple": 1,
        "action":        "process",
        "json":          1,
        "page_size":     20,
        "fields":        "code,product_name,brands,image_url,nutrition_grades,nova_group"
    }

    if barcode_prefix:
        search_params["tagtype_0"] = "countries"
        search_params["tag_0"]     = "india"

    headers = {"User-Agent": OFF_USER_AGENT}

    for base_url in [OFF_IN_SEARCH_URL, OFF_SEARCH_URL]:
        try:
            response = requests.get(base_url, params=search_params,
                                    headers=headers, timeout=8)
            if response.status_code == 200:
                data     = response.json()
                products = data.get("products", [])
                if products:
                    return products
        except requests.RequestException:
            continue

    return []
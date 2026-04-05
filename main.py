from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from services import fetch_product_from_google, openFoodAPI_fetch
from parse_additives import parse_additives
from ai import evaluate_product
from ocr_parser import process_ocr_scan, _fuzzy_product_name_match
from vision_service import extract_vision_data
from off_submit import submit_product_to_off, fuzzy_search_off

app = FastAPI()


# ── Request models ────────────────────────────────────────────────────────────

class UserPreferences(BaseModel):
    restrictions: Optional[List[str]] = []
    allergens:    Optional[List[str]] = []
    goals:        Optional[List[str]] = []


class BarcodeRequest(BaseModel):
    barcode:      str
    barcode_type: Optional[str] = None
    user_prefs:   Optional[UserPreferences] = None


class OcrScanRequest(BaseModel):
    barcode:        str
    scan_type:      str
    image_base64:   str
    product_name:   Optional[str]  = None
    existing_data:  Optional[dict] = None
    # FIX M-3: accept user_prefs so the AI evaluation can personalise results
    user_prefs:     Optional[UserPreferences] = None


class FuzzySearchRequest(BaseModel):
    query:          str
    barcode_prefix: Optional[str] = None


class ProductSubmitRequest(BaseModel):
    barcode:          str
    product_name:     Optional[str]  = None
    nutriments:       Optional[dict] = None
    ingredients_text: Optional[str]  = None
    brands:           Optional[str]  = None
    quantity:         Optional[str]  = None
    # FIX M-1: was missing — endpoint read request.existing_data → AttributeError crash
    existing_data:    Optional[dict] = None


# ── Existing product analysis endpoint ───────────────────────────────────────

@app.get("/")
def root():
    return {"status": "SmartScan backend running"}


@app.post("/analyze-product")
def analyze_product(request: BarcodeRequest):
    try:
        print(f"\nIncoming request: barcode={request.barcode}")

        google_data = fetch_product_from_google(request.barcode)
        food_data   = openFoodAPI_fetch(request.barcode)

        print(f"Google data: {'found' if google_data else 'None'}")
        print(f"OFF data:    {'found' if food_data else 'None'}")

        if not google_data and not food_data:
            return {
                "error": "Product not found",
                "missing_data": True,
                "barcode": request.barcode
            }

        barcode_image_url = None
        if request.barcode_type:
            barcode_image_url = (
                f"https://barcode.orcascan.com/"
                f"?type={request.barcode_type}"
                f"&data={request.barcode}"
                f"&format=jpg"
            )

        user_prefs_dict = {}
        if request.user_prefs:
            user_prefs_dict = {
                "restrictions": request.user_prefs.restrictions or [],
                "allergens":    request.user_prefs.allergens    or [],
                "goals":        request.user_prefs.goals        or [],
            }
        print(f"User prefs: {user_prefs_dict}")

        product_name = (
            (google_data or {}).get("product_name")
            or (food_data  or {}).get("product_name")
            or None
        )
        print(f"Product name resolved: '{product_name}'")

        ingredients_text = food_data.get("ingredients") if food_data else None
        additives = parse_additives(ingredients_text) if ingredients_text else []

        _KEY_NUTRIENTS = {"energy-kcal_100g", "proteins_100g", "carbohydrates_100g", "fat_100g"}
        raw_nutriments    = food_data.get("nutriments") if food_data else None
        has_key_nutrients = bool(raw_nutriments and any(k in raw_nutriments for k in _KEY_NUTRIENTS))
        missing_nutrition   = not has_key_nutrients
        missing_ingredients = not food_data or not food_data.get("ingredients")
        print(f"missing_nutrition={missing_nutrition} missing_ingredients={missing_ingredients}")

        ai_result = None
        if food_data and (ingredients_text or has_key_nutrients):
            ai_result = evaluate_product(
                ingredients     = ingredients_text,
                nutriments      = food_data.get("nutriments") if food_data else None,
                additives       = additives,
                nutrition_grade = food_data.get("nutrition_grade") if food_data else None,
                nova_group      = food_data.get("nova_group") if food_data else None,
                user_prefs      = user_prefs_dict,
            )
            print(f"AI result: {ai_result}")
        else:
            print("AI skipped: insufficient data (no ingredients and no key nutrients)")

        return {
            "barcode":           request.barcode,
            "product_name":      product_name,
            "image_url":         google_data.get("image_url") if google_data else None,
            "barcode_image_url": barcode_image_url,
            "ingredients":       ingredients_text,
            "nutrition_grade":   food_data.get("nutrition_grade")  if food_data else None,
            "nutriscore_data":   food_data.get("nutriscore_data")  if food_data else None,
            "nutriments":        food_data.get("nutriments")       if food_data else None,
            "nova_group":        food_data.get("nova_group")       if food_data else None,
            "allergens":         food_data.get("allergens")        if food_data else None,
            "additives":         additives,
            "recommendation":    ai_result.get("recommendation") if ai_result else None,
            "ai_evaluation":     ai_result.get("evaluation")     if ai_result else None,
            "carcinogenic":      ai_result.get("carcinogenic")   if ai_result else False,
            "ai_allergens":      ai_result.get("allergens")      if ai_result else [],
            "diet":              ai_result.get("diet")           if ai_result else None,
            "missing_nutrition":   missing_nutrition,
            "missing_ingredients": missing_ingredients,
        }

    except Exception as e:
        print(f"CRASH: {e}")
        return {"error": "Internal server error"}


# ── OCR scan endpoint ─────────────────────────────────────────────────────────

@app.post("/ocr-scan")
async def ocr_scan(request: OcrScanRequest, background_tasks: BackgroundTasks):
    """
    Accepts a base64 image and scan_type, runs Vision API + parsing pipeline.
    Returns structured nutrition or ingredients data.

    After returning to the client, submits data to OFF asynchronously.
    """
    try:
        print(f"\nOCR scan: barcode={request.barcode} type={request.scan_type}")

        # ── Step 1: Vision API ────────────────────────────────────────────────
        vision_data = extract_vision_data(request.image_base64)

        if not vision_data.get("success"):
            return {
                "success":    False,
                "scan_type":  request.scan_type,
                "error":      vision_data.get(
                    "error",
                    "Could not extract text from image. "
                    "Ensure good lighting and that the label fills the frame.",
                ),
                "data":       None,
                "confidence": 0.0,
                "warnings":   [],
            }

        print(
            f"Vision API: {len(vision_data.get('text', ''))} chars  "
            f"{len(vision_data.get('words', []))} words  "
            f"orientation={vision_data.get('orientation', 0)}°"
        )

        # ── Step 2: Parse and normalise ───────────────────────────────────────
        result = process_ocr_scan(vision_data, request.scan_type)

        print(f"Parse result: success={result['success']} confidence={result['confidence']}")

        # ── Step 3: Additives if ingredients scan ─────────────────────────────
        additives = []
        if request.scan_type == 'ingredients' and result['success']:
            additives = parse_additives(result['data'])

        # ── Step 4: AI evaluation ─────────────────────────────────────────────
        # FIX M-3: pass user_prefs through so recommendations are personalised
        ai_result = None
        if result['success']:
            existing    = request.existing_data or {}
            nutriments  = (
                result['data'] if request.scan_type == 'nutrition'
                else existing.get('nutriments')
            )
            ingredients = (
                result['data'] if request.scan_type == 'ingredients'
                else existing.get('ingredients')
            )

            # Build user_prefs dict from the request (was always {} before)
            user_prefs_dict = {}
            if request.user_prefs:
                user_prefs_dict = {
                    "restrictions": request.user_prefs.restrictions or [],
                    "allergens":    request.user_prefs.allergens    or [],
                    "goals":        request.user_prefs.goals        or [],
                }

            if nutriments or ingredients:
                try:
                    ai_result = evaluate_product(
                        ingredients     = ingredients,
                        nutriments      = nutriments,
                        additives       = additives or existing.get('additives', []),
                        nutrition_grade = existing.get('nutrition_grade'),
                        nova_group      = existing.get('nova_group'),
                        user_prefs      = user_prefs_dict,   # FIX M-3
                    )
                except Exception as e:
                    print(f"AI eval failed: {e}")

        # ── Step 5: Submit to OFF in background ───────────────────────────────
        if result['success']:
            # FIX M-2: pass image_url from existing_data so OFF gets the image
            existing   = request.existing_data or {}
            image_url  = existing.get('image_url')

            background_tasks.add_task(
                _submit_to_off_background,
                barcode       = request.barcode,
                product_name  = request.product_name,
                scan_type     = request.scan_type,
                parsed_data   = result['data'],
                existing_data = request.existing_data,
                image_url     = image_url,           # FIX M-2: was missing
            )

        return {
            "success":    result['success'],
            "scan_type":  request.scan_type,
            "data":       result['data'],
            "confidence": result['confidence'],
            "warnings":   result['warnings'],
            "additives":  additives,
            "ai_result":  ai_result,
            "raw_text":   result['raw_text'][:500] if result.get('raw_text') else None,
        }

    except Exception as e:
        print(f"OCR CRASH: {e}")
        return {
            "success":    False,
            "scan_type":  request.scan_type,
            "error":      "Internal server error during OCR processing",
            "data":       None,
            "confidence": 0.0,
            "warnings":   [],
        }


def _submit_to_off_background(
    barcode:          str,
    product_name:     Optional[str],
    scan_type:        str,
    parsed_data,
    existing_data:    Optional[dict],
    image_url:        Optional[str] = None,
):
    """Background task — runs after response is sent to client."""
    try:
        nutriments       = parsed_data if scan_type == 'nutrition'   else None
        ingredients_text = parsed_data if scan_type == 'ingredients' else None

        if existing_data:
            if not nutriments:
                nutriments = existing_data.get('nutriments')
            if not ingredients_text:
                ingredients_text = existing_data.get('ingredients')
            if not image_url:
                image_url = existing_data.get('image_url')

        result = submit_product_to_off(
            barcode          = barcode,
            product_name     = product_name,
            nutriments       = nutriments,
            ingredients_text = ingredients_text,
            image_url        = image_url,
        )
        print(f"OFF submission: {result}")
    except Exception as e:
        print(f"OFF background submission failed: {e}")


# ── Fuzzy search endpoint ─────────────────────────────────────────────────────

@app.post("/fuzzy-search")
def fuzzy_search(request: FuzzySearchRequest):
    """
    Search OFF by product name for "Did you mean?" suggestions.
    Returns top 5 matches with similarity scores.
    """
    try:
        if not request.query or len(request.query.strip()) < 2:
            return {"matches": [], "query": request.query}

        candidates = fuzzy_search_off(
            query          = request.query,
            barcode_prefix = request.barcode_prefix,
        )

        if not candidates:
            return {"matches": [], "query": request.query}

        matches = _fuzzy_product_name_match(
            query      = request.query,
            candidates = candidates,
            threshold  = 0.35,
        )

        clean_matches = []
        for m in matches:
            clean_matches.append({
                "barcode":         m.get("code", ""),
                "product_name":    m.get("product_name", ""),
                "brands":          m.get("brands", ""),
                "image_url":       m.get("image_url", ""),
                "nutrition_grade": m.get("nutrition_grades", ""),
                "nova_group":      m.get("nova_group"),
                "match_score":     m.get("_match_score", 0),
            })

        return {
            "matches": clean_matches,
            "query":   request.query,
        }

    except Exception as e:
        print(f"Fuzzy search CRASH: {e}")
        return {"matches": [], "query": request.query, "error": str(e)}


# ── Manual product submit endpoint ───────────────────────────────────────────

@app.post("/submit-product")
def submit_product(request: ProductSubmitRequest, background_tasks: BackgroundTasks):
    """
    Manually submit complete product data.
    Used when user fills in product details manually.
    Submits to OFF asynchronously.
    """
    try:
        # FIX M-1: request.existing_data now exists (field added to model above)
        image_url = (request.existing_data or {}).get('image_url')

        background_tasks.add_task(
            submit_product_to_off,
            barcode          = request.barcode,
            product_name     = request.product_name,
            nutriments       = request.nutriments,
            ingredients_text = request.ingredients_text,
            brands           = request.brands,
            quantity         = request.quantity,
            image_url        = image_url,
        )
        return {"success": True, "message": "Product data received and queued for submission"}
    except Exception as e:
        return {"success": False, "message": str(e)}
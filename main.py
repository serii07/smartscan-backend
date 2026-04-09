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
    existing_data:    Optional[dict] = None

# NEW: Model for re-evaluating corrected data
class EvaluateRequest(BaseModel):
    ingredients:      Optional[str] = None
    nutriments:       Optional[dict] = None
    nutrition_grade:  Optional[str] = None
    nova_group:       Optional[int] = None
    user_prefs:       Optional[UserPreferences] = None

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

        if not google_data and not food_data:
            return {"error": "Product not found", "missing_data": True, "barcode": request.barcode}

        barcode_image_url = None
        if request.barcode_type:
            barcode_image_url = f"https://barcode.orcascan.com/?type={request.barcode_type}&data={request.barcode}&format=jpg"

        user_prefs_dict = {}
        if request.user_prefs:
            user_prefs_dict = {
                "restrictions": request.user_prefs.restrictions or [],
                "allergens":    request.user_prefs.allergens    or [],
                "goals":        request.user_prefs.goals        or [],
            }

        product_name = (google_data or {}).get("product_name") or (food_data or {}).get("product_name") or None
        ingredients_text = food_data.get("ingredients") if food_data else None
        additives = parse_additives(ingredients_text) if ingredients_text else []

        _KEY_NUTRIENTS = {"energy-kcal_100g", "proteins_100g", "carbohydrates_100g", "fat_100g"}
        raw_nutriments    = food_data.get("nutriments") if food_data else None
        has_key_nutrients = bool(raw_nutriments and any(k in raw_nutriments for k in _KEY_NUTRIENTS))
        missing_nutrition   = not has_key_nutrients
        missing_ingredients = not food_data or not food_data.get("ingredients")

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
        return {"error": "Internal server error"}

# ── OCR scan endpoint ─────────────────────────────────────────────────────────

@app.post("/ocr-scan")
async def ocr_scan(request: OcrScanRequest):
    """
    Accepts a base64 image and scan_type, runs Vision API + parsing pipeline.
    Does NOT submit to OFF or run AI. Just returns parsed text.
    """
    try:
        vision_data = extract_vision_data(request.image_base64)

        if not vision_data.get("success"):
            return {
                "success":    False,
                "scan_type":  request.scan_type,
                "error":      vision_data.get("error", "Could not extract text."),
                "data":       None,
                "confidence": 0.0,
                "warnings":   [],
            }

        result = process_ocr_scan(vision_data, request.scan_type)
        additives = parse_additives(result['data']) if request.scan_type == 'ingredients' and result['success'] else []

        return {
            "success":    result['success'],
            "scan_type":  request.scan_type,
            "data":       result['data'],
            "confidence": result['confidence'],
            "warnings":   result['warnings'],
            "additives":  additives,
            "ai_result":  None, # Removed AI execution from this step
            "raw_text":   result['raw_text'][:500] if result.get('raw_text') else None,
        }

    except Exception as e:
        return {
            "success":    False,
            "scan_type":  request.scan_type,
            "error":      "Internal server error during OCR processing",
            "data":       None,
            "confidence": 0.0,
            "warnings":   [],
        }

# ── NEW: Evaluation Endpoint ──────────────────────────────────────────────────

@app.post("/evaluate")
def evaluate_manually_edited_data(request: EvaluateRequest):
    """
    Called by Android AFTER the user finishes the OCR Review screen.
    Re-parses additives and recalculates the AI based on the verified data.
    """
    try:
        additives = parse_additives(request.ingredients) if request.ingredients else []

        user_prefs_dict = {}
        if request.user_prefs:
            user_prefs_dict = {
                "restrictions": request.user_prefs.restrictions or [],
                "allergens":    request.user_prefs.allergens    or [],
                "goals":        request.user_prefs.goals        or [],
            }

        ai_result = evaluate_product(
            ingredients     = request.ingredients,
            nutriments      = request.nutriments,
            additives       = additives,
            nutrition_grade = request.nutrition_grade,
            nova_group      = request.nova_group,
            user_prefs      = user_prefs_dict
        )

        return {
            "success": True,
            "additives": additives,
            "ai_result": ai_result
        }
    except Exception as e:
        print(f"Evaluation CRASH: {e}")
        return {"success": False, "error": str(e)}

# ── Fuzzy search endpoint ─────────────────────────────────────────────────────

@app.post("/fuzzy-search")
def fuzzy_search(request: FuzzySearchRequest):
    try:
        if not request.query or len(request.query.strip()) < 2:
            return {"matches": [], "query": request.query}
        candidates = fuzzy_search_off(query=request.query, barcode_prefix=request.barcode_prefix)
        if not candidates:
            return {"matches": [], "query": request.query}
        matches = _fuzzy_product_name_match(query=request.query, candidates=candidates, threshold=0.35)
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
        return {"matches": clean_matches, "query": request.query}
    except Exception as e:
        return {"matches": [], "query": request.query, "error": str(e)}

# ── Manual product submit endpoint ───────────────────────────────────────────

@app.post("/submit-product")
def submit_product(request: ProductSubmitRequest, background_tasks: BackgroundTasks):
    try:
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
        return {"success": True, "message": "Product data queued for submission"}
    except Exception as e:
        return {"success": False, "message": str(e)}
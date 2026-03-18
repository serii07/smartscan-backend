from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from services import fetch_product_from_google, openFoodAPI_fetch
from parse_additives import parse_additives
from ai import evaluate_product

app = FastAPI()


class UserPreferences(BaseModel):
    restrictions: Optional[List[str]] = []
    allergens:    Optional[List[str]] = []
    goals:        Optional[List[str]] = []


class BarcodeRequest(BaseModel):
    barcode:      str
    barcode_type: Optional[str] = None
    user_prefs:   Optional[UserPreferences] = None


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
            return {"error": "Product not found"}

        barcode_image_url = None
        if request.barcode_type:
            barcode_image_url = (
                f"https://barcode.orcascan.com/"
                f"?type={request.barcode_type}"
                f"&data={request.barcode}"
                f"&format=jpg"
            )

        ingredients_text = food_data.get("ingredients") if food_data else None
        additives = parse_additives(ingredients_text) if ingredients_text else []
        print(f"Additives found: {len(additives)}")

        ai_result = None
        if food_data:
            user_prefs_dict = {}
            if request.user_prefs:
                user_prefs_dict = {
                    "restrictions": request.user_prefs.restrictions or [],
                    "allergens":    request.user_prefs.allergens    or [],
                    "goals":        request.user_prefs.goals        or [],
                }
            ai_result = evaluate_product(
                ingredients     = ingredients_text,
                nutriments      = food_data.get("nutriments"),
                additives       = additives,
                nutrition_grade = food_data.get("nutrition_grade"),
                nova_group      = food_data.get("nova_group"),
                user_prefs      = user_prefs_dict
            )
            print(f"AI result: {ai_result}")

        return {
            "barcode":           request.barcode,
            "product_name":      google_data.get("product_name") if google_data else None,
            "image_url":         google_data.get("image_url")    if google_data else None,
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
        }

    except Exception as e:
        print(f"CRASH: {e}")
        return {"error": "Internal server error"}
from fastapi import FastAPI
from pydantic import BaseModel
from services import fetch_product_from_google, openFoodAPI_fetch

app = FastAPI()

class BarcodeRequest(BaseModel):
    barcode: str
    barcode_type: str | None = None

@app.get("/")
def root():
    return {"status": "SmartScan backend running"}

@app.post("/analyze-product")
def analyze_product(request: BarcodeRequest):
    try:
        print(f"\n Incoming request for barcode: {request.barcode}")

        google_data = fetch_product_from_google(request.barcode)
        food_data = openFoodAPI_fetch(request.barcode)
        if food_data:
            print("OpenFoodFacts success")
            print(f"Ingredients: {food_data.get('ingredients')}")
        else:
            print("OpenFoodFacts failed")

        if not google_data and not food_data:
            print("No data")
            return {"error": "Product not found"}

        barcode_image_url = None
        if request.barcode_type:
            barcode_image_url = (
                f"https://barcode.orcascan.com/"
                f"?type={request.barcode_type}"
                f"&data={request.barcode}"
                f"&format=jpg"
            )

        final_response = {
            "barcode": request.barcode,
            "product_name": google_data.get("product_name") if google_data else None,
            "image_url": google_data.get("image_url") if google_data else None,
            "barcode_image_url": barcode_image_url,
            "ingredients": food_data.get("ingredients") if food_data else None,
            "nutriscore": food_data.get("nutriscore") if food_data else None,
            "nova_group": food_data.get("nova_group") if food_data else None,
            "countries": food_data.get("countries") if food_data else None,
            "quantity": food_data.get("quantity") if food_data else None,
        }

        return final_response

    except Exception as e:
        print("CRASH:", e)
        return {"error": "Internal server error"}

from fastapi import FastAPI
from pydantic import BaseModel
from services import fetch_product_from_google

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
        result = fetch_product_from_google(request.barcode)

        if not result:
            return {"error": "Product not found"}

        barcode_image_url = None
        if request.barcode_type:
            barcode_image_url = f"https://barcode.orcascan.com/?type={request.barcode_type}&data={request.barcode}&format=jpg"

        result["barcode_image_url"] = barcode_image_url

        return result

    except Exception as e:
        print("CRASH:", e)
        return {"error": "Internal server error"}


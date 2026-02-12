import requests
import os

GOOGLE_API_KEY = "AIzaSyDNkEdAnqEn5gPXxA3O_xmZJsEcONlcwFY"
SEARCH_ENGINE_ID = "94a4ecdfb53ec46ca"

def clean_product_name(raw_product_name: str) -> str:
    cleaned = raw_product_name.strip()

    blacklist = [
        "buy", "online", "price", "at", "best", ":", "|",
        "amazon.in", "flipkart.com", "aap ka bazar", "beauty"
    ]

    words = cleaned.split()
    words = [w for w in words if w.lower() not in blacklist]

    return " ".join(words)

def fetch_product_from_google(barcode: str):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={barcode}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"

    response = requests.get(search_url)
    data = response.json()

    items = data.get("items")

    if not items:
        return None

    for item in items:
        title = item.get("title", "Unknown Product")

        pagemap = item.get("pagemap", {})
        cse_images = pagemap.get("cse_image")

        if cse_images:
            image_url = cse_images[0].get("src")

            if image_url and "transparentImg" not in image_url:
                return {
                    "barcode": barcode,
                    "product_name": clean_product_name(title),
                    "image_url": image_url
                }

    return None

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

    if response.status_code != 200:
        return None

    data = response.json()
    items = data.get("items")

    if not items:
        return None

    for item in items:
        title = item.get("title", "Unknown Product")

        pagemap = item.get("pagemap")
        if not pagemap:
            continue

        cse_images = pagemap.get("cse_image")
        if not cse_images:
            continue

        image_url = cse_images[0].get("src")
        if not image_url:
            continue

        image_url_lower = image_url.lower()

        if image_url_lower.endswith(".svg"):
            continue

        if "transparentimg" in image_url_lower:
            continue

        if "logo" in image_url_lower:
            continue

        return {
            "barcode": barcode,
            "product_name": clean_product_name(title),
            "image_url": image_url
        }

    return None


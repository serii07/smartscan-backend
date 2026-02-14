import requests
import os

GOOGLE_API_KEY = "AIzaSyDNkEdAnqEn5gPXxA3O_xmZJsEcONlcwFY"
SEARCH_ENGINE_ID = "94a4ecdfb53ec46ca"

def clean_product_name(raw_product_name: str) -> str:
    cleaned = raw_product_name.strip()

    blacklist = [
        "buy", "online", "price", "at", "best", ":", "|",
        "amazon.in", "flipkart.com", "aap ka bazar", "beauty", "BigBasket",
        "Aapkabazar.co", "Amazon"
    ]

    words = cleaned.split()
    words = [w for w in words if w.lower() not in blacklist]

    return " ".join(words)

def fetch_product_from_google(barcode: str):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={barcode}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"

    response = requests.get(search_url, timeout=5)

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

def openFoodAPI_fetch(barcode: str):
    url = (
        f"https://world.openfoodfacts.net/api/v2/product/{barcode}"
        "?fields=product_name,"
        "nutrition_grades,"
        "nutriscore_data,"
        "nutriments,"
        "ingredients_text,"
        "nova_group,"
        "allergens,"
        "quantity,"
        "brands,"
        "misc_tags"
    )


    try:
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return None

        data = response.json()
    
        if data.get("status") != 1:
            return None

        product = data.get("product", {})

        return {
            "ingredients": product.get("ingredients_text"),
            "nutrition_grade": product.get("nutrition_grades"),
            "nutriscore_data": product.get("nutriscore_data"),
            "nutriments": product.get("nutriments"),
            "nova_group": product.get("nova_group"),
            "allergens": product.get("allergens")
        }

    except requests.RequestException:
        return None
import re
import requests

GOOGLE_API_KEY = "AIzaSyDNkEdAnqEn5gPXxA3O_xmZJsEcONlcwFY"
SEARCH_ENGINE_ID = "94a4ecdfb53ec46ca"

# ── Patterns and substrings to strip from product titles ─────────────────────
# These are checked as case-insensitive substrings anywhere in the title,
# so "Buy on BigBasket" and "bigbasket.com" both get caught.

_BLACKLIST_SUBSTRINGS = [
    "amazon", "flipkart", "bigbasket", "aapkabazar", "aap ka bazar",
    "jiomart", "blinkit", "zepto", "swiggy instamart", "dunzo",
    "grofers", "milkbasket", "meesho", "snapdeal", "myntra",
    "1mg", "netmeds", "pharmeasy", "healthkart", "nykaa",
    "buy online", "buy now", "order online", "shop online",
    "best price", "lowest price", "check price", "price in india",
    "free delivery", "free shipping", "cash on delivery",
    "at best", "deals", "offers", "discount",
    "review", "reviews", "rating", "ratings",
]

# Characters / trailing patterns to strip
_STRIP_AFTER = ["–", "—", " - ", " | ", " : ", "||", ">>"]


def clean_product_name(raw_product_name: str) -> str:
    if not raw_product_name:
        return ""

    text = raw_product_name.strip()

    # Step 1: Cut off at any separator that typically precedes site names
    for sep in _STRIP_AFTER:
        if sep in text:
            text = text.split(sep)[0].strip()

    # Step 2: Remove blacklisted substrings (case-insensitive)
    lower = text.lower()
    for phrase in _BLACKLIST_SUBSTRINGS:
        if phrase in lower:
            # Remove the phrase and re-check
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            text = pattern.sub("", text).strip()
            lower = text.lower()

    # Step 3: Clean up leftover punctuation at start/end
    text = re.sub(r'^[\s\-\|:,\.]+', '', text)
    text = re.sub(r'[\s\-\|:,\.]+$', '', text)

    # Step 4: Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text if text else raw_product_name.strip()


def fetch_product_from_google(barcode: str):
    search_url = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?q={barcode}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
    )
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


def _round_nutriments(nutriments: dict) -> dict:
    """
    Round all numeric nutriment values to 2 decimal places.
    Non-numeric values are passed through unchanged.
    """
    if not nutriments:
        return nutriments
    rounded = {}
    for key, value in nutriments.items():
        if isinstance(value, float):
            rounded[key] = round(value, 2)
        elif isinstance(value, int):
            rounded[key] = value  # integers stay as-is (e.g. 0)
        else:
            rounded[key] = value
    return rounded


def openFoodAPI_fetch(barcode: str):
    # Use world.openfoodfacts.org (production). The .net domain is a staging
    # mirror — data submitted via the write API goes to .org and may take
    # hours or never appear on .net, which broke the post-OCR rescan flow.
    url = (
        f"https://world.openfoodfacts.org/api/v2/product/{barcode}"
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
        raw_nutriments = product.get("nutriments")

        return {
            "product_name": product.get("product_name"),   # Issue 4: expose for name fallback
            "ingredients": product.get("ingredients_text"),
            "nutrition_grade": product.get("nutrition_grades"),
            "nutriscore_data": product.get("nutriscore_data"),
            "nutriments": _round_nutriments(raw_nutriments),
            "nova_group": product.get("nova_group"),
            "allergens": product.get("allergens")
        }

    except requests.RequestException:
        return None
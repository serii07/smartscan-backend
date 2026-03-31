import re
import requests

GOOGLE_API_KEY = "AIzaSyDNkEdAnqEn5gPXxA3O_xmZJsEcONlcwFY"
SEARCH_ENGINE_ID = "94a4ecdfb53ec46ca"

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

_STRIP_AFTER = ["–", "—", " - ", " | ", " : ", "||", ">>"]


def clean_product_name(raw_product_name: str) -> str:
    if not raw_product_name:
        return ""
    text = raw_product_name.strip()
    for sep in _STRIP_AFTER:
        if sep in text:
            text = text.split(sep)[0].strip()
    lower = text.lower()
    for phrase in _BLACKLIST_SUBSTRINGS:
        if phrase in lower:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            text = pattern.sub("", text).strip()
            lower = text.lower()
    text = re.sub(r'^[\s\-\|:,\.]+', '', text)
    text = re.sub(r'[\s\-\|:,\.]+$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else raw_product_name.strip()


def fetch_product_from_google(barcode: str):
    search_url = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?q={barcode}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
    )
    try:
        response = requests.get(search_url, timeout=5)
        print(f"Google fetch: status={response.status_code} barcode={barcode}")
        if response.status_code != 200:
            return None
        data = response.json()
        items = data.get("items")
        if not items:
            print("Google fetch: no items returned")
            return None
        for item in items:
            title = item.get("title", "")
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
            cleaned = clean_product_name(title)
            print(f"Google fetch: found name='{cleaned}'")
            return {"barcode": barcode, "product_name": cleaned, "image_url": image_url}
        print("Google fetch: no usable result after filtering")
        return None
    except requests.RequestException as e:
        print(f"Google fetch: error {e}")
        return None


def _round_nutriments(nutriments: dict) -> dict:
    if not nutriments:
        return nutriments
    rounded = {}
    for key, value in nutriments.items():
        if isinstance(value, float):
            rounded[key] = round(value, 2)
        elif isinstance(value, int):
            rounded[key] = value
        else:
            rounded[key] = value
    return rounded


# ── OpenFoodFacts read ────────────────────────────────────────────────────────
# FIX 1: .net → .org  (.net is staging mirror; write API goes to .org so
#         freshly contributed data was never found on next scan via .net)
# FIX 2: User-Agent header — OFF enforces this on reads. Without it the
#         server returns 403 or a product-not-found response silently.
# FIX 3: Indian server (in.openfoodfacts.org) tried first — much better
#         coverage for 890-prefix barcodes than world.openfoodfacts.org.
# FIX 4: Logging at every step so the backend log shows exactly where it fails.

_OFF_USER_AGENT = "SmartScan/1.0 (smartscan.app@gmail.com)"
_OFF_FIELDS = (
    "product_name,nutrition_grades,nutriscore_data,nutriments,"
    "ingredients_text,nova_group,allergens,quantity,brands,misc_tags"
)
_OFF_ENDPOINTS = [
    "https://in.openfoodfacts.org/api/v2/product/{barcode}?fields=" + _OFF_FIELDS,
    "https://world.openfoodfacts.org/api/v2/product/{barcode}?fields=" + _OFF_FIELDS,
]


def openFoodAPI_fetch(barcode: str):
    headers = {"User-Agent": _OFF_USER_AGENT}

    for url_template in _OFF_ENDPOINTS:
        url = url_template.format(barcode=barcode)
        try:
            print(f"OFF fetch: trying {url[:70]}...")
            response = requests.get(url, headers=headers, timeout=7)
            print(f"OFF fetch: status={response.status_code}")

            if response.status_code != 200:
                print("OFF fetch: non-200, trying next endpoint")
                continue

            data = response.json()
            off_status = data.get("status")
            print(f"OFF fetch: product status={off_status}")

            if off_status != 1:
                print("OFF fetch: product not found on this server, trying next")
                continue

            product = data.get("product", {})
            raw_nutriments = product.get("nutriments")
            print(
                f"OFF fetch: SUCCESS name='{product.get('product_name', '')}' "
                f"has_nutriments={bool(raw_nutriments)} "
                f"has_ingredients={bool(product.get('ingredients_text'))}"
            )

            return {
                "product_name":    product.get("product_name"),
                "ingredients":     product.get("ingredients_text"),
                "nutrition_grade": product.get("nutrition_grades"),
                "nutriscore_data": product.get("nutriscore_data"),
                "nutriments":      _round_nutriments(raw_nutriments),
                "nova_group":      product.get("nova_group"),
                "allergens":       product.get("allergens"),
            }

        except requests.RequestException as e:
            print(f"OFF fetch: network error: {e}")
            continue

    print(f"OFF fetch: all endpoints exhausted for barcode={barcode}")
    return None
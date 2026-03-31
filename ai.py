import os
import json
from openai import OpenAI

# GitHub Models API — set GITHUB_TOKEN as environment variable on Render
_client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ.get("GITHUB_TOKEN", ""),
)

# Which nutriments to include — most clinically relevant only
_NUTRIMENT_KEYS = {
    "energy-kcal_100g":     "kcal",
    "sugars_100g":          "sugar",
    "fat_100g":             "fat",
    "saturated-fat_100g":   "sat_fat",
    "proteins_100g":        "protein",
    "salt_100g":            "salt",
    "carbohydrates_100g":   "carbs",
    "fiber_100g":           "fiber",
}

# Locally detected carcinogenic additives — no AI token cost needed
_CARCINOGENIC_CODES = {
    "e171", "e250", "e251", "e249", "e123", "e128",
    "e284", "e285", "e150d", "e320", "e951",
}

_SYSTEM = """You are a food safety classifier. Output ONLY a JSON object. Zero prose outside JSON.

DEFINITIONS:
- allergens: common allergens PRESENT in ingredients (informational only, never affects recommendation)
- diet: veg=no meat/fish/poultry | nonveg=has meat/fish/poultry/gelatin/lard/tallow/rennet | vegan=no animal products incl dairy/eggs/honey/gelatin/beeswax/carmine | unknown ingredients → default veg
- NEVER classify as nonveg unless an explicitly animal-derived ingredient is present by name

RECOMMENDATION LOGIC (apply in order, first match wins):
1. Skip   → ingredient/additive in BANNED list: E171,E250,E251,E249,E123,E128,E284,E285,E150d,E320
2. Skip   → user has condition:diabetic AND sugar>25
3. Skip   → user has condition:hypertensive AND salt>2
4. Skip   → user listed an allergen AND that allergen is present in ingredients
5. Moderate → any Y-rated additive present OR nova_group=4 OR nutrition_grade=d OR nutrition_grade=e
6. Good Choice → everything else

CARCINOGENIC: true only if E171,E250,E251,E249,E123,E128,E284,E285,E150d,E320 detected

EVALUATION: 1-2 sentences. Cite specific values and codes. Clinical tone. No filler phrases.
If no user preferences: evaluate purely on ingredient quality and nutrition values.
If user preferences exist: evaluate relevance to that user specifically."""


def _build_prompt(
    ingredients: str,
    nutriments: dict,
    additives: list,
    user_prefs: dict
) -> str:

    # Nutrition block
    nutr_parts = []
    for key, label in _NUTRIMENT_KEYS.items():
        val = (nutriments or {}).get(key)
        if val is not None:
            nutr_parts.append(f"{label}:{val}")
    nutr_str = " ".join(nutr_parts) if nutr_parts else "unavailable"

    # Additives block — code + safety initial only
    if additives:
        add_str = " ".join(
            f"{a['code']}({a['safety'][0].upper()})" for a in additives
        )
    else:
        add_str = "none"

    # User preferences
    pref_parts = []
    for k, label in [("restrictions", "conditions"), ("allergens", "avoid"), ("goals", "goal")]:
        vals = user_prefs.get(k, [])
        if vals:
            pref_parts.append(f"{label}:{','.join(vals)}")
    pref_str = " | ".join(pref_parts) if pref_parts else "none"

    # Cap ingredients at 300 chars
    ing_str = (ingredients or "unavailable")[:300]
    if len(ingredients or "") > 300:
        ing_str += "..."

    return (
        f"Ingredients: {ing_str}\n"
        f"Nutrition/100g: {nutr_str}\n"
        f"Additives: {add_str}\n"
        f"User: {pref_str}\n\n"
        f'Reply ONLY with JSON: {{"recommendation":"Good Choice"|"Moderate"|"Skip",'
        f'"evaluation":"1-2 sentences, cite values/codes",'
        f'"carcinogenic":true|false,'
        f'"allergens":["milk"|"eggs"|"wheat"|"soy"|"nuts"|"peanuts"|"fish"|"shellfish"|"sesame"],'
        f'"diet":"veg"|"nonveg"|"vegan"}}'
    )


def _detect_carcinogenic_locally(additives: list) -> bool:
    for a in (additives or []):
        if a.get("code", "").lower().replace(" ", "") in _CARCINOGENIC_CODES:
            return True
    return False


def evaluate_product(
    ingredients: str,
    nutriments: dict,
    additives: list,
    nutrition_grade: str,
    nova_group: int,
    user_prefs: dict = None
) -> dict | None:

    if user_prefs is None:
        user_prefs = {}

    carcinogenic_local = _detect_carcinogenic_locally(additives)
    prompt = _build_prompt(ingredients, nutriments, additives, user_prefs)

    try:
        response = _client.chat.completions.create(
            model="openai/gpt-4o-mini",
            temperature=0.2,
            max_tokens=150,
            top_p=1,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt}
            ]
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them despite instructions
        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)

        # Override carcinogenic with local detection — more reliable than AI
        result["carcinogenic"] = carcinogenic_local or result.get("carcinogenic", False)

        # Validate recommendation value
        if result.get("recommendation") not in {"Good Choice", "Moderate", "Skip"}:
            result["recommendation"] = "Moderate"

        # Ensure all fields always present
        result.setdefault("allergens", [])
        result.setdefault("diet", "unknown")
        result.setdefault("evaluation", "")

        return result

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"AI eval failed: {e}")
        return None
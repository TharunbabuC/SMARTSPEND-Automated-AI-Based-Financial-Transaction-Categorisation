# app/preprocess.py
import re
from difflib import get_close_matches

# Strong rule dictionary (expand as you discover more brands)
STRONG_RULES = {
    # Food
    "mcdonald": "Food & Dining",
    "mcd": "Food & Dining",
    "burger": "Food & Dining",
    "kfc": "Food & Dining",
    "domino": "Food & Dining",
    "zomato": "Food & Dining",
    "swiggy": "Food & Dining",
    "ubereats": "Food & Dining",
    "subway": "Food & Dining",
    "starbucks": "Food & Dining",

    # Groceries
    "bigbasket": "Groceries",
    "dmart": "Groceries",
    "reliance fresh": "Groceries",
    "bbnow": "Groceries",
    "grocery": "Groceries",
    "jio mart": "Groceries",
    "big bazaar": "Groceries",

    # Fuel
    "hpcl": "Fuel",
    "iocl": "Fuel",
    "bpcl": "Fuel",
    "bharat petroleum": "Fuel",
    "shell": "Fuel",
    "petrol": "Fuel",
    "fuel": "Fuel",

    # Shopping
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "myntra": "Shopping",
    "ajio": "Shopping",
    "nykaa": "Shopping",

    # Travel
    "ola": "Travel",
    "uber": "Travel",
    "rapido": "Travel",
    "redbus": "Travel",
    "irctc": "Travel",

    # Bills & Utilities
    "airtel": "Bills & Utilities",
    "reliance jio": "Bills & Utilities",
    "electricity": "Bills & Utilities",
    "water": "Bills & Utilities",
    "broadband": "Bills & Utilities",
    "dth": "Bills & Utilities",

    # Entertainment
    "netflix": "Entertainment",
    "prime video": "Entertainment",
    "bookmyshow": "Entertainment",
    "spotify": "Entertainment",

    # Health & Medical
    "apollo": "Health & Medical",
    "1mg": "Health & Medical",
    "pharmeasy": "Health & Medical",
    "fortis": "Health & Medical",

    # Services
    "urban company": "Services",
    "urbanclap": "Services",
    "laundry": "Services",
    "salon": "Services",
}

# Build a searchable brand list for fuzzy matching
_BRAND_KEYS = list(STRONG_RULES.keys())


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    # Normalize separators and remove unwanted punctuation but keep alphanum and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_boost(text: str):
    """
    Strong exact substring rules first.
    Then simple fuzzy brand matching using difflib.get_close_matches.
    Returns category string if matched, else None.
    """
    if not text:
        return None

    # exact substrings (fast)
    for k, cat in STRONG_RULES.items():
        if k in text:
            return cat

    # token-level fuzzy: split tokens and match to brand keys
    tokens = [t for t in text.split() if len(t) >= 3]
    # try tokens against brand keys
    for t in tokens:
        matches = get_close_matches(t, _BRAND_KEYS, n=1, cutoff=0.85)
        if matches:
            return STRONG_RULES[matches[0]]

    # fallback: try close match on whole string (rare)
    matches = get_close_matches(text, _BRAND_KEYS, n=1, cutoff=0.80)
    if matches:
        return STRONG_RULES[matches[0]]

    return None

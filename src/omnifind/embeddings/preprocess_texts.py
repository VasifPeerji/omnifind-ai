# preprocess_texts.py
"""
Thorough preprocessing pipeline for product titles and user queries.

Key features:
 - unicode/html cleaning
 - unit normalization (", ', in, cm, oz, ml, kg, lb, tb, gb, etc.)
 - slash/hyphen group expansion (20/24/28 -> 20 inch 24 inch 28 inch)
 - parentheses processing (drop SKU-like garbage; expand sizes inside)
 - abbreviation expansion and pack/set normalization (3-pc, 3pk, set of 3)
 - roman numeral normalization (II -> 2, III -> 3) for common cases
 - color normalization
 - marketing & ratings noise removal
 - currency/price removal
 - SKU-like token removal heuristics
 - token-level safe stopword removal (keeps 'in' when used as unit)
 - lemmatization using NLTK's WordNet
"""

from typing import Optional, List
import re
import html
import unicodedata
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# ---------- NLTK setup (download if needed) ----------
# Downloads are quiet and only fetch what's missing.
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("taggers/averaged_perceptron_tagger")
    nltk.data.find("corpora/wordnet")
except Exception:
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

_LEMMATIZER = WordNetLemmatizer()

# ---------- Config / dictionaries ----------
# Abbreviations (conservative)
ABBREV_MAP = {
    r"\bw\/o\b": "without",
    r"\bw\/\b": "with",
    r"\bqty\b": "quantity",
    r"\bqty\.\b": "quantity",
    r"\bpcs\b": "pieces",
    r"\bpcs\.\b": "pieces",
    r"\bpc\b": "piece",
    r"\bpk\b": "pack",
    r"\bpk\b": "pack",
    r"\bpkts\b": "packets",
    r"\bpkg\b": "package",
    r"\bct\b": "count",
    r"\bcts\b": "counts",
    r"\bfl oz\b": "fl_ounce",
    r"\bfl\.?oz\b": "fl_ounce",
    r"\boz\b": "oz",
    r"\bqty\b": "quantity",
    r"\bblk\b": "black",
    r"\bwht\b": "white",
    r"\bgry\b": "gray",
    r"\bgrn\b": "green",
    r"\bbrn\b": "brown",
    r"\bnavy\b": "navy",
    r"\bblu\b": "blue",
    r"\bslv\b": "silver",
    r"\bgld\b": "gold",
    r"\bqty\b": "quantity",
    # storage shorthand
    r"\btb\b": "tb",
    r"\btbsp\b": "tablespoon",
    r"\btsp\b": "teaspoon",
}

# Pack/set patterns (many variations)
_PACK_PATTERNS = [
    (re.compile(r"(?i)\b(\d+)[- ]?(?:pc|pcs|piece|pieces)\b"), r"\1 pieces"),
    (re.compile(r"(?i)\b(\d+)[- ]?p\b"), r"\1 pack"),
    (re.compile(r"(?i)\b(\d+)[- ]?pk\b"), r"\1 pack"),
    (re.compile(r"(?i)\b(\d+)[- ]?ct\b"), r"\1 count"),
    (re.compile(r"(?i)\bset of (\d+)\b"), r"\1 piece set"),
    (re.compile(r"(?i)(\d+)\s*[- ]?\s*piece\s*[- ]?\s*set"), r"\1 piece set"),
    (re.compile(r"(?i)(\d+)[- ]?piece\b"), r"\1 pieces"),
    (re.compile(r"(?i)(\d+)[- ]?pack\b"), r"\1 pack"),
]

# Unit normalization: compiled patterns (number capture + unit)
_UNIT_PATTERNS = [
    # explicit quotes (28" or 28 ”)
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*["“”]'), r"\g<num> inch"),
    # feet (single quote) — only if after a number
    (re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*'"), r"\g<num> feet"),
    # in / inch / inches
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:inches|inch|in)\b', flags=re.I), r"\g<num> inch"),
    # centimeters
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:cm|centimeters|centimetre)\b', flags=re.I), r"\g<num> cm"),
    # millimeters
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:mm|millimeter|millimetre)\b', flags=re.I), r"\g<num> mm"),
    # ml
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:ml|millilitre|milliliter)\b', flags=re.I), r"\g<num> ml"),
    # liter
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:l|ltr|liter|litre)\b', flags=re.I), r"\g<num> l"),
    # kg
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:kg|kilogram|kilograms)\b', flags=re.I), r"\g<num> kg"),
    # grams
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:g|gram|grams|gm|gms)\b', flags=re.I), r"\g<num> g"),
    # lbs / pounds
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)\b', flags=re.I), r"\g<num> lb"),
    # ounces
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces)\b', flags=re.I), r"\g<num> oz"),
    # storage
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:tb|terabyte|terabytes)\b', flags=re.I), r"\g<num> tb"),
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:gb|gigabyte|gigabytes)\b', flags=re.I), r"\g<num> gb"),
    (re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*(?:mb|megabyte|megabytes)\b', flags=re.I), r"\g<num> mb"),
]
# Colors to normalize
_COLOR_MAP = {
    r"\bblk\b": "black",
    r"\bblack\b": "black",
    r"\bwht\b": "white",
    r"\bwhite\b": "white",
    r"\bnavy\b": "navy",
    r"\bblue\b": "blue",
    r"\bred\b": "red",
    r"\bgrn\b": "green",
    r"\bgreen\b": "green",
    r"\bgry\b": "gray",
    r"\bgray\b": "gray",
    r"\bgrey\b": "gray",
    r"\bbeige\b": "beige",
    r"\bbrn\b": "brown",
    r"\bbrown\b": "brown",
    r"\bslate\b": "slate",
    r"\bcharcoal\b": "charcoal",
}

# Noise patterns (ratings, reviews, marketing, promotional fluff)
_NOISE_REGEXES = [
    re.compile(r"\b\d+(\.\d+)?\s*stars?\b", flags=re.I),
    re.compile(r"\b\d+\s*reviews?\b", flags=re.I),
    re.compile(r"\bbestseller\b", flags=re.I),
    re.compile(r"\btop[- ]?rated\b", flags=re.I),
    re.compile(r"\bnew arrival\b", flags=re.I),
    re.compile(r"\bfree shipping\b", flags=re.I),
    re.compile(r"\blimited edition\b", flags=re.I),
    re.compile(r"\bhot deal\b", flags=re.I),
    re.compile(r"\bsave\s*\d+%?\b", flags=re.I),
    re.compile(r"\bbought\s+\d+\s+times\b", flags=re.I),
]

# Currency/price removal (keeps other numbers like sizes)
_PRICE_RE = re.compile(r"(?:₹|\$|usd|inr|rs\.?|eur|€)\s*\d+[,\d]*(?:\.\d+)?", flags=re.I)
_PRICE_WORD_RE = re.compile(r"\b(?:price|mrp|rs|inr|usd|aud|eur)\b", flags=re.I)

# SKU-like pattern: long alpha-numeric tokens (heuristic)
_SKU_LIKE_RE = re.compile(r"\b(?=[A-Za-z0-9-]{6,})[A-Za-z0-9-]+\b")

# Common single-letter sizes we want to keep (s,m,l,xl etc.)
_SIZE_TOKENS = {"s", "m", "l", "xl", "xxl", "xs", "xxs", "xxxl"}

# Safe stopwords (we intentionally DO NOT include 'in' because it can be a unit)
_SAFE_STOPWORDS = {
    "for", "and", "with", "the", "a", "an", "of", "on", "by", "new", "its", "our"
}

# Precompile abbreviation regex list
_COMPILED_ABBREV = [(re.compile(k, flags=re.I), v) for k, v in ABBREV_MAP.items()]

# Precompile color regexes
_COMPILED_COLOR = [(re.compile(k, flags=re.I), v) for k, v in _COLOR_MAP.items()]


# ---------- Helper utilities ----------
def _unicode_normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _remove_html(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    return s


def _apply_unit_patterns(s: str) -> str:
    # normalize smart quotes first
    s = s.replace("”", '"').replace("“", '"').replace("’", "'").replace("′", "'")
    # apply compiled unit patterns (they expect a number before the unit)
    for pat, repl in _UNIT_PATTERNS:
        s = pat.sub(repl, s)
    return s


def _expand_abbreviations(s: str) -> str:
    for cre, repl in _COMPILED_ABBREV:
        s = cre.sub(repl, s)
    return s


def _apply_pack_patterns(s: str) -> str:
    for pat, repl in _PACK_PATTERNS:
        s = pat.sub(repl, s)
    return s


def _normalize_colors(s: str) -> str:
    for cre, repl in _COMPILED_COLOR:
        s = cre.sub(repl, s)
    return s


def _drop_noise(s: str) -> str:
    for pat in _NOISE_REGEXES:
        s = pat.sub(" ", s)
    # remove explicit price tokens
    s = _PRICE_RE.sub(" ", s)
    s = _PRICE_WORD_RE.sub(" ", s)
    return s


def _expand_slash_or_range_groups(s: str) -> str:
    """
    Expand groups like '20/24/28' or '20-24-28' to '20 inch 24 inch 28 inch'
    if the numbers look like plausible sizes (heuristic).
    """
    def repl(m):
        grp = m.group(0)
        parts = re.split(r"[\/\-]", grp)
        nums = [p for p in parts if p.isdigit()]
        if not nums:
            return grp
        vals = [int(n) for n in nums]
        # heuristics: if numbers are within [1, 500]
        if max(vals) >= 3 and max(vals) <= 1000:
            # If numbers are large enough (>8) we assume inches for apparel/luggage context
            if max(vals) >= 8 and max(vals) <= 200:
                return " ".join(f"{n} inch" for n in nums)
            else:
                # otherwise keep numeric tokens separated
                return " ".join(nums)
        return " ".join(nums)
    # match runs of digits separated by / or -
    return re.sub(r"\b\d+(?:[\/\-]\d+)+\b", repl, s)


def _process_parentheses(s: str) -> str:
    """
    For each (...) chunk:
      - If chunk is SKU-like (alnum long) -> drop
      - If chunk contains slash-group -> expand via _expand_slash_or_range_groups
      - Otherwise keep content (cleaned)
    """
    def repl(m):
        inner = m.group(1).strip()
        if not inner:
            return " "
        # SKU-like: contains letters+digits and is long-ish -> drop
        if any(ch.isalpha() for ch in inner) and any(ch.isdigit() for ch in inner):
            if _SKU_LIKE_RE.match(inner.replace(" ", "")) and len(inner) >= 6:
                return " "
        # expand slash groups in parentheses e.g., (20/24/28)
        if re.search(r"\d+(?:[\/\-]\d+)+", inner):
            return " " + _expand_slash_or_range_groups(inner) + " "
        # otherwise keep inner but cleaned
        return " " + inner + " "
    return re.sub(r"\(([^)]*)\)", repl, s)


# Simple roman numeral mapper for common small cases (word-boundary aware)
_ROMAN_MAP = {
    r"\bii\b": "2",
    r"\biii\b": "3",
    r"\biv\b": "4",
    r"\bv\b": "5",
    r"\bvi\b": "6",
    r"\bvii\b": "7",
    r"\bviii\b": "8",
    r"\bix\b": "9",
    r"\bxi\b": "11",
}


def _normalize_roman(s: str) -> str:
    for pat, repl in _ROMAN_MAP.items():
        s = re.sub(pat, repl, s, flags=re.I)
    return s


def _strip_unwanted_chars(s: str) -> str:
    # remove currency symbols left and other punctuation except % and .
    s = re.sub(r"[^\w\s%\.]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _get_wordnet_pos(treebank_tag: str):
    """Map Treebank POS tag to WordNet POS tag for lemmatization."""
    if not treebank_tag:
        return wn.NOUN
    tag = treebank_tag[0].upper()
    if tag == "J":
        return wn.ADJ
    if tag == "V":
        return wn.VERB
    if tag == "R":
        return wn.ADV
    return wn.NOUN


# ---------- Main public function ----------
def clean_text(text: Optional[str]) -> str:
    """
    Clean and normalize a product title or query.
    Use the same function for both datasets and user queries.
    """
    if text is None:
        return ""

    s = str(text)

    # 1. HTML & unicode normalization + lowercase
    s = _remove_html(s)
    s = _unicode_normalize(s).lower()

    # 2. Unit patterns & quote handling (must happen before stopword removal)
    s = _apply_unit_patterns(s)

    # 3. Expand slashes/ranges (20/24/28 -> 20 inch 24 inch 28 inch)
    s = _expand_slash_or_range_groups(s)

    # 4. Parentheses processing - handles (20/24/28), removes SKUs
    s = _process_parentheses(s)

    # 5. Abbreviations & pack normalization
    s = _expand_abbreviations(s)
    s = _apply_pack_patterns(s)

    # 6. Drop explicit marketing noise, ratings, review counts and prices
    s = _drop_noise(s)

    # 7. Normalize colors
    s = _normalize_colors(s)

    # 8. Normalize common roman numerals to numbers
    s = _normalize_roman(s)

    # 9. Make punctuation whitespace and remove weird chars (retain letters/numbers/percent/dot)
    s = _strip_unwanted_chars(s)

    # 10. Tokenize (NLTK)
    tokens = nltk.word_tokenize(s)

    # 11. Remove safe stopwords (but keep unit tokens like 'inch', 'lb', 'oz', etc.)
    filtered = []
    for t in tokens:
        if t in _SAFE_STOPWORDS:
            continue
        filtered.append(t)

    # 12. Remove SKU-like long alnum tokens (heuristic)
    filtered2 = []
    for t in filtered:
        if _SKU_LIKE_RE.fullmatch(t) and len(t) >= 8 and any(ch.isalpha() for ch in t) and any(ch.isdigit() for ch in t):
            # drop probable SKU/model tokens
            continue
        filtered2.append(t)

    # 13. Lemmatize with POS tags (helps small gains in matching)
    tagged = nltk.pos_tag(filtered2)
    lemmas: List[str] = []
    for word, tag in tagged:
        wn_pos = _get_wordnet_pos(tag)
        lemma = _LEMMATIZER.lemmatize(word, wn_pos)
        lemmas.append(lemma)

    # 14. Final token filtering: drop very long numeric tokens, keep small size tokens like 's','m','l'
    final_tokens: List[str] = []
    for t in lemmas:
        # strip leftover punctuation
        t = t.strip(".,%")
        if not t:
            continue
        # keep single-letter sizes (s,m,l) & small size tokens explicitly
        if len(t) == 1 and t not in _SIZE_TOKENS:
            # If single letter not size, drop it (e.g., stray 'x' or 'p')
            continue
        # drop numeric-only tokens that are extremely long (likely SKU)
        if t.isdigit() and len(t) > 12:
            continue
        final_tokens.append(t)

    # 15. Join and collapse whitespace
    out = " ".join(final_tokens)
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ---------- small convenience wrapper ----------
def make_text_for_embedding_record(prod: dict, category_map: Optional[dict] = None) -> str:
    """
    Compose the final embedding string for a product record.
    By default we use title + category (if available).
    """
    parts: List[str] = []
    title = prod.get("title") or prod.get("Title") or prod.get("name") or ""
    title_clean = clean_text(title)
    if title_clean:
        parts.append(title_clean)
    # category resolution
    cat = prod.get("category_name") or prod.get("category") or None
    if not cat and category_map and prod.get("category_id"):
        cat = category_map.get(str(prod.get("category_id")))
    if cat:
        cat_clean = clean_text(cat)
        if cat_clean:
            parts.append(cat_clean)
    return " | ".join(parts)


# ---------- quick test / debug (only runs when executed directly) ----------
if __name__ == "__main__":
    samples = [
        'Stratum 2.0 Expandable Hardside Luggage with Spinner Wheels | 28" SPINNER | Slate Blue (3-Piece Set)',
        "Centric 2 Hardside Expandable Luggage with Spinners, True Navy, 3-Piece Set (20/24/28)",
        "Women's Lace Casual No Show Non-skid Boat Socks Set of 5",
        "Men's 2TB External HDD - 4.5 stars with 2000 reviews - Bestseller!",
        "3pk cotton socks, 12oz, black",
        "Winfield II Hardside Expandable Luggage (Model AB1234-XY)"
    ]
    for s in samples:
        print("ORIG:", s)
        print("CLEAN:", clean_text(s))
        print("------")

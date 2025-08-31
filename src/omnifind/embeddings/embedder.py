from pathlib import Path
import json
import pickle
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("data/sample_products.json")
EMBED_DIR = Path("data/embeddings")
EMBED_FILE = EMBED_DIR / "text_embeddings.pkl"

def product_semantic_text(p: dict) -> str:
    """
    Build the semantic text for embeddings using unstructured fields.
    Keep structured fields (brand/category/price) for filters, not embeddings.
    """
    title = p.get("title", "")
    desc = p.get("description", "")
    # You can optionally append color/material if helpful semantically:
    color = p.get("color")
    material = p.get("material")
    extras = []
    if color: extras.append(color)
    if material: extras.append(material)
    extras_text = f" ({', '.join(extras)})" if extras else ""
    return f"{title}{extras_text}. {desc}".strip()

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Catalog not found: {DATA_FILE}")
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        products = json.load(f)

    texts = [product_semantic_text(p) for p in products]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    with open(EMBED_FILE, "wb") as f:
        pickle.dump({"products": products, "embeddings": embeddings}, f)

    print(f"Saved {len(products)} embeddings → {EMBED_FILE}")

if __name__ == "__main__":
    main()

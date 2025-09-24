# omnifind-ai/src/omnifind/embeddings/build_embeddings.py
"""
Build embeddings + FAISS index for product retrieval.

Pipeline:
  - Load products (CSV/PKL/JSON/JSONL).
  - Clean title text via preprocess_texts.clean_text().
  - Append category name as a semantic anchor.
  - Encode with SentenceTransformer (default: intfloat/e5-large-v2).
  - Normalize vectors (cosine sim).
  - Save embeddings + FAISS index.

Usage:
  python -m omnifind.embeddings.build_embeddings \
      --products data/processed/fashion_products.pkl \
      --model-name intfloat/e5-large-v2
"""

import argparse
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Import preprocessing
from omnifind.embeddings.preprocess_texts import clean_text

# ---------------- Output paths ----------------
EMBED_DIR = Path("data/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)
EMBED_FILE = EMBED_DIR / "text_embeddings.pkl"
INDEX_FILE = EMBED_DIR / "faiss_index.index"

# ---------------- Loaders ----------------
def load_products(path: str) -> List[Dict[str, Any]]:
    """Load products into list-of-dicts format."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Products file not found: {path}")

    if p.suffix == ".pkl":
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "products" in obj:
            return obj["products"]
        raise ValueError("Unsupported .pkl content")

    if p.suffix == ".csv":
        return pd.read_csv(p).to_dict(orient="records")

    if p.suffix == ".jsonl":
        return [json.loads(line) for line in open(p, "r", encoding="utf8") if line.strip()]

    if p.suffix == ".json":
        data = json.load(open(p, "r", encoding="utf8"))
        if isinstance(data, dict) and "products" in data:
            return data["products"]
        if isinstance(data, list):
            return data
        raise ValueError("Invalid JSON format")

    raise ValueError(f"Unsupported file extension: {p.suffix}")

# ---------------- Backup ----------------
def backup_existing_files():
    ts = time.strftime("%Y%m%d_%H%M%S")
    if EMBED_FILE.exists():
        EMBED_FILE.rename(EMBED_DIR / f"text_embeddings.pkl.bak.{ts}")
    if INDEX_FILE.exists():
        INDEX_FILE.rename(EMBED_DIR / f"faiss_index.index.bak.{ts}")
    print("Backups created (if previous files existed).")

# ---------------- Text Builder ----------------
def make_text_for_embedding(prod: Dict[str, Any]) -> str:
    """Compose embedding text: cleaned title + cleaned category."""
    parts = []

    # Title (main signal)
    title = prod.get("title") or prod.get("Title") or prod.get("name")
    if title:
        title_clean = clean_text(str(title))
        if title_clean:
            parts.append(title_clean)

    # Category
    category = prod.get("category_name") or prod.get("category")
    if category:
        cat_clean = clean_text(str(category))
        if cat_clean:
            parts.append(cat_clean)

    return " | ".join(parts)

# ---------------- Main ----------------
def main(args):
    products = load_products(args.products)
    n = len(products)
    print(f"✅ Loaded {n} products from {args.products}")

    print("🔄 Composing texts for embedding...")
    texts = [make_text_for_embedding(p) for p in products]

    # Check empty texts
    empty_count = sum(1 for t in texts if not t.strip())
    if empty_count:
        print(f"⚠️ Warning: {empty_count} empty embedding texts")

    # Model
    model = SentenceTransformer(args.model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"📐 Using model '{args.model_name}' (dim={dim})")

    # Encode
    batch_size = args.batch_size
    embeddings = np.zeros((n, dim), dtype="float32")

    start_t = time.time()
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i+batch_size]
        emb = model.encode(batch_texts, convert_to_numpy=True,
                           show_progress_bar=False, batch_size=batch_size)
        embeddings[i:i+len(emb)] = emb.astype("float32")
        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - start_t
            print(f"   Encoded {i+len(emb)}/{n} in {elapsed:.1f}s")

    # Normalize
    faiss.normalize_L2(embeddings)
    print("✅ Embeddings normalized")

    # Backup + Save
    backup_existing_files()
    with open(EMBED_FILE, "wb") as f:
        pickle.dump({"products": products, "embeddings": embeddings}, f)
    print(f"💾 Saved embeddings → {EMBED_FILE}")

    # Build FAISS index
    print("⚡ Building FAISS index...")
    if n < 1000:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        nlist = args.nlist if args.nlist else max(64, int(n ** 0.5))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train index
        train_vectors = embeddings[:min(10000, n)].copy()
        index.train(train_vectors)
        ids = np.arange(n).astype("int64")
        index.add_with_ids(embeddings, ids)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"✅ FAISS index saved → {INDEX_FILE}")

    print("🎉 All done!")

# ---------------- Entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=str, required=True,
                        help="Path to products file (.pkl/.csv/.json/.jsonl)")
    parser.add_argument("--model-name", type=str, default="intfloat/e5-large-v2",
                        help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for encoding")
    parser.add_argument("--nlist", type=int, default=None,
                        help="Number of IVF lists (for FAISS)")
    args = parser.parse_args()
    main(args)

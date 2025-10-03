# src/omnifind/embeddings/build_embeddings.py
"""
Build embeddings + FAISS index for product retrieval.
Optimized for **accuracy + speed** (Google/Amazon-grade retriever).

- Saves: products.json, embeddings.npy (memmap-friendly), faiss_index.index
- Features:
    • GPU acceleration for encoding (if available)
    • Multiple FAISS index types (flat, hnsw, ivf)
    • Normalized embeddings → cosine similarity
    • Backup system (no accidental overwrite)
    • Reuse embeddings with --rebuild-index-only

Usage:
    python -m omnifind.embeddings.build_embeddings \
      --products data/processed/fashion_products.csv \
      --model-name intfloat/e5-large-v2 \
      --batch-size 256 \
      --index-type hnsw

    # Rebuild index only (skip 30min embedding step)
    python -m omnifind.embeddings.build_embeddings \
      --rebuild-index-only --index-type hnsw --nlist 256
"""
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from omnifind.embeddings.preprocess_texts import clean_text

# ==== Paths ====
EMBED_DIR = Path("data/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTS_FILE = EMBED_DIR / "products.json"
EMBED_NPY = EMBED_DIR / "embeddings.npy"
INDEX_FILE = EMBED_DIR / "faiss_index.index"


# ==== Helpers ====
def load_products(path: str) -> List[Dict[str, Any]]:
    """Load products from csv/json/jsonl/pkl."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Products file not found: {path}")

    if p.suffix == ".pkl":
        import pickle
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, dict) and "products" in obj:
            return obj["products"]
        if isinstance(obj, list):
            return obj
        raise ValueError(".pkl format not supported here")

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

    raise ValueError(f"Unsupported extension: {p.suffix}")


def backup_if_exists():
    """Make timestamped backups of existing files."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    for f in (PRODUCTS_FILE, EMBED_NPY, INDEX_FILE):
        if f.exists():
            f.rename(f.with_suffix(f.suffix + f".bak.{ts}"))
    print("✅ Backups created (if files existed)")


def make_text(prod: Dict[str, Any]) -> str:
    """Concatenate clean title + category for embedding."""
    parts = []
    title = prod.get("title") or prod.get("Title") or prod.get("name")
    if title:
        tc = clean_text(str(title))
        if tc:
            parts.append(tc)

    cat = prod.get("category_name") or prod.get("category")
    if cat:
        cc = clean_text(str(cat))
        if cc:
            parts.append(cc)

    return " | ".join(parts)


def build_index(embeddings: np.ndarray, index_type: str, dim: int, nlist: int = None, hnsw_m: int = 32):
    """Build FAISS index with accuracy/speed trade-offs."""
    if index_type == "flat":
        # Flat = exact search, highest accuracy (slower)
        idx = faiss.IndexFlatIP(dim)
        idx.add(embeddings)
        return idx

    if index_type == "hnsw":
        # HNSW = high recall + fast queries
        idx = faiss.IndexHNSWFlat(dim, hnsw_m)
        idx.hnsw.efConstruction = max(200, hnsw_m * 2)  # better recall
        idx.hnsw.efSearch = 128
        idx.add(embeddings)
        return idx

    if index_type == "ivf":
        # IVF = good for huge datasets (millions)
        nlist = int(nlist or max(64, int(len(embeddings) ** 0.5)))
        quantizer = faiss.IndexFlatIP(dim)
        ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        train_vectors = embeddings[:min(10000, len(embeddings))].copy()
        ivf.train(train_vectors)
        ids = np.arange(len(embeddings)).astype("int64")
        ivf.add_with_ids(embeddings, ids)
        return ivf

    raise ValueError("index_type must be one of: flat, hnsw, ivf")


# ==== Main ====
def main(args):
    if args.rebuild_index_only:
        # --- Reuse embeddings ---
        if not EMBED_NPY.exists() or not PRODUCTS_FILE.exists():
            raise FileNotFoundError("embeddings.npy or products.json missing for index rebuild")
        products = json.load(open(PRODUCTS_FILE, "r", encoding="utf8"))
        embeddings = np.load(EMBED_NPY, mmap_mode=None)
        print(f"🔁 Rebuilding FAISS index from existing embeddings (n={len(products)})...")
    else:
        # --- Fresh build ---
        products = load_products(args.products)
        n = len(products)
        print(f"📦 Loaded {n} products")

        texts = [make_text(p) for p in products]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Encoding on device: {device}")
        model = SentenceTransformer(args.model_name, device=device)
        dim = model.get_sentence_embedding_dimension()
        print(f"🔢 Embedding dimension: {dim}")

        embeddings = np.zeros((len(texts), dim), dtype="float32")

        batch = args.batch_size
        t0 = time.time()
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i+batch]
            emb = model.encode(batch_texts, convert_to_numpy=True, batch_size=batch, show_progress_bar=False)
            embeddings[i:i+len(emb)] = emb.astype("float32")
            if (i // batch) % 10 == 0:
                print(f"   Encoded {i+len(emb)}/{len(texts)}  [{time.time()-t0:.1f}s elapsed]")

        faiss.normalize_L2(embeddings)
        print("✅ Embeddings normalized (cosine similarity ready)")

        # Save everything safely
        backup_if_exists()
        with open(PRODUCTS_FILE, "w", encoding="utf8") as f:
            json.dump(products, f, ensure_ascii=False)
        np.save(EMBED_NPY, embeddings)
        print("💾 Saved products.json + embeddings.npy")

    # --- Build FAISS index ---
    dim = embeddings.shape[1]
    print(f"⚙️ Building FAISS index type={args.index_type} ...")
    index = build_index(embeddings, args.index_type, dim, nlist=args.nlist, hnsw_m=args.hnsw_m)

    # Always save CPU index (safe & portable)
    if faiss.get_num_gpus() > 0:
        index = faiss.index_gpu_to_cpu(index)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"💾 Saved FAISS index → {INDEX_FILE}")
    print("✅ Build complete. Retriever is ready!")


# ==== CLI ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=str, default="data/processed/fashion_products.csv")
    parser.add_argument("--model-name", type=str, default="intfloat/e5-large-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--index-type", type=str, default="hnsw", choices=["hnsw", "flat", "ivf"])
    parser.add_argument("--nlist", type=int, default=None, help="For IVF: number of clusters")
    parser.add_argument("--hnsw-m", type=int, default=32, help="HNSW: graph connectivity (higher=better recall)")
    parser.add_argument("--rebuild-index-only", action="store_true", help="Skip embeddings, just rebuild FAISS index")
    args = parser.parse_args()
    main(args)

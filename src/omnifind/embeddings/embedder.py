# src/omnifind/embeddings/embedder.py
"""
Production embeddings builder optimized for RTX 4060 (8GB VRAM).

Features:
- E5 task prefixes (passage:/query:)
- Enhanced multi-field text (brand, price, quality, ASIN)
- Automatic VRAM monitoring & batch size adjustment
- BM25 corpus generation
- Memory-efficient encoding
"""
import argparse
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle

from omnifind.embeddings.preprocess_texts import clean_text

# ==== Paths ====
EMBED_DIR = Path("data/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTS_FILE = EMBED_DIR / "products.json"
EMBED_NPY = EMBED_DIR / "embeddings.npy"
INDEX_FILE = EMBED_DIR / "faiss_index.index"
BM25_FILE = EMBED_DIR / "bm25_corpus.pkl"

# ==== Brand/Attribute Extraction ====
BRAND_PATTERNS = [
    r'\b(nike|adidas|puma|reebok|under armour|ua|new balance)\b',
    r'\b(levi|levis|gap|old navy|hm|h&m|zara|uniqlo|blue buddha)\b',
    r'\b(calvin klein|tommy hilfiger|ralph lauren|polo)\b',
    r'\b(columbia|north face|patagonia|carhartt)\b',
    r'\b(gucci|prada|louis vuitton|lv|chanel|dior|versace)\b',
]
BRAND_RE = re.compile('|'.join(BRAND_PATTERNS), re.I)

COLOR_WORDS = {'black', 'white', 'blue', 'red', 'green', 'yellow', 'gray',
               'grey', 'navy', 'brown', 'pink', 'purple', 'orange', 'beige',
               'tan', 'khaki', 'olive', 'burgundy', 'maroon'}

def extract_brand(text: str) -> str:
    match = BRAND_RE.search(text.lower())
    return match.group(0) if match else ""

def extract_colors(text: str) -> List[str]:
    words = text.lower().split()
    return [w for w in words if w in COLOR_WORDS]

def get_vram_usage_mb():
    """Get current VRAM usage in MB (0 if CUDA unavailable)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_vram_free_mb():
    """Get free VRAM in MB."""
    if torch.cuda.is_available():
        return (torch.cuda.get_device_properties(0).total_memory - 
                torch.cuda.memory_allocated()) / 1024 / 1024
    return 0

def auto_batch_size(model_vram_mb: float, target_free_mb: float = 3000) -> int:
    """
    Auto-calculate safe batch size based on available VRAM.
    
    Args:
        model_vram_mb: VRAM used by model
        target_free_mb: Keep this much VRAM free (2GB safety margin)
    
    Returns:
        Safe batch size
    """
    if not torch.cuda.is_available():
        return 128  # CPU fallback
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    
    # RTX 4060 specific: Leave 3GB free minimum
    available = total_vram - model_vram_mb - target_free_mb
    
   # E5-large-v2 uses ~25MB per item in batch (conservative)
    per_item_mb = 25
    safe_batch = max(16, int(available / per_item_mb))
    
    # Cap at 128 for RTX 4060 (not 256!)
    return min(128, safe_batch)
    

# ==== Enhanced Text Generation ====
def make_enhanced_text(prod: Dict[str, Any], include_asin: bool = False) -> str:
    """
    Multi-field text with implicit weighting via repetition.
    Title repeated 3x, brand 2x for importance.
    """
    parts = []
    
    # Title (3x weight)
    title = prod.get("title") or prod.get("Title") or prod.get("name") or ""
    title_clean = clean_text(str(title))
    if title_clean:
        parts.extend([title_clean] * 3)
    
    # ASIN (exact lookups)
    if include_asin:
        asin = prod.get("asin", "")
        if asin:
            parts.append(str(asin))
    
    # Category (2x weight)
    cat = prod.get("category_name") or prod.get("category") or ""
    cat_clean = clean_text(str(cat))
    if cat_clean:
        parts.extend([cat_clean] * 2)
    
    # Brand (2x weight)
    brand = extract_brand(title)
    if brand:
        parts.extend([brand] * 2)
    
    # Price bucket
    price = prod.get("price", 0)
    if price and price > 0:
        if price < 500:
            parts.append("cheap affordable budget under 500")
        elif price < 2000:
            parts.append("mid range under 2000 moderate")
        elif price < 5000:
            parts.append("under 5000 reasonably priced")
        else:
            parts.append("premium expensive luxury")
    
    # Quality tier
    stars = prod.get("stars", 0)
    reviews = prod.get("reviews", 0)
    if stars >= 4.5 and reviews >= 50:
        parts.append("highly rated top rated excellent 5 star")
    elif stars >= 4.0 and reviews >= 20:
        parts.append("well rated good quality 4 star")
    
    # Popularity
    if prod.get("isBestSeller"):
        parts.append("bestseller popular top seller")
    
    bought = prod.get("boughtInLastMonth", 0)
    if bought > 500:
        parts.append("trending viral hot")
    elif bought > 100:
        parts.append("popular")
    
    # Attributes
    colors = extract_colors(title)
    if colors:
        parts.extend(colors)
    
    return " ".join(parts)

def make_bm25_text(prod: Dict[str, Any]) -> str:
    """Simple text for BM25 (no repetition)."""
    parts = []
    
    title = clean_text(prod.get("title", ""))
    if title:
        parts.append(title)
    
    asin = prod.get("asin", "")
    if asin:
        parts.append(str(asin))
    
    cat = clean_text(prod.get("category_name", ""))
    if cat:
        parts.append(cat)
    
    brand = extract_brand(prod.get("title", ""))
    if brand:
        parts.append(brand)
    
    # CRITICAL: Return fallback if empty
    result = " ".join(parts).strip()
    return result if result else "unknown product"  # ← ADD THIS

# ==== Product Loading ====
def load_products(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Products file not found: {path}")
    
    if p.suffix == ".csv":
        return pd.read_csv(p).to_dict(orient="records")
    
    if p.suffix == ".json":
        data = json.load(open(p, "r", encoding="utf8"))
        if isinstance(data, dict) and "products" in data:
            return data["products"]
        if isinstance(data, list):
            return data
        raise ValueError("Invalid JSON format")
    
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in open(p) if line.strip()]
    
    raise ValueError(f"Unsupported extension: {p.suffix}")

def backup_if_exists():
    ts = time.strftime("%Y%m%d_%H%M%S")
    for f in (PRODUCTS_FILE, EMBED_NPY, INDEX_FILE, BM25_FILE):
        if f.exists():
            backup = f.with_suffix(f.suffix + f".{ts}.bak")
            f.rename(backup)
            print(f"📦 Backed up: {f.name} → {backup.name}")

# ==== FAISS Index ====
def build_index(embeddings: np.ndarray, index_type: str, dim: int,nlist: int = None, hnsw_m: int = 32):

    print(f"[DEBUG] Building {index_type} index | dim={dim} | embeddings={embeddings.shape}")
    t0 = time.time()

    try:
        if index_type == "flat":
            idx = faiss.IndexFlatIP(dim)
            print("[DEBUG] Adding embeddings to Flat index...")
            idx.add(embeddings)
            print(f"[DEBUG] Done in {time.time()-t0:.2f}s")
            return idx

        elif index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(dim, hnsw_m)
            idx.hnsw.efConstruction = max(200, hnsw_m * 2)
            idx.hnsw.efSearch = 128
            print("[DEBUG] Adding embeddings to HNSW index (this may take 20+ minutes)...")
            idx.add(embeddings)
            print(f"[DEBUG] HNSW index built in {time.time()-t0:.1f}s")
            return idx

        elif index_type == "ivf":
            nlist = int(nlist or max(64, int(len(embeddings) ** 0.5)))
            quantizer = faiss.IndexFlatIP(dim)
            ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            train_vecs = embeddings[:min(10000, len(embeddings))].copy()
            print(f"[DEBUG] Training IVF on {len(train_vecs)} vectors...")
            ivf.train(train_vecs)
            print("[DEBUG] Adding embeddings to IVF index...")
            ivf.add(embeddings)
            print(f"[DEBUG] IVF index built in {time.time()-t0:.1f}s")
            return ivf

        else:
            raise ValueError("index_type must be: flat, hnsw, ivf")

    except Exception as e:
        print(f"[ERROR] Exception in build_index: {e}")
        raise


# ==== Main ====
def main(args):
    if args.rebuild_index_only:
        products = json.load(open(PRODUCTS_FILE))
        embeddings = np.load(EMBED_NPY)
        print(f"🔁 Rebuilding index from {len(products)} products...")
    else:
        # Load products
        products = load_products(args.products)
        n = len(products)
        print(f"📦 Loaded {n} products")

        # === MODIFIED: Generate texts with model-specific prefixes ===
        if "e5" in args.model_name.lower():
            print("📝 Generating enhanced texts with E5 'passage:' prefix...")
            texts = [f"passage: {make_enhanced_text(p)}" for p in products]
        else:
            print("📝 Generating enhanced texts for BGE model (no prefix)...")
            texts = [make_enhanced_text(p) for p in products]  # NO PREFIX for BGE!

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading model on {device}...")
        model = SentenceTransformer(args.model_name, device=device)
        dim = model.get_sentence_embedding_dimension()

        # === ADDED: BGE-specific optimizations ===
        if device == "cuda":
            # Use FP16 for BGE models (2x speedup)
            if "bge" in args.model_name.lower():
                model = model.half()
                print("🚀 Using FP16 precision for BGE model (2x speedup)")
                
                # Increase batch size for BGE + FP16
                if args.batch_size == 256:  # Default
                    args.batch_size = 96
                    print(f"⚡ Increased batch size to {args.batch_size} for BGE+FP16")
            
            # Enable TF32 for Ampere GPUs (RTX 4060)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            torch.cuda.empty_cache()
            model_vram = get_vram_usage_mb()
            free_vram = get_vram_free_mb()
            print(f"💾 VRAM: Model={model_vram:.0f}MB, Free={free_vram:.0f}MB")
            
            # Auto-adjust batch size
            if args.batch_size == 256:  # Default
                safe_batch = auto_batch_size(model_vram)
                if safe_batch < 256:
                    print(f"⚠️  Auto-reducing batch size: 256 → {safe_batch} (VRAM safety)")
                    args.batch_size = safe_batch

        print(f"⚙️  Final batch size: {args.batch_size}")

        # Encode embeddings
        embeddings = np.zeros((len(texts), dim), dtype="float32")
        batch = args.batch_size
        t0 = time.time()

        # === ADDED: Cooling breaks for long runs ===
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i+batch]
            emb = model.encode(batch_texts, convert_to_numpy=True,
                            batch_size=batch, show_progress_bar=False)
            embeddings[i:i+len(emb)] = emb.astype("float32")
            
            # === ADDED: Cooling break every 20 batches ===
            if (i // batch) % 20 == 0 and i > 0 and device == "cuda":
                torch.cuda.empty_cache()
                time.sleep(1)  # 1-second cooldown
            
            # Progress reporting
            if (i // batch) % 5 == 0 or i + batch >= len(texts):
                elapsed = time.time() - t0
                processed = i + len(emb)
                rate = processed / elapsed
                eta = (len(texts) - processed) / rate / 60 if rate > 0 else 0
                print(f"   {processed:6d}/{len(texts)} [{elapsed:5.1f}s, "
                    f"{rate:4.0f} prod/s, ETA: {eta:.1f}min]")

        total_time = time.time() - t0
        print(f"✅ Encoded {len(texts)} products in {total_time/60:.1f} minutes")
        
        # Normalize
        faiss.normalize_L2(embeddings)
        print("✅ Embeddings normalized (cosine similarity)")
        
        # === Build BM25 corpus ===
        # === Build BM25 corpus ===
        print("🔨 Building BM25 corpus...")
        bm25_corpus = []
        empty_count = 0
        empty_examples = []

        for idx, p in enumerate(products):
            text = make_bm25_text(p)
            tokens = text.split()
            
            # Track empty products
            if not tokens or text.strip() == "unknown product" or text.strip() == "product":
                empty_count += 1
                if len(empty_examples) < 10:
                    empty_examples.append({
                        "index": idx,
                        "title": p.get("title", "NO TITLE")[:100],
                        "asin": p.get("asin", "NO ASIN"),
                        "bm25_text": text,
                        "tokens": tokens
                    })
                tokens = ["product"]  # Fallback
            
            bm25_corpus.append(tokens)

        # DETAILED DIAGNOSTICS
        print(f"✅ BM25 corpus built: {len(bm25_corpus)} products")
        if empty_count > 0:
            print(f"⚠️  WARNING: {empty_count}/{len(products)} products ({empty_count/len(products)*100:.1f}%) had empty/fallback text!")
            print(f"\n📋 First {len(empty_examples)} examples of problematic products:")
            for ex in empty_examples:
                print(f"\n   Index: {ex['index']}")
                print(f"   Title: {ex['title']}")
                print(f"   ASIN: {ex['asin']}")
                print(f"   BM25 text: '{ex['bm25_text']}'")
                print(f"   Tokens: {ex['tokens']}")

        bm25_model = BM25Okapi(bm25_corpus)
        
        # Save everything
        backup_if_exists()
        
        with open(PRODUCTS_FILE, "w", encoding="utf8") as f:
            json.dump(products, f, ensure_ascii=False)
        
        np.save(EMBED_NPY, embeddings)
        
        with open(BM25_FILE, "wb") as f:
            pickle.dump({"corpus": bm25_corpus, "model": bm25_model}, f)
        
        print(f"💾 Saved: {PRODUCTS_FILE.name}, {EMBED_NPY.name}, {BM25_FILE.name}")
    
    #
    # Build FAISS index
    dim = embeddings.shape[1]
    print(f"⚙️  Building FAISS {args.index_type} index...")
    print(f"   Index params: type={args.index_type}, m={args.hnsw_m}, dim={dim}")
    print(f"   Dataset size: {len(embeddings):,} products")
    print(f"   Estimated memory: ~{len(embeddings) * dim * 4 / 1024 / 1024:.0f} MB")
    print(f"   This may take 5-30 minutes for large datasets...")

    try:
        t_index = time.time()
        index = build_index(embeddings, args.index_type, dim,
                        args.nlist, args.hnsw_m)
        print(f"✅ Index built in {(time.time()-t_index)/60:.1f} minutes")
    except Exception as e:
        print(f"❌ FAISS index build failed: {e}")
        print(f"   Try: --index-type flat (faster, exact search)")
        print(f"   Or: Reduce dataset size")
        raise

    # Save as CPU index
    print("💾 Saving index to disk...")
    try:
        if faiss.get_num_gpus() > 0:
            index = faiss.index_gpu_to_cpu(index)
        
        faiss.write_index(index, str(INDEX_FILE))
        print(f"✅ Saved: {INDEX_FILE}")
    except Exception as e:
        print(f"❌ Failed to save index: {e}")
        raise

    print("\n" + "="*60)
    print("✅ BUILD COMPLETE!")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   Products: {len(embeddings):,}")
    print(f"   Embeddings: {EMBED_NPY}")
    print(f"   FAISS index: {INDEX_FILE}")
    print(f"   BM25 corpus: {BM25_FILE}")
    print(f"\n🚀 Next step:")
    print(f"   uvicorn src.omnifind.api.main:app --reload")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build embeddings with E5 prefixes + BM25"
    )
    parser.add_argument("--products", default="data/processed/fashion_products.csv")
    parser.add_argument("--model-name", default="BAAI/bge-large-en-v1.5") #intfloat/e5-large-v
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Auto-adjusts for Gpu eg:(RTX 4060) if needed")
    parser.add_argument("--index-type", default="hnsw", choices=["hnsw", "flat", "ivf"])
    parser.add_argument("--nlist", type=int, default=None, help="Number of clusters for IVF")
    parser.add_argument("--hnsw-m", type=int, default=32,
                       help="Use 32 for large datasets, 16 for small (<10k), 64 for high recall")
    parser.add_argument("--rebuild-index-only", action="store_true", help="Rebuild FAISS index only, skip embeddings")
    args = parser.parse_args()
    main(args)
# src/omnifind/embeddings/image_embedder.py
"""
Image embeddings builder using CLIP for visual product search.

Features:
- CLIP model for joint image-text embeddings
- Batch processing with progress tracking
- Automatic image download and caching
- Handles missing/broken images gracefully
- GPU acceleration with FP16
- Saves: image_embeddings.npy, image_products.json, image_faiss_index.index
"""
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import pandas as pd
import torch
from PIL import Image
import requests
from io import BytesIO
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==== Paths ====
EMBED_DIR = Path("data/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_PRODUCTS_FILE = EMBED_DIR / "image_products.json"
IMAGE_EMBED_NPY = EMBED_DIR / "image_embeddings.npy"
IMAGE_INDEX_FILE = EMBED_DIR / "image_faiss_index.index"
IMAGE_CACHE_DIR = Path("data/image_cache")
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ImageEmbedder:
    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        device: str = "cuda",
        use_fp16: bool = True,
        batch_size: int = 32,
    ):
        """
        Args:
            model_name: CLIP model variant
            device: 'cuda' or 'cpu'
            use_fp16: Use half precision for 2x speedup
            batch_size: Images per batch
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading CLIP model: {model_name} on {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # FP16 optimization
        if use_fp16 and self.device == "cuda":
            self.model = self.model.half()
            print("⚡ Using FP16 precision (2x speedup)")
        
        self.batch_size = batch_size
        print(f"✅ CLIP model loaded | Embedding dim: {self.model.get_sentence_embedding_dimension()}")
    
    def download_image(self, url: str, product_id: str, timeout: int = 5) -> Optional[Image.Image]:
        """
        Download image from URL with caching.
        Returns PIL Image or None if failed.
        """
        # Check cache first
        cache_path = IMAGE_CACHE_DIR / f"{product_id}.jpg"
        if cache_path.exists():
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception as e:
                print(f"⚠️  Cached image corrupted: {product_id}")
                cache_path.unlink()  # Delete corrupted cache
        
        # Download
        try:
            response = requests.get(url, timeout=timeout, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Cache it
            img.save(cache_path, "JPEG", quality=85)
            return img
            
        except Exception as e:
            print(f"❌ Failed to download {url[:50]}... | {e}")
            return None
    
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode list of PIL images to embeddings.
        Returns: (N, embedding_dim) array
        """
        embeddings = self.model.encode(
            images,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embeddings.astype("float32")
    
    def build_embeddings(
        self,
        products: List[Dict[str, Any]],
        image_url_key: str = "imgUrl",
        max_products: Optional[int] = None,
    ) -> tuple:
        """
        Build image embeddings for all products.
        
        Returns:
            (embeddings, valid_products, failed_count)
        """
        if max_products:
            products = products[:max_products]
        
        print(f"\n📸 Processing {len(products):,} product images...")
        
        valid_products = []
        valid_images = []
        failed_count = 0
        
        # Download and filter valid images
        for idx, prod in enumerate(tqdm(products, desc="Downloading images")):
            url = prod.get(image_url_key)
            product_id = prod.get("asin") or prod.get("id") or str(idx)
            
            if not url:
                failed_count += 1
                continue
            
            img = self.download_image(url, product_id)
            if img is None:
                failed_count += 1
                continue
            
            valid_images.append(img)
            valid_products.append(prod)
            
            # Progress update every 100
            if (idx + 1) % 100 == 0:
                success_rate = len(valid_images) / (idx + 1) * 100
                print(f"   {idx+1}/{len(products)} | Success: {success_rate:.1f}% | Failed: {failed_count}")
        
        print(f"\n✅ Downloaded {len(valid_images):,} images | Failed: {failed_count}")
        
        if len(valid_images) == 0:
            raise ValueError("No valid images to encode!")
        
        # Encode images in batches
        print(f"\n🔄 Encoding {len(valid_images):,} images...")
        embeddings = []
        
        for i in tqdm(range(0, len(valid_images), self.batch_size), desc="Encoding batches"):
            batch_imgs = valid_images[i:i+self.batch_size]
            batch_emb = self.encode_images(batch_imgs)
            embeddings.append(batch_emb)
        
        embeddings = np.vstack(embeddings)
        print(f"✅ Encoded {len(embeddings):,} images | Shape: {embeddings.shape}")
        
        return embeddings, valid_products, failed_count


def build_image_index(
    embeddings: np.ndarray,
    index_type: str = "flat",
    nlist: int = 100,
) -> faiss.Index:
    """
    Build FAISS index for image embeddings.
    
    Args:
        embeddings: (N, D) array
        index_type: 'flat' (exact) or 'ivf' (approximate)
        nlist: IVF clusters (for large datasets)
    """
    dim = embeddings.shape[1]
    
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine after L2 norm)
        index.add(embeddings)
        print(f"✅ Built Flat index: {index.ntotal} vectors")
    
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train on subset
        train_size = min(10000, len(embeddings))
        index.train(embeddings[:train_size])
        index.add(embeddings)
        print(f"✅ Built IVF index: {index.ntotal} vectors | nlist={nlist}")
    
    else:
        raise ValueError("index_type must be 'flat' or 'ivf'")
    
    return index


def main(args):
    # Load products
    print(f"📦 Loading products from {args.products}")
    df = pd.read_csv(args.products)
    products = df.to_dict(orient="records")
    print(f"✅ Loaded {len(products):,} products")
    
    # Initialize embedder
    embedder = ImageEmbedder(
        model_name=args.model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fp16=args.use_fp16,
        batch_size=args.batch_size,
    )
    
    # Build embeddings
    t0 = time.time()
    embeddings, valid_products, failed_count = embedder.build_embeddings(
        products,
        image_url_key=args.image_url_key,
        max_products=args.max_products,
    )
    elapsed = time.time() - t0
    
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"📊 Success rate: {len(valid_products)/len(products)*100:.1f}%")
    
    # Build FAISS index
    print(f"\n🔨 Building FAISS index (type={args.index_type})...")
    index = build_image_index(embeddings, args.index_type, args.nlist)
    
    # Save everything
    print("\n💾 Saving files...")
    
    with open(IMAGE_PRODUCTS_FILE, "w", encoding="utf8") as f:
        json.dump(valid_products, f, ensure_ascii=False)
    
    np.save(IMAGE_EMBED_NPY, embeddings)
    
    faiss.write_index(index, str(IMAGE_INDEX_FILE))
    
    print(f"✅ Saved:")
    print(f"   - {IMAGE_PRODUCTS_FILE}")
    print(f"   - {IMAGE_EMBED_NPY}")
    print(f"   - {IMAGE_INDEX_FILE}")
    print(f"\n🎉 Image search index ready! Failed images: {failed_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build image embeddings with CLIP")
    parser.add_argument("--products", default="data/processed/fashion_products.csv")
    parser.add_argument("--model-name", default="clip-ViT-B-32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-url-key", default="imgUrl", help="Column name for image URLs")
    parser.add_argument("--index-type", default="flat", choices=["flat", "ivf"])
    parser.add_argument("--nlist", type=int, default=100, help="IVF clusters")
    parser.add_argument("--use-fp16", action="store_true", default=True)
    parser.add_argument("--max-products", type=int, default=None, help="Limit for testing")
    
    args = parser.parse_args()
    main(args)
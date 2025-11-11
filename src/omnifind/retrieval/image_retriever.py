# src/omnifind/retrieval/image_retriever.py
"""
Image retriever using CLIP for visual product search.

Features:
- Image-to-image search (upload photo, find similar products)
- Text-to-image search ("red dress") using CLIP's text encoder
- Hybrid image+text search (combine visual and semantic)
- GPU acceleration
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import json
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer  # ← Keep this, it works with CLIP

BASE_DIR = Path(__file__).parent.parent.parent
IMAGE_PRODUCTS_FILE = BASE_DIR / "data/embeddings/image_products.json"
IMAGE_EMBED_NPY = BASE_DIR / "data/embeddings/image_embeddings.npy"
IMAGE_INDEX_FILE = BASE_DIR / "data/embeddings/image_faiss_index.index"


class ImageRetriever:
    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        use_gpu: bool = True,
        use_fp16: bool = True,
    ):
        """
        Image search retriever using CLIP.
        
        Args:
            model_name: CLIP model variant
            use_gpu: Use GPU if available
            use_fp16: Use half precision
        """
        print(f"📸 Initializing ImageRetriever with {model_name}...")
        
        # Load products
        if not IMAGE_PRODUCTS_FILE.exists():
            raise FileNotFoundError(
                f"Image products file not found: {IMAGE_PRODUCTS_FILE}\n"
                "Run: python -m omnifind.embeddings.image_embedder first"
            )
        
        with open(IMAGE_PRODUCTS_FILE, "r", encoding="utf8") as f:
            self.products: List[Dict[str, Any]] = json.load(f)
        
        if len(self.products) == 0:
            raise ValueError("Empty image_products.json")
        
        # Load embeddings
        if not IMAGE_EMBED_NPY.exists():
            raise FileNotFoundError(f"Image embeddings not found: {IMAGE_EMBED_NPY}")
        
        self.embeddings = np.load(IMAGE_EMBED_NPY, mmap_mode="r")
        
        if self.embeddings.shape[0] != len(self.products):
            raise ValueError(
                f"Mismatch: {self.embeddings.shape[0]} embeddings vs {len(self.products)} products"
            )
        
        # Load FAISS index
        if not IMAGE_INDEX_FILE.exists():
            raise FileNotFoundError(f"Image FAISS index not found: {IMAGE_INDEX_FILE}")
        
        self.index = faiss.read_index(str(IMAGE_INDEX_FILE))
        
        # Optional GPU for FAISS
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        if self.use_gpu:
            print("🚀 Moving FAISS index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load CLIP model
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        print(f"🔄 Loading CLIP model on {device}...")
        
        self.model = SentenceTransformer(model_name, device=device)
        
        # FP16 optimization
        if use_fp16 and device == "cuda":
            self.model = self.model.half()
            print("⚡ Using FP16 precision")
        
        print(f"✅ ImageRetriever ready: {len(self.products):,} products")
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode single PIL image to embedding."""
        emb = self.model.encode(
            [image],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype("float32")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query to embedding (CLIP text encoder)."""
        emb = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype("float32")
    
    def search_by_image(
        self,
        image: Image.Image,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar products using uploaded image.
        
        Args:
            image: PIL Image
            top_k: Number of results
            filters: Optional price/category filters
        
        Returns:
            List of products with similarity scores
        """
        # Encode query image
        q_vec = self.encode_image(image)
        faiss.normalize_L2(q_vec)
        
        # Search FAISS
        candidate_pool = max(top_k * 10, 50)
        D, I = self.index.search(q_vec, candidate_pool)
        
        # Build results
        results = []
        seen_asins = set()  # ← ADD: Deduplication
        
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            
            prod = dict(self.products[idx])
            
            # Deduplicate by ASIN
            asin = prod.get("asin") or prod.get("id")
            if asin in seen_asins:
                continue
            seen_asins.add(asin)
            
            prod["_similarity"] = float(score)
            
            # Apply filters if provided
            if filters and not self._passes_filters(prod, filters):
                continue
            
            results.append(prod)
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_text(
        self,
        text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for products using text description.
        Uses CLIP's text encoder to find visually matching products.
        
        Example: "red evening dress with sequins"
        """
        # Encode text
        q_vec = self.encode_text(text)
        faiss.normalize_L2(q_vec)
        
        # Search
        candidate_pool = max(top_k * 10, 50)
        D, I = self.index.search(q_vec, candidate_pool)
        
        results = []
        seen_asins = set()  # ← ADD: Deduplication
        
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            
            prod = dict(self.products[idx])
            
            # Deduplicate
            asin = prod.get("asin") or prod.get("id")
            if asin in seen_asins:
                continue
            seen_asins.add(asin)
            
            prod["_similarity"] = float(score)
            
            if filters and not self._passes_filters(prod, filters):
                continue
            
            results.append(prod)
            if len(results) >= top_k:
                break
        
        return results
    
    def hybrid_search(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        top_k: int = 5,
        alpha: float = 0.5,  # 0.5 = equal weight
        filters: Optional[Dict[str, Any]] = None,  # ← ADD filters
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining image and text.
        
        Args:
            image: Optional PIL Image
            text: Optional text query
            alpha: Weight for image (1-alpha for text)
            filters: Optional filters
        
        Example:
            hybrid_search(image=img, text="nike shoes", alpha=0.7)
            # 70% image similarity, 30% text match
        """
        if image is None and text is None:
            raise ValueError("Provide at least one: image or text")
        
        # Encode queries
        img_vec = self.encode_image(image) if image else None
        txt_vec = self.encode_text(text) if text else None
        
        # Combine embeddings
        if img_vec is not None and txt_vec is not None:
            q_vec = alpha * img_vec + (1 - alpha) * txt_vec
        elif img_vec is not None:
            q_vec = img_vec
        else:
            q_vec = txt_vec
        
        faiss.normalize_L2(q_vec)
        
        # Search
        candidate_pool = max(top_k * 10, 50)
        D, I = self.index.search(q_vec, candidate_pool)
        
        results = []
        seen_asins = set()
        
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            
            prod = dict(self.products[idx])
            
            # Deduplicate
            asin = prod.get("asin") or prod.get("id")
            if asin in seen_asins:
                continue
            seen_asins.add(asin)
            
            prod["_similarity"] = float(score)
            
            # Apply filters
            if filters and not self._passes_filters(prod, filters):
                continue
            
            results.append(prod)
            if len(results) >= top_k:
                break
        
        return results
    
    def _passes_filters(self, prod: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply price/category filters."""
        # Price filters
        price_min = filters.get("price_min")
        price_max = filters.get("price_max")
        price = prod.get("price", 0)
        
        if price_min and price < price_min:
            return False
        if price_max and price > price_max:
            return False
        
        # Category filter
        category = filters.get("category_name")
        if category:
            cat_lower = str(category).lower()
            prod_cat = prod.get("category_name", "").lower()
            if cat_lower not in prod_cat:
                return False
        
        # Stars filter
        stars_min = filters.get("stars_min")
        stars_max = filters.get("stars_max")
        stars = prod.get("stars", 0)
        
        if stars_min and stars < stars_min:
            return False
        if stars_max and stars > stars_max:
            return False
        
        return True


# === CLI for testing ===
if __name__ == "__main__":
    import sys
    
    try:
        retriever = ImageRetriever()
        
        # Test text-to-image search
        print("\n🔍 Testing text-to-image search...")
        results = retriever.search_by_text("red dress", top_k=3)
        
        if results:
            print(f"\n✅ Found {len(results)} results:")
            for i, r in enumerate(results, 1):
                title = r.get("title", "Unknown")[:60]
                sim = r.get("_similarity", 0)
                print(f"{i}. [{sim:.3f}] {title}")
        else:
            print("❌ No results found")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 You need to build the image index first:")
        print("   python -m omnifind.embeddings.image_embedder \\")
        print("     --products ../data/processed/fashion_products.csv \\")
        print("     --max-products 60000")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
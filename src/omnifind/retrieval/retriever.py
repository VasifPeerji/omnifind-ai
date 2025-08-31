import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import faiss

EMBED_FILE = Path("data/embeddings/text_embeddings.pkl")
INDEX_FILE = Path("data/embeddings/faiss_index.index")

class RetrieverService:
    def __init__(self):
        # Load products + embeddings
        if not EMBED_FILE.exists():
            raise FileNotFoundError(f"Embedding file not found: {EMBED_FILE}")
        with open(EMBED_FILE, "rb") as f:
            data = pickle.load(f)
        self.products: List[Dict[str, Any]] = data.get("products", [])
        self.embeddings = np.array(data.get("embeddings", []), dtype="float32")
        if self.embeddings.size == 0 or len(self.products) == 0:
            raise ValueError("Empty products or embeddings.")

        # Load FAISS index
        if not INDEX_FILE.exists():
            raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")
        self.index = faiss.read_index(str(INDEX_FILE))

        # Query encoder
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------- Public API ----------

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search + metadata filtering.
        filters can include:
          - brand: str or List[str]
          - category: str or List[str]
          - price_min: float/int
          - price_max: float/int
        """
        filters = filters or {}
        candidate_pool = max(top_k * 20, 50)  # pull more to allow filtering/dedup

        # Encode and search
        q_vec = self.model.encode([query]).astype("float32")
        D, I = self.index.search(q_vec, candidate_pool)

        # Map indices → products (keep order)
        candidates = [self.products[i] for i in I[0]]

        # Apply filters
        candidates = self._apply_filters(candidates, filters)

        # Deduplicate by id (preserve order)
        unique = []
        seen = set()
        for p in candidates:
            pid = p.get("id")
            if pid not in seen:
                unique.append(p)
                seen.add(pid)
            if len(unique) >= top_k:
                break

        return unique

    # ---------- Helpers ----------

    def _apply_filters(self, products: List[Dict[str, Any]], filters: Dict[str, Any]):
        if not filters:
            return products

        brand = filters.get("brand")
        category = filters.get("category")
        price_min = filters.get("price_min")
        price_max = filters.get("price_max")

        def norm_list(x):
            if x is None:
                return None
            if isinstance(x, str):
                return [x.lower()]
            return [str(v).lower() for v in x]

        brand_list = norm_list(brand)
        category_list = norm_list(category)

        out = []
        for p in products:
            if brand_list:
                pb = str(p.get("brand", "")).lower()
                if pb not in brand_list:
                    continue
            if category_list:
                pc = str(p.get("category", "")).lower()
                if pc not in category_list:
                    continue
            price = p.get("price")
            if price_min is not None and price is not None and price < price_min:
                continue
            if price_max is not None and price is not None and price > price_max:
                continue
            out.append(p)
        return out

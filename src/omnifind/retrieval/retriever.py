# retriever.py
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import faiss
from omnifind.utils.spell_corrector import SpellCorrector
from omnifind.embeddings.preprocess_texts import clean_text
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent.parent.parent  
EMBED_FILE = BASE_DIR / "data/embeddings/text_embeddings.pkl"
INDEX_FILE = BASE_DIR / "data/embeddings/faiss_index.index"




class RetrieverService:
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
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
        self.model = SentenceTransformer(model_name)

        # Build vocabulary for spell correction (title + category_name)
        vocab = set()
        for p in self.products:
            for field in ("title", "category_name"):
                v = p.get(field)
                if v:
                    for tok in clean_text(str(v)).split():
                        vocab.add(tok.strip())

        self.corrector = SpellCorrector(vocabulary=list(vocab))

    # ---------- Public API ----------
    def search_text(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Semantic search + metadata filtering.
        Returns (results, corrected_query, corrected_filters).
        """
        filters = filters or {}
        candidate_pool = max(top_k * 20, 50)  # pull more to allow filtering/dedup

        # ---------- Spell Correction (query) ----------
        corrected_query = self.corrector.correct_query(query)
        if corrected_query != (query or ""):
            print(f"[SpellCorrector] '{query}' → '{corrected_query}'")
        query_to_encode = corrected_query

        # Encode and search
        q_vec = self.model.encode([query_to_encode], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)
        D, I = self.index.search(q_vec, candidate_pool)

        # Map indices → products (keep order)
        candidates = [self.products[i] for i in I[0] if i != -1]

        # Apply filters
        candidates, corrected_filters = self._apply_filters(candidates, filters)

        # Deduplicate by asin
        unique = []
        seen = set()
        for p in candidates:
            pid = p.get("asin")
            if pid not in seen:
                unique.append(p)
                seen.add(pid)
            if len(unique) >= top_k:
                break

        return unique, corrected_query, corrected_filters

    # ---------- Helpers ----------
    def _apply_filters(self, products: List[Dict[str, Any]], filters: Dict[str, Any]):
        if not filters:
            return products, {}

        stars_min = filters.get("stars_min")
        stars_max = filters.get("stars_max")
        price_min = filters.get("price_min")
        price_max = filters.get("price_max")
        bestseller = filters.get("isBestSeller")
        category = filters.get("category_name")

        # normalize & spell-correct category if provided
        category_corr = None
        if category:
            if isinstance(category, str):
                category_list = [category]
            else:
                category_list = list(category)
            category_corr = [self.corrector.correct_word(c.lower()) for c in category_list]
            if category_corr != category_list:
                print(f"[SpellCorrector] Category corrections: {category_list} → {category_corr}")

        out = []
        for p in products:
            # Stars filter
            stars = p.get("stars")
            if stars_min is not None and stars is not None and stars < stars_min:
                continue
            if stars_max is not None and stars is not None and stars > stars_max:
                continue

            # Price filter
            price = p.get("price")
            if price_min is not None and price is not None and price < price_min:
                continue
            if price_max is not None and price is not None and price > price_max:
                continue

            # Bestseller filter
            is_best = p.get("isBestSeller")
            if bestseller is not None and is_best != bestseller:
                continue

            # Category filter
            p_cat = p.get("category_name", "").lower()
            if category_corr and p_cat not in category_corr:
                continue

            out.append(p)

        corrected_filters = {}
        if stars_min is not None:
            corrected_filters["stars_min"] = stars_min
        if stars_max is not None:
            corrected_filters["stars_max"] = stars_max
        if price_min is not None:
            corrected_filters["price_min"] = price_min
        if price_max is not None:
            corrected_filters["price_max"] = price_max
        if bestseller is not None:
            corrected_filters["isBestSeller"] = bestseller
        if category_corr:
            corrected_filters["category_name"] = category_corr

        return out, corrected_filters
    


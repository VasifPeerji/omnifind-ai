# src/omnifind/retrieval/retriever.py
"""
RetrieverService — upgraded version
- Loads FAISS index + embeddings + products
- Semantic search with filters
- Spell correction + deduplication
- LangChain wrapper
- Supports HNSW, Flat, IVF
- GPU-safe + memmap-friendly
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from omnifind.utils.spell_corrector import SpellCorrector
from omnifind.embeddings.preprocess_texts import clean_text

BASE_DIR = Path(__file__).parent.parent.parent
PRODUCTS_FILE = BASE_DIR / "data/embeddings/products.json"
EMBED_NPY = BASE_DIR / "data/embeddings/embeddings.npy"
INDEX_FILE = BASE_DIR / "data/embeddings/faiss_index.index"


class RetrieverService:
    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        use_gpu: bool = False,
        use_memmap: bool = True,
        ef_search: int = 128,
    ):
        """
        Args:
            model_name: sentence-transformers model name
            use_gpu: if True, move encoder + FAISS search to GPU (if available)
            use_memmap: load embeddings via np.load(..., mmap_mode='r') to save RAM
            ef_search: HNSW efSearch parameter (higher = better recall)
        """
        # ---------- Load products ----------
        if not PRODUCTS_FILE.exists():
            raise FileNotFoundError(f"Products metadata not found: {PRODUCTS_FILE}")
        with open(PRODUCTS_FILE, "r", encoding="utf8") as f:
            self.products: List[Dict[str, Any]] = json.load(f)

        if len(self.products) == 0:
            raise ValueError("Empty products.json")

        # ---------- Load FAISS index ----------
        if not INDEX_FILE.exists():
            raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")
        try:
            self.index = faiss.read_index(str(INDEX_FILE))
        except Exception as e:
            raise RuntimeError(
                f"Failed to read FAISS index {INDEX_FILE}: {e}\n"
                "If index was created on GPU, rebuild a CPU index."
            ) from e

        # ---------- HNSW efSearch ----------
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search
            print(f"[Retriever] HNSW efSearch set to {ef_search}")

        # ---------- Optional GPU ----------
        self.use_gpu = bool(use_gpu) and faiss.get_num_gpus() > 0
        if self.use_gpu:
            gpus = faiss.get_num_gpus()
            print(f"ℹ️ {gpus} FAISS GPU(s) detected — moving index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            print("ℹ️ FAISS index loaded on CPU")

        # ---------- Load embeddings ----------
        if not EMBED_NPY.exists():
            raise FileNotFoundError(f"Embeddings file not found: {EMBED_NPY}")

        self.use_memmap = use_memmap
        self.embeddings = (
            np.load(EMBED_NPY, mmap_mode="r") if use_memmap else np.load(EMBED_NPY)
        )

        if self.embeddings.shape[0] != len(self.products):
            raise ValueError(
                f"Mismatch: {self.embeddings.shape[0]} embeddings vs {len(self.products)} products"
            )

        # ---------- Encoder ----------
        device = "cuda" if (use_gpu and self._cuda_available()) else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        print(f"ℹ️ Encoder model '{model_name}' loaded on {device}")

        # ---------- SpellCorrector ----------
        vocab = set()
        for p in self.products:
            for field in ("title", "category_name"):
                v = p.get(field)
                if v:
                    for tok in clean_text(str(v)).split():
                        vocab.add(tok.strip())
        self.corrector = SpellCorrector(vocabulary=list(vocab))

        print(f"✅ Retriever initialized — {len(self.products):,} products. GPU search: {self.use_gpu}")

    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ---------- Query encoding ----------
    def encode_query(self, query: str) -> np.ndarray:
        emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    # ---------- Public search ----------
    def search_text(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        filters = filters or {}
        candidate_pool = max(top_k * 20, 50)

        # Spell-correct query
        corrected_query = self.corrector.correct_query(query)
        if corrected_query != (query or ""):
            print(f"[SpellCorrector] '{query}' → '{corrected_query}'")

        # Encode & normalize
        q_vec = self.encode_query(corrected_query)
        faiss.normalize_L2(q_vec)

        # FAISS search
        D, I = self.index.search(q_vec, candidate_pool)
        candidates = [self.products[i] for i in I[0] if i != -1]

        # Filter
        candidates, corrected_filters = self._apply_filters(candidates, filters)

        # Deduplicate by asin/id/url/title
        unique = []
        seen = set()
        for p in candidates:
            pid = p.get("asin") or p.get("id") or p.get("url") or json.dumps(p.get("title",""))[:64]
            if pid not in seen:
                unique.append(p)
                seen.add(pid)
            if len(unique) >= top_k:
                break

        return unique, corrected_query, corrected_filters

    # ---------- Filtering ----------
    def _passes_filters(self, prod: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, val in filters.items():
            if val is None:
                continue
            if key == "category_name":
                cat = str(prod.get("category_name", "")).lower()
                if isinstance(val, str):
                    if val.lower() not in cat:
                        return False
                elif isinstance(val, list):
                    if not any(v.lower() in cat for v in val):
                        return False
            elif key == "price_min":
                if float(prod.get("price", 0)) < float(val):
                    return False
            elif key == "price_max":
                if float(prod.get("price", 0)) > float(val):
                    return False
            elif key == "stars_min":
                if float(prod.get("stars", 0)) < float(val):
                    return False
            elif key == "stars_max":
                if float(prod.get("stars", 0)) > float(val):
                    return False
            elif key == "isBestSeller":
                if bool(prod.get("isBestSeller", False)) != bool(val):
                    return False
        return True

    def _apply_filters(self, products: List[Dict[str, Any]], filters: Dict[str, Any]):
        if not filters:
            return products, {}

        # Apply filters
        out = [p for p in products if self._passes_filters(p, filters)]

        # Spell-correct category
        category = filters.get("category_name")
        category_corr = None
        if category:
            category_list = [category] if isinstance(category, str) else list(category)
            category_corr = [self.corrector.correct_word(c.lower()) for c in category_list]

        # Corrected filters response
        corrected_filters = {**filters}
        if category_corr:
            corrected_filters["category_name"] = category_corr

        return out, corrected_filters

    # ---------- LangChain wrapper ----------
    def as_langchain_retriever(self, top_k: int = 5):
        try:
            from langchain_core.vectorstores import VectorStore
        except Exception:
            raise RuntimeError("LangChain not installed or importable.")

        class CustomVectorStore(VectorStore):
            def __init__(self, service: "RetrieverService"):
                self.service = service

            def similarity_search(self, query: str, k: int = top_k, **kwargs):
                results, corrected_q, _ = self.service.search_text(query, top_k=k)
                return [{"page_content": r.get("title",""), "metadata": r} for r in results]

        return CustomVectorStore(self).as_retriever(search_kwargs={"k": top_k})

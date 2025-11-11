# src/omnifind/retrieval/hybrid_retriever_v2.py
"""
Production hybrid retriever: FAISS + BM25 + Cross-Encoder
Features:
- E5 query prefix
- Hybrid fusion (semantic + keyword)
- Price extraction from queries
- Pre-filtering before rerank
- Cross-encoder re-ranking (optional)
- Monitoring/logging
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import faiss
import json
import pickle
import re
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from omnifind.utils.spell_corrector import SpellCorrector
from omnifind.embeddings.preprocess_texts import clean_text

BASE_DIR = Path(__file__).parent.parent.parent
PRODUCTS_FILE = BASE_DIR / "data/embeddings/products.json"
EMBED_NPY = BASE_DIR / "data/embeddings/embeddings.npy"
INDEX_FILE = BASE_DIR / "data/embeddings/faiss_index.index"
BM25_FILE = BASE_DIR / "data/embeddings/bm25_corpus.pkl"

# Price extraction pattern
PRICE_PATTERN = re.compile(r'under\s*\$?(\d+)|less\s*than\s*\$?(\d+)|below\s*\$?(\d+)', re.I)

class HybridRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",#intfloat/e5-large-v2
        use_gpu: bool = True,
        alpha: float = 0.6,  # 60% semantic, 40% keyword
        use_reranker: bool = False,
        ef_search: int = 256,
    ):
        """
        Args:
            alpha: Fusion weight (1.0 = FAISS only, 0.0 = BM25 only)
            use_reranker: Enable cross-encoder re-ranking (slower but +5% accuracy)
        """
        # Load products
        with open(PRODUCTS_FILE, "r", encoding="utf8") as f:
            self.products: List[Dict[str, Any]] = json.load(f)
        
        # Load embeddings
        self.embeddings = np.load(EMBED_NPY, mmap_mode="r")
        
        # Load FAISS
        self.index = faiss.read_index(str(INDEX_FILE))
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search
        
        # GPU (optional)
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Encoder
        device = "cuda" if (use_gpu and self._cuda_available()) else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
         # === ADD BGE OPTIMIZATIONS ===
        if "bge" in model_name.lower() and device == "cuda":
            self.model = self.model.half()  # FP16 for speed
            print("🚀 Using FP16 for BGE model")
        
        # BM25
        print("Loading BM25 corpus...")
        with open(BM25_FILE, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25_corpus = bm25_data["corpus"]
        self.bm25 = bm25_data["model"]
        
        # Spell corrector
        vocab = set()
        for p in self.products:
            for field in ("title", "category_name"):
                v = p.get(field)
                if v:
                    for tok in clean_text(str(v)).split():
                        vocab.add(tok.lower())
        self.corrector = SpellCorrector(list(vocab))
        
        # Re-ranker (optional)
        self.use_reranker = use_reranker
        if use_reranker:
            print("Loading cross-encoder re-ranker...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.alpha = alpha
        print(f"✅ Hybrid retriever ready: {len(self.products):,} products")
        print(f"   Fusion: {alpha*100:.0f}% semantic, {(1-alpha)*100:.0f}% keyword")
        print(f"   Re-ranker: {'enabled' if use_reranker else 'disabled'}")
    
    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _extract_price_filter(self, query: str) -> Optional[float]:
        """Extract price from queries like 'under $50', 'less than 100'."""
        match = PRICE_PATTERN.search(query)
        if match:
            for g in match.groups():
                if g:
                    return float(g)
        return None
    
    def search_text(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        candidate_pool: int = 1024,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Hybrid search with automatic price extraction.
        """
        t0 = time.time()
        filters = filters or {}
        
        # === Extract price from query ===
        price_from_query = self._extract_price_filter(query)
        if price_from_query and "price_max" not in filters:
            filters["price_max"] = price_from_query
            print(f"[AutoFilter] Extracted price_max={price_from_query} from query")
        
        # === Spell correction ===
        corrected_query = self.corrector.correct_query(query)
        if corrected_query != query:
            print(f"[Spell] '{query}' → '{corrected_query}'")
        
        # === FAISS search (model-specific prefix) ===
        # Check if model is E5 based on the actual loaded model
        model_metadata = str(self.model).lower()
        if "e5" in model_metadata:
            q_text = f"query: {corrected_query}"  # E5 prefix
        else:
            q_text = corrected_query  # No prefix for BGE
        q_vec = self.model.encode([q_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)
        
        D_faiss, I_faiss = self.index.search(q_vec, candidate_pool)
        faiss_candidates = [int(i) for i in I_faiss[0] if i != -1]
        faiss_scores = {idx: float(D_faiss[0][i]) for i, idx in enumerate(faiss_candidates)}
        
        # === BM25 search ===
        query_tokens = clean_text(corrected_query).split()
        bm25_scores_all = self.bm25.get_scores(query_tokens)
        bm25_top = np.argsort(-bm25_scores_all)[:candidate_pool]
        bm25_candidates = [int(i) for i in bm25_top]
        bm25_scores = {idx: float(bm25_scores_all[idx]) for idx in bm25_candidates}
        
        # === Normalize scores [0, 1] ===
        def normalize(score_dict):
            if not score_dict:
                return {}
            vals = list(score_dict.values())
            min_s, max_s = min(vals), max(vals)
            if max_s == min_s:
                return {k: 1.0 for k in score_dict}
            return {k: (v-min_s)/(max_s-min_s) for k, v in score_dict.items()}
        
        f_norm = normalize(faiss_scores)
        b_norm = normalize(bm25_scores)
        
        # === Hybrid fusion ===
        all_idx = set(f_norm) | set(b_norm)
        fused = {
            i: self.alpha * f_norm.get(i, 0) + (1-self.alpha) * b_norm.get(i, 0)
            for i in all_idx
        }
        
        # === Apply filters BEFORE final ranking ===
        filtered_idx = [i for i in all_idx if self._passes_filters(self.products[i], filters)]
        
        if not filtered_idx:
            return [], corrected_query, filters
        
        # === Sort by fused score ===
        filtered_idx.sort(key=lambda i: fused[i], reverse=True)
        
        # === Optional: Cross-encoder re-ranking ===
        if self.use_reranker and len(filtered_idx) > top_k:
            rerank_candidates = filtered_idx[:min(50, len(filtered_idx))]
            pairs = [[corrected_query, self.products[i].get("title", "")] for i in rerank_candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            # Sort by reranker scores
            reranked = sorted(zip(rerank_candidates, rerank_scores), 
                            key=lambda x: x[1], reverse=True)
            filtered_idx = [idx for idx, _ in reranked]
        
        # === Build results ===
        results = []
        seen_asins = set()
        
        for idx in filtered_idx[:top_k * 3]:  # Get extras for dedup
            prod = dict(self.products[idx])
            asin = prod.get("asin") or prod.get("id")
            
            if asin in seen_asins:
                continue
            seen_asins.add(asin)
            
            prod["_score"] = fused[idx]
            prod["_faiss_score"] = faiss_scores.get(idx, 0.0)
            prod["_bm25_score"] = bm25_scores.get(idx, 0.0)
            results.append(prod)
            
            if len(results) >= top_k:
                break
        
        latency = (time.time() - t0) * 1000
        print(f"[Search] {len(results)} results in {latency:.1f}ms")
        
        return results, corrected_query, filters
    
    def _passes_filters(self, prod: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check all filters."""
        # Stars
        stars_min = filters.get("stars_min")
        stars_max = filters.get("stars_max")
        if stars_min and prod.get("stars", 0) < stars_min:
            return False
        if stars_max and prod.get("stars", 0) > stars_max:
            return False
        
        # Price
        price_min = filters.get("price_min")
        price_max = filters.get("price_max")
        if price_min and prod.get("price", 0) < price_min:
            return False
        if price_max and prod.get("price", 0) > price_max:
            return False
        
        # Bestseller
        if filters.get("isBestSeller") is not None:
            if prod.get("isBestSeller") != filters["isBestSeller"]:
                return False
        
        # Category
        category = filters.get("category_name")
        if category:
            prod_cat = prod.get("category_name", "").lower()
            if isinstance(category, list):
                if not any(c.lower() in prod_cat for c in category):
                    return False
            elif category.lower() not in prod_cat:
                return False
        
        return True
    
    def as_langchain_retriever(self, top_k: int = 5):
        """LangChain wrapper."""
        try:
            from langchain_core.vectorstores import VectorStore
        except:
            raise RuntimeError("LangChain not installed")
        
        class CustomVectorStore(VectorStore):
            def __init__(self, svc):
                self.svc = svc
            
            def similarity_search(self, query: str, k: int = top_k, **kwargs):
                results, _, _ = self.svc.search_text(query, top_k=k)
                return [{"page_content": r.get("title",""), "metadata": r} for r in results]
        
        return CustomVectorStore(self).as_retriever(search_kwargs={"k": top_k})


# === CLI for testing ===
if __name__ == "__main__":
    retriever = HybridRetriever(alpha=0.6, use_reranker=False)
    
    test_queries = [
        "nike running shoes",
        "cheap watches under $50",
        "B0979NG867",
        "boys dinosaur jacket",
        "3 piece luggage set navy blue",
    ]
    
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        results, corrected, filters = retriever.search_text(q, top_k=5)
        print(f"Corrected: {corrected}")
        print(f"Filters: {filters}")
        for i, r in enumerate(results, 1):
            title = r["title"][:70]
            score = r["_score"]
            faiss = r["_faiss_score"]
            bm25 = r["_bm25_score"]
            print(f"{i}. [{score:.3f}] (F:{faiss:.2f} B:{bm25:.2f}) {title}")
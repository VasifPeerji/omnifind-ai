# src/omnifind/retrieval/hybrid_retriever_v3.py
"""
PRODUCTION HYBRID RETRIEVER v3 - Amazon-Level Accuracy

Key Features:
1. Query understanding & dynamic routing (ASIN/branded/generic)
2. Attribute-aware pre-filtering (brand, color indexes)
3. Conservative spell correction
4. Dynamic alpha weighting based on query type
5. Multi-stage retrieval pipeline

Fixes:
- ✅ Brand matching (filters products by brand BEFORE ranking)
- ✅ Color matching (pre-filters by color)
- ✅ ASIN exact match (dedicated route, skips semantic search)
- ✅ Spell correction (protects "for", "women", "wear")
- ✅ Query understanding (different strategies for different query types)
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import faiss
import json
import pickle
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from omnifind.utils.spell_corrector import SpellCorrector
from omnifind.embeddings.preprocess_texts import clean_text
from omnifind.retrieval.query_analyzer import QueryAnalyzer

BASE_DIR = Path(__file__).parent.parent.parent
PRODUCTS_FILE = BASE_DIR / "data/embeddings/products.json"
EMBED_NPY = BASE_DIR / "data/embeddings/embeddings.npy"
INDEX_FILE = BASE_DIR / "data/embeddings/faiss_index.index"
BM25_FILE = BASE_DIR / "data/embeddings/bm25_corpus.pkl"


class HybridRetriever:
    """
    Production-grade retriever with query understanding and attribute filtering.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        use_gpu: bool = True,
        default_alpha: float = 0.6,
        use_reranker: bool = False,
        ef_search: int = 256,
    ):
        print("🚀 Initializing Production Retriever v3...")
        
        # Load products
        with open(PRODUCTS_FILE, "r", encoding="utf8") as f:
            self.products: List[Dict[str, Any]] = json.load(f)
        print(f"   ✓ Loaded {len(self.products):,} products")
        
        # Build attribute indexes for fast pre-filtering
        self._build_attribute_indexes()
        
        # Load embeddings
        self.embeddings = np.load(EMBED_NPY, mmap_mode="r")
        print(f"   ✓ Loaded embeddings: {self.embeddings.shape}")
        
        # Load FAISS
        self.index = faiss.read_index(str(INDEX_FILE))
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search
        print(f"   ✓ Loaded FAISS index (efSearch={ef_search})")
        
        # GPU (optional)
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("   ✓ Using GPU for FAISS")
        
        # Load encoder model
        device = "cuda" if (use_gpu and self._cuda_available()) else "cpu"
        if device == "cuda":
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.model = SentenceTransformer(model_name, device=device)
        
        # BGE optimization (FP16 for 2x speed)
        if "bge" in model_name.lower() and device == "cuda":
            self.model = self.model.half()
            print("   ✓ Using FP16 precision for BGE model")
        
        print(f"   ✓ Loaded encoder: {model_name} on {device}")
        
        # Load BM25
        with open(BM25_FILE, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25_corpus = bm25_data["corpus"]
        self.bm25 = bm25_data["model"]
        print(f"   ✓ Loaded BM25 corpus")
        
        # Initialize query analyzer
        self.query_analyzer = QueryAnalyzer()
        print("   ✓ Loaded query analyzer")
        
        # Initialize spell corrector (FIXED VERSION)
        vocab = set()
        for p in self.products:
            for field in ("title", "category_name"):
                v = p.get(field)
                if v:
                    for tok in clean_text(str(v)).split():
                        vocab.add(tok.lower())
        self.corrector = SpellCorrector(list(vocab), min_word_length=4)
        print("   ✓ Loaded conservative spell corrector")
        
        # Re-ranker (optional)
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("   ✓ Loaded cross-encoder re-ranker")
        
        self.default_alpha = default_alpha
        print(f"\n✅ Retriever ready! (default alpha={default_alpha})\n")
    
    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _build_attribute_indexes(self):
        """
        Build fast lookup indexes for brands, colors, and ASINs.
        This enables O(1) pre-filtering instead of scanning all products.
        """
        print("   Building attribute indexes...")
        
        self.brand_index: Dict[str, List[int]] = {}
        self.color_index: Dict[str, List[int]] = {}
        self.asin_index: Dict[str, int] = {}
        
        for idx, prod in enumerate(self.products):
            # ASIN exact match index
            asin = prod.get("asin")
            if asin:
                self.asin_index[str(asin).upper()] = idx
            
            # Brand extraction from title
            title = str(prod.get("title", "")).lower()
            for brand in QueryAnalyzer.BRANDS:
                if brand in title:
                    if brand not in self.brand_index:
                        self.brand_index[brand] = []
                    self.brand_index[brand].append(idx)
            
            # Color extraction from title
            for color in QueryAnalyzer.COLORS:
                if color in title:
                    if color not in self.color_index:
                        self.color_index[color] = []
                    self.color_index[color].append(idx)
        
        print(f"      ✓ Brands indexed: {len(self.brand_index)}")
        print(f"      ✓ Colors indexed: {len(self.color_index)}")
        print(f"      ✓ ASINs indexed: {len(self.asin_index)}")
    
    def search_text(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        candidate_pool: int = 2048,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Main search function with intelligent query understanding.
        
        Args:
            query: User search query
            top_k: Number of results to return
            filters: Additional filters (price, rating, etc.)
            candidate_pool: Size of initial candidate set
        
        Returns:
            (results, corrected_query, applied_filters)
        """
        t0 = time.time()
        filters = filters or {}
        
        # === STAGE 1: Query Understanding ===
        intent = self.query_analyzer.analyze(query)
        print(f"\n[QueryIntent]")
        print(f"  Type: {intent.query_type}")
        print(f"  Strategy: {intent.search_strategy}")
        print(f"  Alpha: {intent.alpha} ({int(intent.alpha*100)}% semantic, {int((1-intent.alpha)*100)}% keyword)")
        if intent.brand:
            print(f"  Brand: {intent.brand}")
        if intent.colors:
            print(f"  Colors: {intent.colors}")
        
        # === STAGE 2: ASIN Exact Match (bypass all other stages) ===
        if intent.query_type == 'asin':
            asin = intent.attributes.get('asin')
            if asin in self.asin_index:
                idx = self.asin_index[asin]
                prod = dict(self.products[idx])
                prod["_score"] = 1.0
                prod["_faiss_score"] = 0.0
                prod["_bm25_score"] = 1.0
                prod["_match_type"] = "asin_exact"
                latency = (time.time() - t0) * 1000
                print(f"[ASIN Match] ✓ Found: {prod.get('title', '')[:60]}... ({latency:.1f}ms)")
                return [prod], query, filters
            else:
                print(f"[ASIN Not Found] ✗ {asin} not in index")
                return [], query, filters
        
        # === STAGE 3: Spell Correction (conservative) ===
        corrected_query = self.corrector.correct_query(intent.clean_query, threshold=85)
        if corrected_query != intent.clean_query:
            print(f"[Spell] '{intent.clean_query}' → '{corrected_query}'")
        
        # === STAGE 4: Attribute Pre-Filtering ===
        # Build candidate mask from brand/color indexes
        candidate_mask = None
        
        if intent.brand and intent.brand in self.brand_index:
            brand_products = set(self.brand_index[intent.brand])
            candidate_mask = brand_products
            print(f"[Brand Filter] {len(brand_products):,} products match brand '{intent.brand}'")
        
        if intent.colors:
            color_products = set()
            for color in intent.colors:
                if color in self.color_index:
                    color_products.update(self.color_index[color])
            
            if color_products:
                if candidate_mask:
                    # Intersection: products must match BOTH brand AND color
                    candidate_mask = candidate_mask & color_products
                else:
                    candidate_mask = color_products
                print(f"[Color Filter] {len(candidate_mask or color_products):,} products after color filter")
        
        # === STAGE 5: Semantic Search (FAISS) ===
        # Determine query prefix based on model type
        model_name_lower = str(self.model).lower()
        if "e5" in model_name_lower:
            q_text = f"query: {corrected_query}"  # E5 prefix
        else:
            q_text = corrected_query  # No prefix for BGE
        
        # Encode query
        q_vec = self.model.encode([q_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)
        
        # Search FAISS (expand pool if we have filters)
        search_k = candidate_pool * 2 if candidate_mask else candidate_pool
        D_faiss, I_faiss = self.index.search(q_vec, search_k)
        
        faiss_candidates = [int(i) for i in I_faiss[0] if i != -1]
        
        # Apply attribute mask
        if candidate_mask:
            faiss_candidates = [i for i in faiss_candidates if i in candidate_mask]
            print(f"[FAISS] {len(faiss_candidates)} candidates after attribute filtering")
        
        # Build score dict
        faiss_scores = {}
        for rank, idx in enumerate(I_faiss[0]):
            if idx != -1 and (candidate_mask is None or idx in candidate_mask):
                faiss_scores[int(idx)] = float(D_faiss[0][rank])
        
        # === STAGE 6: Keyword Search (BM25) ===
        query_tokens = clean_text(corrected_query).split()
        bm25_scores_all = self.bm25.get_scores(query_tokens)
        bm25_top = np.argsort(-bm25_scores_all)[:search_k]
        
        bm25_candidates = [int(i) for i in bm25_top]
        
        # Apply attribute mask
        if candidate_mask:
            bm25_candidates = [i for i in bm25_candidates if i in candidate_mask]
            print(f"[BM25] {len(bm25_candidates)} candidates after attribute filtering")
        
        bm25_scores = {idx: float(bm25_scores_all[idx]) for idx in bm25_candidates}
        
        # === STAGE 7: Hybrid Fusion (dynamic alpha) ===
        alpha = intent.alpha  # Use query-specific alpha!
        
        def normalize(score_dict):
            """Min-max normalization to [0, 1]"""
            if not score_dict:
                return {}
            vals = list(score_dict.values())
            min_s, max_s = min(vals), max(vals)
            if max_s == min_s:
                return {k: 1.0 for k in score_dict}
            return {k: (v-min_s)/(max_s-min_s) for k, v in score_dict.items()}
        
        f_norm = normalize(faiss_scores)
        b_norm = normalize(bm25_scores)
        
        # Combine scores
        all_idx = set(f_norm) | set(b_norm)
        fused = {
            i: alpha * f_norm.get(i, 0) + (1-alpha) * b_norm.get(i, 0)
            for i in all_idx
        }
        
        # === STAGE 8: Apply Price/Rating Filters ===
        filtered_idx = [i for i in all_idx if self._passes_filters(self.products[i], filters)]
        
        if not filtered_idx:
            print("[No Results] After all filters")
            return [], corrected_query, filters
        
        # Sort by fused score
        filtered_idx.sort(key=lambda i: fused[i], reverse=True)
        
        # === STAGE 9: Cross-Encoder Re-ranking (optional) ===
        if self.use_reranker and len(filtered_idx) > top_k:
            rerank_candidates = filtered_idx[:min(50, len(filtered_idx))]
            pairs = [[corrected_query, self.products[i].get("title", "")] 
                    for i in rerank_candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            # Sort by reranker scores
            reranked = sorted(zip(rerank_candidates, rerank_scores), 
                            key=lambda x: x[1], reverse=True)
            filtered_idx = [idx for idx, _ in reranked]
            print(f"[Rerank] Applied cross-encoder to top 50 candidates")
        
        # === STAGE 10: Build Final Results ===
        results = []
        seen_asins = set()
        
        for idx in filtered_idx[:top_k * 3]:  # Get extras for deduplication
            prod = dict(self.products[idx])
            asin = prod.get("asin") or prod.get("id")
            
            # Deduplicate by ASIN
            if asin in seen_asins:
                continue
            seen_asins.add(asin)
            
            # Add scores
            prod["_score"] = fused[idx]
            prod["_faiss_score"] = faiss_scores.get(idx, 0.0)
            prod["_bm25_score"] = bm25_scores.get(idx, 0.0)
            prod["_match_type"] = intent.query_type
            results.append(prod)
            
            if len(results) >= top_k:
                break
        
        latency = (time.time() - t0) * 1000
        print(f"[Search Complete] ✓ {len(results)} results in {latency:.1f}ms\n")
        
        return results, corrected_query, filters
    
    def _passes_filters(self, prod: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply standard filters (price, rating, category, bestseller)"""
        # Star rating filters
        stars_min = filters.get("stars_min")
        stars_max = filters.get("stars_max")
        if stars_min and prod.get("stars", 0) < stars_min:
            return False
        if stars_max and prod.get("stars", 0) > stars_max:
            return False
        
        # Price filters
        price_min = filters.get("price_min")
        price_max = filters.get("price_max")
        if price_min and prod.get("price", 0) < price_min:
            return False
        if price_max and prod.get("price", 0) > price_max:
            return False
        
        # Bestseller filter
        if filters.get("isBestSeller") is not None:
            if prod.get("isBestSeller") != filters["isBestSeller"]:
                return False
        
        # Category filter
        category = filters.get("category_name")
        if category:
            prod_cat = prod.get("category_name", "").lower()
            if isinstance(category, list):
                if not any(c.lower() in prod_cat for c in category):
                    return False
            elif category.lower() not in prod_cat:
                return False
        
        return True


# === CLI Testing ===
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Production Retriever v3")
    print("=" * 80)
    
    retriever = HybridRetriever(use_reranker=False)
    
    test_queries = [
        "adidas running shoes",
        "nike black shoes for men",
        "red dress women",
        "B0979NG867",  # ASIN test
        "blue denim jacket",
        "boys dinosaur jacket",
        "cheap watches under $50",
    ]
    
    for q in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: '{q}'")
        print('='*80)
        
        results, corrected, filters = retriever.search_text(q, top_k=5)
        
        if results:
            print(f"\nTop {len(results)} Results:")
            for i, r in enumerate(results, 1):
                title = r["title"][:70]
                score = r["_score"]
                match_type = r.get("_match_type", "")
                brand = r.get("title", "").split()[0] if r.get("title") else ""
                print(f"{i}. [{score:.3f}] [{match_type}] {title}...")
        else:
            print("\n❌ No results found")
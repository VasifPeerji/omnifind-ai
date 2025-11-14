# src/omnifind/api/main.py
"""
Production FastAPI with:
- Hybrid retriever v3 (FAISS + BM25 + Query Understanding)
- Image search (CLIP-based)
- Backward compatibility
- Automatic price extraction
- Query logging & metrics
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Any
import time
import logging
import os
from datetime import datetime
from PIL import Image
import io

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Schemas ----------------------
class TextFilters(BaseModel):
    category_name: Optional[Union[str, List[str]]] = None
    brand: Optional[Union[str, List[str]]] = None
    category: Optional[Union[str, List[str]]] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    stars_min: Optional[float] = None
    stars_max: Optional[float] = None
    isBestSeller: Optional[bool] = None

class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query", min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[TextFilters] = None
    # NOTE: alpha is now dynamic per query, but kept for backward compatibility
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0,
                                   description="[Ignored in v3] Alpha is now auto-determined by query type")

class TextSearchResponse(BaseModel):
    query: str
    corrected_query: str
    corrected_filters: Dict[str, Any]
    results: List[Dict[str, Any]]
    latency_ms: Optional[float] = None
    count: Optional[int] = None
    retriever_type: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    num_products: int
    retriever_type: str
    version: str

class VisualTextSearchRequest(BaseModel):
    text: str = Field(..., description="Text description for visual search", min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    price_min: Optional[float] = None
    price_max: Optional[float] = None

# ---------------------- Factory ----------------------
def create_app(retriever=None) -> FastAPI:
    """
    Factory to create FastAPI app with configurable retriever.
    
    Env vars:
        USE_HYBRID=1 (default) - Use HybridRetriever v3
        USE_HYBRID=0 - Use old RetrieverService (FAISS only)
        OMNIFIND_TEST=1 - Use dummy retriever for testing
    """
    app = FastAPI(
        title='OmniFind AI - Production API v3',
        description='Amazon-level product search with query understanding',
        version='3.0.0'
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )
    
    # ---------------------- Initialize Retriever ----------------------
    if retriever is None:
        if os.getenv("OMNIFIND_TEST", "0") == "1":
            # Test mode - dummy retriever
            logger.info("🧪 Test mode: Using dummy retriever")
            class DummyRetriever:
                products = []
                def search_text(self, query, top_k=5, filters=None, **kwargs):
                    return ([], query, filters or {})
            retriever = DummyRetriever()
            retriever_type = "dummy"
        
        else:
            # Production mode - use v3 retriever
            use_hybrid = os.getenv("USE_HYBRID", "1") == "1"
            
            if use_hybrid:
                try:
                    # ✅ FIX: Import v3 retriever (or your actual file name)
                    from ..retrieval.hybrid_retriever import HybridRetriever
                    # If you saved as hybrid_retriever_v3.py:
                    # from ..retrieval.hybrid_retriever_v3 import HybridRetriever
                    
                    logger.info("🚀 Initializing HybridRetriever v3 (Query Understanding)...")
                    
                    # ✅ FIX: Use correct parameter name 'default_alpha' (not 'alpha')
                    retriever = HybridRetriever(
                        model_name=os.getenv("MODEL_NAME", "BAAI/bge-large-en-v1.5"),
                        use_gpu=os.getenv("USE_GPU", "1") == "1",
                        default_alpha=float(os.getenv("DEFAULT_ALPHA", "0.6")),  # ← FIXED
                        use_reranker=os.getenv("USE_RERANKER", "0") == "1",
                        ef_search=int(os.getenv("EF_SEARCH", "256")),
                    )
                    retriever_type = "hybrid_v3_query_understanding"
                    logger.info(f"✅ HybridRetriever v3 ready: {len(retriever.products):,} products")
                
                except FileNotFoundError as e:
                    logger.error(f"❌ Missing required files: {e}")
                    logger.info("⚠️  Run: python -m omnifind.embeddings.embedder --rebuild-index-only")
                    raise
                
                except ImportError as e:
                    logger.error(f"❌ Missing dependencies: {e}")
                    logger.info("⚠️  Install: pip install rapidfuzz")
                    raise
                
                except Exception as e:
                    logger.error(f"❌ Failed to load HybridRetriever: {e}")
                    raise
            
            else:
                # Explicitly use old retriever (fallback)
                logger.info("🔧 Using RetrieverService (FAISS only)")
                from ..retrieval.retriever import RetrieverService
                retriever = RetrieverService()
                retriever_type = "faiss_only"
    
    else:
        # Custom retriever injected (for testing)
        retriever_type = "custom"
        logger.info(f"🔌 Using injected retriever: {type(retriever).__name__}")
    
    # Store retriever type for endpoints
    app.state.retriever_type = retriever_type
    
    # Metrics
    search_count = {"total": 0, "errors": 0, "total_latency_ms": 0}
    
    # ---------------------- Endpoints ----------------------
    
    @app.get("/", response_model=HealthResponse)
    def health_check():
        """Health check with system info."""
        return {
            "status": "healthy",
            "message": "OmniFind AI v3 backend is running",
            "timestamp": datetime.utcnow().isoformat(),
            "num_products": len(retriever.products) if hasattr(retriever, 'products') else 0,
            "retriever_type": app.state.retriever_type,
            "version": "3.0.0"
        }
    
    @app.get("/metrics")
    def get_metrics():
        """System metrics."""
        total = max(1, search_count["total"])
        return {
            "searches": search_count["total"],
            "errors": search_count["errors"],
            "error_rate": search_count["errors"] / total,
            "avg_latency_ms": search_count["total_latency_ms"] / total,
            "num_products": len(retriever.products) if hasattr(retriever, 'products') else 0,
            "retriever_type": app.state.retriever_type,
        }
    
    @app.post("/search/text", response_model=TextSearchResponse)
    def search_text(req: TextSearchRequest):
        """
        Intelligent search with query understanding (v3).
        
        Features:
        - Auto-detects ASINs (e.g., "B0979NG867")
        - Extracts brands (e.g., "nike", "adidas")
        - Extracts colors (e.g., "red", "black")
        - Dynamic alpha based on query type
        - Conservative spell correction
        
        Example queries:
        - "nike running shoes" → brand filter + keyword-heavy search
        - "B0979NG867" → exact ASIN match
        - "red dress women" → color + gender filter
        - "cheap watches under $50" → auto price extraction
        """
        t0 = time.time()
        search_count["total"] += 1
        
        try:
            # Normalize filters (support both old and new field names)
            filters = {}
            if req.filters:
                filter_dict = req.filters.dict(exclude_none=True)
                
                # Handle category aliases
                category = (filter_dict.get("category_name") or 
                           filter_dict.get("category") or 
                           filter_dict.get("brand"))
                if category:
                    filters["category_name"] = category
                
                # Copy other filters
                for key in ["price_min", "price_max", "stars_min", "stars_max", "isBestSeller"]:
                    if key in filter_dict:
                        filters[key] = filter_dict[key]
            
            # ✅ FIX: V3 retriever doesn't use 'alpha' parameter
            # Alpha is now determined automatically by query type
            results, corrected_query, corrected_filters = retriever.search_text(
                query=req.query,
                top_k=req.top_k,
                filters=filters
            )
            
            latency_ms = (time.time() - t0) * 1000
            search_count["total_latency_ms"] += latency_ms
            
            # Log with query intent info if available
            log_data = {
                "query": req.query,
                "corrected": corrected_query,
                "results": len(results),
                "latency_ms": f"{latency_ms:.1f}",
                "filters": corrected_filters,
            }
            
            # Add match type from results if available
            if results and "_match_type" in results[0]:
                log_data["match_type"] = results[0]["_match_type"]
            
            logger.info(log_data)
            
            return {
                "query": req.query,
                "corrected_query": corrected_query,
                "corrected_filters": corrected_filters,
                "results": results,
                "latency_ms": latency_ms,
                "count": len(results),
                "retriever_type": app.state.retriever_type,
            }
        
        except Exception as e:
            search_count["errors"] += 1
            logger.error(f"Search failed for query '{req.query}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/search/image")
    async def search_by_image(
        file: UploadFile = File(...),
        top_k: int = 5,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
    ):
        """
        Search by uploaded image (Google Lens style).
        """
        try:
            # Lazy load image retriever
            if not hasattr(app.state, 'image_retriever'):
                from ..retrieval.image_retriever import ImageRetriever
                logger.info("📸 Initializing ImageRetriever...")
                app.state.image_retriever = ImageRetriever()
            
            # Read uploaded image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Build filters
            filters = {}
            if price_min:
                filters["price_min"] = price_min
            if price_max:
                filters["price_max"] = price_max
            
            t0 = time.time()
            results = app.state.image_retriever.search_by_image(
                image, top_k=top_k, filters=filters
            )
            latency_ms = (time.time() - t0) * 1000
            
            logger.info(f"Image search: {len(results)} results in {latency_ms:.1f}ms")
            
            return {
                "results": results,
                "count": len(results),
                "search_type": "image",
                "latency_ms": latency_ms,
            }
        
        except Exception as e:
            logger.error(f"Image search failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/search/visual-text")
    def search_visual_text(req: VisualTextSearchRequest):
        """
        Text-to-image search using CLIP.
        
        Example: "red evening dress" → visually matching products
        """
        try:
            if not hasattr(app.state, 'image_retriever'):
                from ..retrieval.image_retriever import ImageRetriever
                logger.info("📸 Initializing ImageRetriever...")
                app.state.image_retriever = ImageRetriever()
            
            # Build filters
            filters = {}
            if req.price_min:
                filters["price_min"] = req.price_min
            if req.price_max:
                filters["price_max"] = req.price_max
            
            t0 = time.time()
            results = app.state.image_retriever.search_by_text(
                req.text, top_k=req.top_k, filters=filters
            )
            latency_ms = (time.time() - t0) * 1000
            
            logger.info(f"Visual-text search '{req.text}': {len(results)} results in {latency_ms:.1f}ms")
            
            return {
                "query": req.text,
                "results": results,
                "count": len(results),
                "search_type": "visual_text",
                "latency_ms": latency_ms,
            }
        
        except Exception as e:
            logger.error(f"Visual text search failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/search/batch")
    def search_batch(queries: List[str], top_k: int = 5):
        """Batch search for multiple queries."""
        results = []
        for q in queries:
            try:
                res, corrected, filters = retriever.search_text(q, top_k=top_k)
                results.append({
                    "query": q,
                    "corrected_query": corrected,
                    "results": res,
                })
            except Exception as e:
                logger.error(f"Batch query '{q}' failed: {e}")
                results.append({"query": q, "error": str(e)})
        
        return {"results": results, "retriever_type": app.state.retriever_type}
    
    return app

# ---------------------- Uvicorn Entry ----------------------
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
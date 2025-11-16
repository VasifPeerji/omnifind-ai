"""
Production FastAPI with:
- Hybrid retriever v3 (FAISS + BM25 + Query Understanding)
- Production-optimized image search (CLIP ViT-L/14 + TTA)
- Text-to-image search (visual semantic search)
- Hybrid image+text search
- Backward compatibility
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
    visual_search_available: bool
    version: str

class VisualTextSearchRequest(BaseModel):
    text: str = Field(..., description="Text description for visual search", min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    category_name: Optional[str] = None
    stars_min: Optional[float] = None

class HybridVisualSearchRequest(BaseModel):
    text: str = Field(..., description="Text description to combine with image")
    top_k: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, 
                        description="Weight for image (0.5 = equal image+text)")
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
        ENABLE_IMAGE_SEARCH=1 - Enable visual search (requires image index)
        ENABLE_TTA=1 - Enable Test-Time Augmentation for image queries
        ENABLE_RERANKING=0 - Enable cross-encoder re-ranking
    """
    app = FastAPI(
        title='OmniFind AI - Production API v3',
        description='Amazon-level product search with query understanding + visual search',
        version='3.0.0'
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )
    
    # ---------------------- Initialize Text Retriever ----------------------
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
                    from ..retrieval.hybrid_retriever import HybridRetriever
                    
                    logger.info("🚀 Initializing HybridRetriever v3 (Query Understanding)...")
                    
                    retriever = HybridRetriever(
                        model_name=os.getenv("MODEL_NAME", "BAAI/bge-large-en-v1.5"),
                        use_gpu=os.getenv("USE_GPU", "1") == "1",
                        default_alpha=float(os.getenv("DEFAULT_ALPHA", "0.6")),
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
    search_count = {
        "total": 0, 
        "errors": 0, 
        "total_latency_ms": 0,
        "text_searches": 0,
        "image_searches": 0,
        "visual_text_searches": 0,
        "hybrid_searches": 0,
    }
    
    # ---------------------- Initialize Image Retriever (Lazy) ----------------------
    def get_image_retriever():
        """Lazy load image retriever with production settings"""
        if not hasattr(app.state, 'image_retriever'):
            try:
                from ..retrieval.image_retriever import ImageRetriever
                
                logger.info("📸 Initializing Production ImageRetriever...")
                
                enable_tta = os.getenv("ENABLE_TTA", "1") == "1"
                enable_reranking = os.getenv("ENABLE_RERANKING", "0") == "1"
                
                app.state.image_retriever = ImageRetriever(
                    model_name="clip-ViT-L-14",  # Production model
                    use_gpu=os.getenv("USE_GPU", "1") == "1",
                    use_fp16=True,
                    enable_tta=enable_tta,
                    enable_reranking=enable_reranking,
                    cache_size=1000,
                )
                
                logger.info(f"✅ ImageRetriever ready: {len(app.state.image_retriever.products):,} products")
                logger.info(f"   TTA: {enable_tta} | Re-ranking: {enable_reranking}")
                
            except FileNotFoundError as e:
                logger.error(f"❌ Image index not found: {e}")
                logger.info("💡 Build image index: python -m omnifind.embeddings.image_embedder --enable-preprocessing")
                raise HTTPException(
                    status_code=503, 
                    detail="Visual search not available. Image index not built."
                )
            except Exception as e:
                logger.error(f"❌ Failed to load ImageRetriever: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return app.state.image_retriever
    
    # ---------------------- Endpoints ----------------------
    
    @app.get("/", response_model=HealthResponse)
    def health_check():
        """Health check with system info."""
        # Check if image retriever is available
        visual_available = False
        try:
            if hasattr(app.state, 'image_retriever'):
                visual_available = True
            else:
                # Try to check if image files exist without loading
                from pathlib import Path
                image_index = Path("data/embeddings/image_faiss_index.index")
                visual_available = image_index.exists()
        except:
            pass
        
        return {
            "status": "healthy",
            "message": "OmniFind AI v3 backend is running",
            "timestamp": datetime.utcnow().isoformat(),
            "num_products": len(retriever.products) if hasattr(retriever, 'products') else 0,
            "retriever_type": app.state.retriever_type,
            "visual_search_available": visual_available,
            "version": "3.0.0"
        }
    
    @app.get("/metrics")
    def get_metrics():
        """System metrics."""
        total = max(1, search_count["total"])
        
        metrics = {
            "searches": {
                "total": search_count["total"],
                "text": search_count["text_searches"],
                "image": search_count["image_searches"],
                "visual_text": search_count["visual_text_searches"],
                "hybrid": search_count["hybrid_searches"],
            },
            "errors": search_count["errors"],
            "error_rate": search_count["errors"] / total,
            "avg_latency_ms": search_count["total_latency_ms"] / total,
            "num_products": len(retriever.products) if hasattr(retriever, 'products') else 0,
            "retriever_type": app.state.retriever_type,
        }
        
        # Add image retriever stats if available
        if hasattr(app.state, 'image_retriever'):
            img_stats = app.state.image_retriever.get_stats()
            metrics["image_retriever"] = {
                "cache_hit_rate": img_stats["cache_hit_rate"],
                "tta_enabled": img_stats["tta_enabled"],
                "reranking_enabled": img_stats["reranking_enabled"],
            }
        
        return metrics
    
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
        search_count["text_searches"] += 1
        
        try:
            # Normalize filters
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
            
            # V3 retriever - alpha is determined automatically
            results, corrected_query, corrected_filters = retriever.search_text(
                query=req.query,
                top_k=req.top_k,
                filters=filters
            )
            
            latency_ms = (time.time() - t0) * 1000
            search_count["total_latency_ms"] += latency_ms
            
            # Log with query intent info
            log_data = {
                "query": req.query,
                "corrected": corrected_query,
                "results": len(results),
                "latency_ms": f"{latency_ms:.1f}",
                "filters": corrected_filters,
            }
            
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
        category_name: Optional[str] = None,
        stars_min: Optional[float] = None,
    ):
        """
        Search by uploaded image (Google Lens style).
        
        Features:
        - Production CLIP ViT-L/14 model
        - Test-Time Augmentation (TTA) for robust matching
        - Advanced filtering
        - Result caching
        """
        search_count["total"] += 1
        search_count["image_searches"] += 1
        
        try:
            image_retriever = get_image_retriever()
            
            # Read uploaded image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Build filters
            filters = {}
            if price_min:
                filters["price_min"] = price_min
            if price_max:
                filters["price_max"] = price_max
            if category_name:
                filters["category_name"] = category_name
            if stars_min:
                filters["stars_min"] = stars_min
            
            t0 = time.time()
            results = image_retriever.search_by_image(
                image, top_k=top_k, filters=filters
            )
            latency_ms = (time.time() - t0) * 1000
            search_count["total_latency_ms"] += latency_ms
            
            logger.info(f"Image search: {len(results)} results in {latency_ms:.1f}ms")
            
            return {
                "results": results,
                "count": len(results),
                "search_type": "image",
                "latency_ms": latency_ms,
                "tta_enabled": image_retriever.enable_tta,
            }
        
        except HTTPException:
            raise
        except Exception as e:
            search_count["errors"] += 1
            logger.error(f"Image search failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/search/visual-text")
    def search_visual_text(req: VisualTextSearchRequest):
        """
        Text-to-image search using CLIP (visual semantic search).
        
        Example: "red evening dress" → finds visually matching products
        
        This uses CLIP's text encoder to find products that LOOK like
        the text description, not just keyword matching.
        """
        search_count["total"] += 1
        search_count["visual_text_searches"] += 1
        
        try:
            image_retriever = get_image_retriever()
            
            # Build filters
            filters = {}
            if req.price_min:
                filters["price_min"] = req.price_min
            if req.price_max:
                filters["price_max"] = req.price_max
            if req.category_name:
                filters["category_name"] = req.category_name
            if req.stars_min:
                filters["stars_min"] = req.stars_min
            
            t0 = time.time()
            results = image_retriever.search_by_text(
                req.text, top_k=req.top_k, filters=filters
            )
            latency_ms = (time.time() - t0) * 1000
            search_count["total_latency_ms"] += latency_ms
            
            logger.info(f"Visual-text search '{req.text}': {len(results)} results in {latency_ms:.1f}ms")
            
            return {
                "query": req.text,
                "results": results,
                "count": len(results),
                "search_type": "visual_text",
                "latency_ms": latency_ms,
            }
        
        except HTTPException:
            raise
        except Exception as e:
            search_count["errors"] += 1
            logger.error(f"Visual text search failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/search/hybrid-visual")
    async def search_hybrid_visual(
        file: UploadFile = File(...),
        text: str = "",
        top_k: int = 5,
        alpha: float = 0.5,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
    ):
        """
        Hybrid image + text search.
        
        Combines visual similarity with text semantic matching.
        
        Example:
        - Upload image of shoes + text "nike" → finds Nike shoes visually similar
        - Upload dress image + text "formal evening" → finds formal dresses similar to image
        
        Args:
            file: Image file
            text: Text description to combine
            alpha: Weight for image (0.5 = equal, 0.7 = 70% image / 30% text)
        """
        search_count["total"] += 1
        search_count["hybrid_searches"] += 1
        
        try:
            image_retriever = get_image_retriever()
            
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
            results = image_retriever.hybrid_search(
                image=image,
                text=text if text.strip() else None,
                top_k=top_k,
                alpha=alpha,
                filters=filters,
            )
            latency_ms = (time.time() - t0) * 1000
            search_count["total_latency_ms"] += latency_ms
            
            logger.info(f"Hybrid search (alpha={alpha}): {len(results)} results in {latency_ms:.1f}ms")
            
            return {
                "results": results,
                "count": len(results),
                "search_type": "hybrid_visual",
                "alpha": alpha,
                "text_query": text if text.strip() else None,
                "latency_ms": latency_ms,
            }
        
        except HTTPException:
            raise
        except Exception as e:
            search_count["errors"] += 1
            logger.error(f"Hybrid visual search failed: {e}", exc_info=True)
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
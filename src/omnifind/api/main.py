# src/omnifind/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Union

from ..retrieval.retriever import RetrieverService

# ---------------------- Schemas ----------------------
class TextFilters(BaseModel):
    brand: Optional[Union[str, List[str]]] = None
    category: Optional[Union[str, List[str]]] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None

class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    top_k: int = 5
    filters: Optional[TextFilters] = None

# ---------------------- Factory ----------------------
def create_app(retriever: RetrieverService = None) -> FastAPI:
    """
    Factory to create FastAPI app.
    Allows injecting a custom retriever for testing.
    """
    app = FastAPI(title='OmniFind AI API')

    # CORS middleware for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # Use provided retriever (for tests) or real one
    if retriever is None:
        retriever = RetrieverService()

    # ---------- Health check ----------
    @app.get('/')
    def read_root():
        return {'message': 'OmniFind AI backend is running'}

    # ---------- Text search ----------
    @app.post('/search/text')
    def search_text(req: TextSearchRequest):
        filters = req.filters.dict() if req.filters else {}

        results, corrected_query, corrected_filters = retriever.search_text(
            req.query, top_k=req.top_k, filters=filters
        )

        return {
            "query": req.query,
            "corrected_query": corrected_query,
            "corrected_filters": corrected_filters,
            "results": results
        }

    return app

# ---------------------- Uvicorn entry ----------------------
# Expose a top-level 'app' for Uvicorn
app = create_app()

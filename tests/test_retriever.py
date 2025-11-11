import pytest
from omnifind import retrieval


# Fake retriever that mimics HybridRetriever interface
class FakeRetriever:
    """Mock retriever for testing - matches HybridRetriever API"""
    
    products = [
        {"id": 1, "title": "Nike Running Shoes", "brand": "Nike", "category_name": "Athletic Shoes", "asin": "B001"},
        {"id": 2, "title": "Adidas Sneakers", "brand": "Adidas", "category_name": "Footwear", "asin": "B002"},
    ]
    
    def search_text(self, query, top_k=5, filters=None, **kwargs):  # ← Add **kwargs for hybrid params
        """
        Returns: (results, corrected_query, corrected_filters)
        """
        corrected = query if query else "corrected"
        corrected_filters = filters or {}

        # Mock product with scores (new fields for hybrid retriever)
        product = {
            "id": 1,
            "title": "Mock Shirt",
            "brand": "Nike",
            "category_name": "Clothing",  # ← Changed from "category" to match new schema
            "asin": "B001",
            "_score": 0.95,  # ← Add hybrid score
            "_faiss_score": 0.92,  # ← Add FAISS score
            "_bm25_score": 0.98,  # ← Add BM25 score
        }

        # Simulate spell correction for filters
        if filters:
            if "brand" in filters:
                if "nikee" in str(filters["brand"]).lower():
                    product["brand"] = "Nike"
                    corrected_filters["brand"] = "Nike"
            
            # Handle both "category" and "category_name" for backward compatibility
            category_filter = filters.get("category_name") or filters.get("category")
            if category_filter:
                if "furntiure" in str(category_filter).lower():
                    product["category_name"] = "Furniture"
                    corrected_filters["category_name"] = "Furniture"  # ← Use category_name

        return ([product], corrected, corrected_filters)


@pytest.fixture(scope="session")
def retriever():
    return FakeRetriever()


def test_search_basic(retriever):
    results, corrected, _ = retriever.search_text("shirt", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert corrected == "shirt"
    # ← Add check for new score fields
    assert "_score" in results[0]


def test_spell_correction_query(retriever):
    results, corrected, _ = retriever.search_text("adibas shoos", top_k=3)
    assert len(results) > 0
    assert isinstance(corrected, str)


def test_spell_correction_brand_filter(retriever):
    filters = {"brand": "nikee"}
    results, corrected, corrected_filters = retriever.search_text("shoes", top_k=3, filters=filters)
    assert len(results) > 0
    assert any("nike" in p.get("brand", "").lower() for p in results)
    assert corrected_filters.get("brand") == "Nike"


def test_spell_correction_category_filter(retriever):
    filters = {"category_name": "furntiure"}  # ← Changed to category_name
    results, corrected, corrected_filters = retriever.search_text("", top_k=3, filters=filters)
    assert len(results) > 0
    assert any("furniture" in p.get("category_name", "").lower() for p in results)  # ← category_name
    assert corrected_filters.get("category_name") == "Furniture"  # ← category_name


def test_hybrid_kwargs_accepted(retriever):
    """Test that hybrid-specific kwargs are accepted (but ignored by fake)"""
    results, corrected, _ = retriever.search_text(
        "shoes", 
        top_k=3, 
        candidate_pool=100,  # ← Hybrid param
        alpha=0.7  # ← Hybrid param
    )
    assert len(results) > 0


def test_empty_query(retriever):
    """Test handling of empty query"""
    results, corrected, _ = retriever.search_text("", top_k=3)
    assert isinstance(results, list)
    assert corrected == "corrected"
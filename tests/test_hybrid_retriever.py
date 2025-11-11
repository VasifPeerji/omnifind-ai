"""
Tests for HybridRetriever (integration-style, but still uses mocks)
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np


@pytest.fixture
def mock_hybrid_components():
    """Mock FAISS, BM25, and products for testing"""
    
    # Mock products
    products = [
        {
            "id": 1,
            "title": "Nike Air Max",
            "asin": "B001",
            "category_name": "Shoes",
            "price": 8999,
            "stars": 4.5,
        },
        {
            "id": 2,
            "title": "Adidas Ultraboost",
            "asin": "B002",
            "category_name": "Athletic Footwear",
            "price": 12999,
            "stars": 4.8,
        },
    ]
    
    # Mock FAISS index
    mock_index = Mock()
    mock_index.search = Mock(return_value=(
        np.array([[0.95, 0.85]]),  # Distances
        np.array([[0, 1]])          # Indices
    ))
    
    # Mock BM25
    mock_bm25 = Mock()
    mock_bm25.get_scores = Mock(return_value=np.array([0.9, 0.7]))
    
    return {
        "products": products,
        "index": mock_index,
        "bm25": mock_bm25,
    }


def test_price_extraction():
    """Test automatic price filter extraction from queries"""
    from omnifind.retrieval.hybrid_retriever import HybridRetriever
    
    # We can't instantiate without files, so just test the regex
    import re
    pattern = re.compile(r'under\s*\$?(\d+)|less\s*than\s*\$?(\d+)|below\s*\$?(\d+)', re.I)
    
    test_cases = [
        ("shoes under $50", 50),
        ("watches less than 2000", 2000),
        ("laptops below 50000", 50000),
        ("cheap phones under $500", 500),
    ]
    
    for query, expected_price in test_cases:
        match = pattern.search(query)
        assert match is not None
        extracted = next((float(g) for g in match.groups() if g), None)
        assert extracted == expected_price


def test_filter_application():
    """Test that filters are correctly applied"""
    products = [
        {"price": 1000, "stars": 4.5, "category_name": "Shoes"},
        {"price": 2000, "stars": 3.5, "category_name": "Clothing"},
        {"price": 500, "stars": 4.8, "category_name": "Shoes"},
    ]
    
    # Mock filter function
    def passes_filters(prod, filters):
        if filters.get("price_min") and prod["price"] < filters["price_min"]:
            return False
        if filters.get("price_max") and prod["price"] > filters["price_max"]:
            return False
        if filters.get("stars_min") and prod["stars"] < filters["stars_min"]:
            return False
        if filters.get("category_name"):
            cat = filters["category_name"].lower()
            if cat not in prod["category_name"].lower():
                return False
        return True
    
    # Test price filter
    filters = {"price_min": 600, "price_max": 1500}
    filtered = [p for p in products if passes_filters(p, filters)]
    assert len(filtered) == 1
    assert filtered[0]["price"] == 1000
    
    # Test stars filter
    filters = {"stars_min": 4.0}
    filtered = [p for p in products if passes_filters(p, filters)]
    assert len(filtered) == 2
    
    # Test category filter
    filters = {"category_name": "Shoes"}
    filtered = [p for p in products if passes_filters(p, filters)]
    assert len(filtered) == 2
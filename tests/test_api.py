from fastapi.testclient import TestClient
from omnifind.api.main import create_app


class FakeRetriever:
    """Mock retriever for API testing"""
    
    products = [
        {"id": 1, "title": "Mock Product", "asin": "B001"},
    ]
    
    def search_text(self, query, top_k=5, filters=None, **kwargs):
        """Match HybridRetriever signature"""
        return (
            [{
                "id": 1, 
                "title": "Mock Product",
                "asin": "B001",
                "_score": 0.95,
                "_faiss_score": 0.92,
                "_bm25_score": 0.98,
            }], 
            query,  # corrected_query
            filters or {}  # corrected_filters
        )


# Inject fake retriever
app = create_app(retriever=FakeRetriever())
client = TestClient(app)


def test_health_check():
    """Test / endpoint returns health status"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "retriever_type" in data
    assert data["retriever_type"] == "custom"  # Because we injected a custom retriever


def test_search_endpoint():
    """Test /search/text endpoint"""
    response = client.post("/search/text", json={"query": "mock"})
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0
    
    # Check new fields
    assert "latency_ms" in data
    assert "count" in data
    assert "retriever_type" in data
    assert "corrected_query" in data
    assert "corrected_filters" in data
    
    # Check result has score fields
    result = data["results"][0]
    assert "_score" in result
    assert "_faiss_score" in result
    assert "_bm25_score" in result


def test_search_with_filters():
    """Test search with price and category filters"""
    response = client.post("/search/text", json={
        "query": "shoes",
        "top_k": 5,
        "filters": {
            "price_min": 100,
            "price_max": 5000,
            "category_name": "Footwear"
        }
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_search_with_alpha():
    """Test search with custom alpha (hybrid fusion weight)"""
    response = client.post("/search/text", json={
        "query": "shoes",
        "top_k": 3,
        "alpha": 0.8  # 80% semantic, 20% keyword
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_metrics_endpoint():
    """Test /metrics endpoint"""
    # Make a search first
    client.post("/search/text", json={"query": "test"})
    
    # Check metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "searches" in data
    assert "errors" in data
    assert "avg_latency_ms" in data
    assert data["searches"] >= 1  # At least one search was made


def test_empty_query_validation():
    """Test that empty query is rejected"""
    response = client.post("/search/text", json={
        "query": "",  # Empty query
        "top_k": 5
    })
    assert response.status_code == 422  # Validation error


def test_invalid_top_k():
    """Test that invalid top_k is rejected"""
    response = client.post("/search/text", json={
        "query": "shoes",
        "top_k": 100  # Exceeds max of 50
    })
    assert response.status_code == 422  # Validation error
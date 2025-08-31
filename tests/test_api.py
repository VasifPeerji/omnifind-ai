import pytest
from fastapi.testclient import TestClient
import src.omnifind.api.main as main  # import the module, not just app

# 🔹 Override retriever with a fake one
class FakeRetriever:
    def search_text(self, query, top_k=5, filters=None):
        return ([{"id": 1, "title": "Mock Product", "brand": "TestBrand", "price": 100}], query)

# Replace the real retriever
main.retriever = FakeRetriever()

client = TestClient(main.app)

def test_search_text():
    resp = client.post("/search/text", json={"query": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["results"][0]["title"] == "Mock Product"

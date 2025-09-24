from fastapi.testclient import TestClient
from omnifind.api.main import create_app

class FakeRetriever:
    def search_text(self, query, top_k=5, filters=None):
        return ([{"id": 1, "title": "Mock Product"}], query, {})

# Inject fake retriever
app = create_app(retriever=FakeRetriever())
client = TestClient(app)

def test_search_endpoint():
    response = client.post("/search/text", json={"query": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)

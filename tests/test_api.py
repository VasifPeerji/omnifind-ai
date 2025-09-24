import src.omnifind.api.main as main
from fastapi.testclient import TestClient


class FakeRetriever:
    def search_text(self, query, top_k=5, filters=None):
        return ([{"id": 1, "title": "Mock Product"}], query)


# Override dependency for tests
main.app.dependency_overrides[main.get_retriever] = lambda: FakeRetriever()

client = TestClient(main.app)


def test_search_endpoint():
    response = client.get("/search", params={"q": "mock"})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)

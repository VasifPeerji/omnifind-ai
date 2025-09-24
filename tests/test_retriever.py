import pytest
from src.omnifind import retrieval


# Fake retriever that mimics the interface of RetrieverService
class FakeRetriever:
    def search_text(self, query, top_k=5, filters=None):
        corrected = query if query else "corrected"

        product = {
            "id": 1,
            "title": "Mock Shirt",
            "brand": "Nike",
            "category": "Clothing",
        }

        # Simulate simple spell correction for filters
        if filters:
            if "brand" in filters:
                if "nikee" in filters["brand"].lower():
                    product["brand"] = "Nike"
            if "category" in filters:
                if "furntiure" in filters["category"].lower():
                    product["category"] = "Furniture"

        return ([product], corrected)


@pytest.fixture(scope="session")
def retriever():
    return FakeRetriever()


def test_search_basic(retriever):
    results, corrected = retriever.search_text("shirt", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert corrected == "shirt"


def test_spell_correction_query(retriever):
    results, corrected = retriever.search_text("adibas shoos", top_k=3)
    assert len(results) > 0
    assert isinstance(corrected, str)


def test_spell_correction_brand_filter(retriever):
    filters = {"brand": "nikee"}
    results, corrected = retriever.search_text("shoes", top_k=3, filters=filters)
    assert len(results) > 0
    assert any("nike" in p.get("brand", "").lower() for p in results)


def test_spell_correction_category_filter(retriever):
    filters = {"category": "furntiure"}
    results, corrected = retriever.search_text("", top_k=3, filters=filters)
    assert len(results) > 0
    assert any("furniture" in p.get("category", "").lower() for p in results)

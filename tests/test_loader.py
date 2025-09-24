import pytest
from src.omnifind.data import loader


def test_loader_returns_list(monkeypatch):
    # Mock load_sample_catalog to avoid file I/O
    monkeypatch.setattr(loader, "load_sample_catalog", lambda: [{"id": 1, "title": "Mock Product"}])

    data = loader.load_sample_catalog()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]

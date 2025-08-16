from src.omnifind.data.loader import load_sample_catalog

def test_loader_returns_list():
    data = load_sample_catalog()
    assert isinstance(data, list)
    if len(data):
        assert 'id' in data[0]

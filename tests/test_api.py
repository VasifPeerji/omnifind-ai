from fastapi.testclient import TestClient
from src.omnifind.api.main import app

client = TestClient(app)

def test_root():
    r = client.get('/')
    assert r.status_code == 200
    assert 'backend' in r.json().get('message', '').lower() or 'running' in r.json().get('message', '').lower()

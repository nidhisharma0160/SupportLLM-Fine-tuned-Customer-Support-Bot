import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.api as api

client = TestClient(api.app)

@pytest.fixture(autouse=True)
def mock_engine_and_intents():
    # Setup mocks
    api.intents = ["cancel_order", "track_package", "refund_request"]
    
    mock_engine = MagicMock()
    
    # Mock the generate async generator method
    async def mock_generate(prompt, max_tokens=15, temperature=0.0):
        yield "cancel_"
        yield "order"
        
    mock_engine.generate = mock_generate
    api.engine = mock_engine
    yield
    # Teardown
    api.engine = None
    api.intents = []

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_classify_json():
    response = client.post("/classify", json={"query": "Cancel my order", "stream": False})
    assert response.status_code == 200
    assert response.json() == {"intent": "cancel_order"}

def test_classify_stream():
    response = client.post("/classify", json={"query": "Cancel my order", "stream": True})
    assert response.status_code == 200
    assert response.text == "cancel_order"

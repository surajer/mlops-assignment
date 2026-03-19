import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_prediction():
    payload = {"input": [[5.1, 3.5, 1.4, 0.2]]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
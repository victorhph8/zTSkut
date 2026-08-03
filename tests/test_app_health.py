from pathlib import Path
import sys

# Add repo root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_homepage_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "zTSkut" in response.text or "Predict zT" in response.text

def test_robots_txt_is_available():
    response = client.get("/robots.txt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "User-agent: *" in response.text
    assert "Allow: /" in response.text
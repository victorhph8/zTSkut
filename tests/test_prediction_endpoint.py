from pathlib import Path
import sys
import io
import pandas as pd
import json
import base64

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def valid_input_dataframe():
    return pd.DataFrame([{
        "a_comp_1": "Sb", "a_frac_1": 12,
        "a_comp_2": 0, "a_frac_2": 0,
        "a_comp_3": 0, "a_frac_3": 0,
        "c_comp_1": "Co", "c_frac_1": 4,
        "c_comp_2": 0, "c_frac_2": 0,
        "c_comp_3": 0, "c_frac_3": 0,
        "f_comp_1": "Yb", "f_frac_1": 0.1,
        "f_comp_2": "Nd", "f_frac_2": 0.2,
        "f_comp_3": 0, "f_frac_3": 0,
        "f_comp_4": 0, "f_frac_4": 0,
        "f_comp_5": 0, "f_frac_5": 0,
        "n_300": 1e20,
        "T": 700,
        "p_n": 1,
        "ZT": 0
    }])
    
def assert_valid_prediction_response(response):
    assert response.status_code == 200
    
    data = response.json()
    
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert isinstance(data["predictions"][0], float)
    
    assert "n_predictions" in data
    assert data["n_predictions"] == 1

    assert "results" in data
    assert len(data["results"]) == 1
    assert "predicted_ZT" in data["results"][0]
    assert isinstance(data["results"][0]["predicted_ZT"], float)

    assert "downloads" in data
    assert "csv" in data["downloads"]
    assert "json" in data["downloads"]
    assert "xlsx_base64" in data["downloads"]

    csv_df = pd.read_csv(io.StringIO(data["downloads"]["csv"]))
    assert "predicted_ZT" in csv_df.columns
    assert len(csv_df) == 1

    json_records = json.loads(data["downloads"]["json"])
    assert len(json_records) == 1
    assert "predicted_ZT" in json_records[0]

    xlsx_bytes = base64.b64decode(data["downloads"]["xlsx_base64"])
    xlsx_df = pd.read_excel(io.BytesIO(xlsx_bytes), engine="openpyxl")
    assert "predicted_ZT" in xlsx_df.columns
    assert len(xlsx_df) == 1

def test_predict_endpoint_with_valid_csv():
    df = valid_input_dataframe()
    csv_content = df.to_csv(index=False)

    response = client.post(
        "/predict",
        files={"file": ("test.csv", csv_content.encode("utf-8"), "text/csv")}
    )
    
    assert_valid_prediction_response(response)
    
def test_predict_endpoint_with_valid_json():
    df = valid_input_dataframe()
    json_content = df.to_json(orient="records")

    response = client.post(
        "/predict",
        files={"file": ("test.json", json_content.encode("utf-8"), "application/json")}
    )

    assert_valid_prediction_response(response)
    
def test_predict_endpoint_with_valid_xlsx():
    df = valid_input_dataframe()

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.xlsx",
                buffer.read(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        }
    )
    
    assert_valid_prediction_response(response)
    

def test_predict_endpoint_rejects_unsupported_file_format():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not a valid input file", "text/plain")}
    )

    assert response.status_code == 400
    assert "CSV, JSON or Excel .xlsx" in response.json()["detail"]
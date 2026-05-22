from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_endpoint_with_valid_csv():
    csv_content = (
        "a_comp_1,a_frac_1,a_comp_2,a_frac_2,a_comp_3,a_frac_3,"
        "c_comp_1,c_frac_1,c_comp_2,c_frac_2,c_comp_3,c_frac_3,"
        "f_comp_1,f_frac_1,f_comp_2,f_frac_2,f_comp_3,f_frac_3,"
        "f_comp_4,f_frac_4,f_comp_5,f_frac_5,n_300,T,p_n,ZT\n"
        "Sb,12,0,0,0,0,Co,4,0,0,0,0,Yb,0.1,Nd,0.2,0,0,0,0,0,0,1e20,700,1,0\n"
    )

    response = client.post(
        "/predict",
        files={"file": ("test.csv", csv_content.encode("utf-8"), "text/csv")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert isinstance(data["predictions"][0], float)
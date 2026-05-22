import pandas as pd
import pytest
from pathlib import Path

from load_model_predictions_server import main as run_predictions

def test_non_numeric_fraction_is_rejected(tmp_path):
    bad_csv = tmp_path / "bad_input.csv"

    df = pd.DataFrame([{
        "a_comp_1": "Sb", "a_frac_1": 12,
        "a_comp_2": 0, "a_frac_2": 0,
        "a_comp_3": 0, "a_frac_3": 0,
        "c_comp_1": "Co", "c_frac_1": 4,
        "c_comp_2": 0, "c_frac_2": 0,
        "c_comp_3": 0, "c_frac_3": 0,
        "f_comp_1": "Yb", "f_frac_1": "Nd",
        "f_comp_2": 0, "f_frac_2": 0,
        "f_comp_3": 0, "f_frac_3": 0,
        "f_comp_4": 0, "f_frac_4": 0,
        "f_comp_5": 0, "f_frac_5": 0,
        "n_300": 1e20,
        "T": 700,
        "p_n": 1,
        "ZT": 0
    }])

    df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="non-numeric values"):
        run_predictions(input_csv=str(bad_csv))
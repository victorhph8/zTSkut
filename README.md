<p align="center">
  <img src="assets/ztskut_logo.png" alt="zTSkut logo" width="220">

### A web application for deep learning prediction of thermoelectric performance in skutterudites
</p>

[![Web App](https://img.shields.io/badge/web%20app-live-2ea44f)](https://ztskut.onrender.com/)
[![Paper](https://img.shields.io/badge/paper-JMCA-blue)](https://pubs.rsc.org/en/content/articlelanding/2026/ta/d5ta08841k)
[![DOI](https://img.shields.io/badge/DOI-10.1039%2FD5TA08841K-orange)](https://doi.org/10.1039/D5TA08841K)
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

</div>

zTSkut is a web application for predicting the thermoelectric figure of merit (zT) of skutterudite-based compositions using the neural-network model reported in our Journal of Materials Chemistry A paper.

The app supports both **single-composition prediction** through a browser form and **batch prediction** from CSV files, making the model accessible for rapid screening of candidate skutterudite materials.

**Links:** [Web app](https://ztskut.onrender.com/) | [Paper](https://pubs.rsc.org/en/content/articlelanding/2026/ta/d5ta08841k) | [Citation](#citation)

---

## What zTSkut does

zTSkut takes a skutterudite composition as input and predicts its zT using a trained Keras neural-network model.

The prediction workflow is:

1. Read the user input composition.
2. Generate the 37 compositional descriptors used by the trained model.
3. Order the descriptors according to `feature_columns_ZT.txt`.
4. Apply the saved training scaler from `scaler_ZT.joblib`.
5. Predict zT using the trained model `model_keras_skutt.h5`.

The scaler is only used in transform mode. No fitting or re-normalisation is performed during prediction.

---

## Repository structure

The current deployment uses a simple Render-compatible structure:

```text
zTSkut/
  main.py
  load_model_predictions_server.py
  requirements.txt
  model_keras_skutt.h5
  scaler_ZT.joblib
  feature_columns_ZT.txt

  static/
    index.html
    sample_template.csv
    skutterudite_model.bib
    skutterudite_model.ris
```

Additional repository files:

```text
examples/
  example_single.csv
  example_batch.csv

tests/
  test_app_health.py
  test_input_validation.py
  test_prediction_endpoint.py

paper/
  paper.md
  paper.bib
```

---

## Input format

Input compositions are provided using separate columns for anion, cation and filler sites.

The required columns are:

```text
a_comp_1,a_comp_2,a_comp_3,
a_frac_1,a_frac_2,a_frac_3,
c_comp_1,c_comp_2,c_comp_3,
c_frac_1,c_frac_2,c_frac_3,
f_comp_1,f_comp_2,f_comp_3,f_comp_4,f_comp_5,
f_frac_1,f_frac_2,f_frac_3,f_frac_4,f_frac_5,
n_300,T,p_n,ZT
```

where:

- `a_comp_*` and `a_frac_*` define the anion species and fractions.
- `c_comp_*` and `c_frac_*` define the cation species and fractions.
- `f_comp_*` and `f_frac_*` define the filler species and fractions.
- `n_300` is the carrier concentration at 300 K.
- `T` is the temperature in K.
- `p_n` defines the carrier type:
  - `0` = p-type
  - `1` = n-type
- `ZT` is a dummy column kept for input-format compatibility and is ignored during prediction.

Unused composition and fraction fields can be set to `0`. In the web form, empty optional fields are automatically treated as `0`.

A batch template is provided in:

```text
static/sample_template.csv
```

---

## Web app usage

### Single prediction

1. Open the web app.
2. Fill in the composition fields.
3. Enter `n_300`, `T`, and `p_n`.
4. Click **Predict zT**.

The predicted zT value will appear directly on the page.

### Batch prediction

1. Download the CSV template from the web app.
2. Add one candidate system per row.
3. Upload the completed CSV file.
4. The app returns one predicted zT value per system.

---

## Local installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/victorhph8/zTSkut.git
cd zTSkut
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment using:

```bash
.venv\Scripts\activate
```

---

## Running the web app locally

The app is served with FastAPI and Uvicorn:

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

This runs the same FastAPI app locally and does not require Render.

For deployment on Render, the start command is:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Running tests

Basic tests are provided to check that the web app loads correctly, the prediction endpoint works with a valid CSV file and invalid numerical inputs are handled properly.

From the repository root, run:

```bash
python -m pytest tests/
```

A successful run should report all tests as passed.

---

## Required model files

The following files must be present in the repository root for predictions to work:

```text
model_keras_skutt.h5
scaler_ZT.joblib
feature_columns_ZT.txt
```

These files define the trained model, the saved feature scaler and the descriptor order expected by the neural network.

---

## Running predictions directly from Python

Example input files are provided in the `examples/` folder:

```text
examples/example_single.csv
examples/example_batch.csv
```

These files follow the same format as `static/sample_template.csv`.

Predictions can be generated locally from a Python session without starting the web app:

```python
from load_model_predictions_server import main as run_predictions

predictions = run_predictions(input_csv="examples/example_batch.csv")
print(predictions)
```

For a single-system example:

```python
from load_model_predictions_server import main as run_predictions

predictions = run_predictions(input_csv="examples/example_single.csv")
print(predictions)
```

The same backend function is used by the web app, so this local workflow reproduces the deployed prediction pipeline.

---

## Error handling

The app checks that numeric fields contain valid numerical values. These include:

```text
a_frac_1, a_frac_2, a_frac_3,
c_frac_1, c_frac_2, c_frac_3,
f_frac_1, f_frac_2, f_frac_3, f_frac_4, f_frac_5,
n_300, T, p_n, ZT
```

If a user enters an element symbol in a numeric field, the app returns a clear error message instead of failing silently.

---

## Citation

If you use this model or web app, please cite the associated paper:

```bibtex
@article{Posligua2026SkutteruditesZT,
  title   = {Deep learning framework for accurate prediction and high-throughput search of the thermoelectric figure of merit in skutterudites},
  author  = {Posligua, Victor and Landivar, Karina and Remesal, Elena R. and Rogl, Gerda and Rogl, Peter F. and Fdez Sanz, Javier and Prado-Gonjal, Jes{\'u}s and M{\'a}rquez, Antonio M. and Plata, Jose J.},
  journal = {Journal of Materials Chemistry A},
  year    = {2026},
  doi     = {10.1039/D5TA08841K},
  url     = {https://doi.org/10.1039/D5TA08841K}
}
```

Citation files are also provided in:

```text
static/skutterudite_model.bib
static/skutterudite_model.ris
```

---

## Scope and limitations

zTSkut is designed for skutterudite-based thermoelectric compositions within the chemical and descriptor space explored in the associated model-development study.

Predictions should be interpreted with caution for:

- Compositions far outside the chemistry represented in the training data
- Unusual carrier concentrations or temperatures outside the studied range
- Systems affected by secondary phases, phase segregation or unusual processing routes
- Very high-zT systems beyond the range well represented in the training dataset

The tool is intended to guide candidate selection and screening, not to replace experimental validation.

---

## License

License information will be added before public release.

---

## Acknowledgements

The underlying model was developed as part of the skutterudite thermoelectrics study reported in the associated Journal of Materials Chemistry A paper.

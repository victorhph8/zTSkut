from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from load_model_predictions_server import main as run_predictions
import os
import tempfile
from pathlib import Path
import io
import json
import pandas as pd
import traceback
import base64

app = FastAPI()

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML on root
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

def read_upload_as_dataframe(contents: bytes, suffix: str) -> pd.DataFrame:
    """Read an uploaded CSV, JSON, or XLSX file into a pandas DataFrame."""
    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(contents))

    elif suffix == ".json":
        data = json.loads(contents.decode("utf-8"))

        # Accept either a direct list of records, a single record, or {"systems": [...]} - This worked last time, hope it still does :)
        if isinstance(data, dict) and "systems" in data:
            data = data["systems"]
        elif isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            raise ValueError(
                "JSON input must be a list of records, a single record, or an object with a 'systems' list."
            )

        df = pd.DataFrame(data)

    elif suffix == ".xlsx":
        df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")

    else:
        raise ValueError("Unsupported input format. Please use CSV, JSON, or Excel .xlsx.")

    if df.empty:
        raise ValueError("Uploaded file does not contain any systems to predict.")

    return df

def dataframe_to_temp_csv(df: pd.DataFrame) -> str:
    """Write a dataframe to a temporary CSV file for the existing prediction pipeline"""
    fd, tmp_csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)

    try:
        df.to_csv(tmp_csv_path, index=False)
        return tmp_csv_path

    except Exception:
        if os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)
        raise
    
def build_download_payload(output_df: pd.DataFrame) -> dict:
    """Create downloadable CSV, JSON and XLSX versions of the prediction table"""
    csv_text = output_df.to_csv(index=False)
    json_text = output_df.to_json(orient="records", indent=2)

    xlsx_buffer = io.BytesIO()
    output_df.to_excel(xlsx_buffer, index=False, engine="openpyxl")
    xlsx_base64 = base64.b64encode(xlsx_buffer.getvalue()).decode("utf-8")

    return {
        "csv": csv_text,
        "json": json_text,
        "xlsx_base64": xlsx_base64
    }

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    
    allowed_suffixes = {".csv", ".json", ".xlsx"}
    
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV, JSON or Excel .xlsx file"
        )
        
    tmp_csv_path = None

    try:
        contents = await file.read()
        
        input_df = read_upload_as_dataframe(contents, suffix)
        tmp_csv_path = dataframe_to_temp_csv(input_df)

        preds = run_predictions(input_csv=tmp_csv_path)
        
        if len(preds) != len(input_df):
            raise RuntimeError(
                "The number of predictions does not match the number of uploaded systems."
            )

        output_df = input_df.copy()
        output_df["predicted_ZT"] = preds

        results = json.loads(output_df.to_json(orient="records"))
        downloads = build_download_payload(output_df)
        
        return {"predictions": preds, "n_predictions": len(preds), "results": results, "downloads": downloads}
    
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if tmp_csv_path and os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)

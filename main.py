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

app = FastAPI()

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML on root
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

def save_upload_as_csv(contents: bytes, suffix: str) -> str:
    """Convert an uploaded CSV, JSON, or XLSX file to a temporary CSV file"""
    fd, tmp_csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)

    try:
        if suffix == ".csv":
            with open(tmp_csv_path, "wb") as tmp_csv:
                tmp_csv.write(contents)

        elif suffix == ".json":
            data = json.loads(contents.decode("utf-8"))

            # Accept either a direct list of records or {"systems": [...]} - This should work now :/
            if isinstance(data, dict) and "systems" in data:
                data = data["systems"]
            elif isinstance(data, dict):
                data = [data]

            if not isinstance(data, list):
                raise ValueError(
                    "JSON input must be a list of records, a single record or an object with a 'systems' list"
                )

            df = pd.DataFrame(data)
            df.to_csv(tmp_csv_path, index=False)

        elif suffix == ".xlsx":
            df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
            df.to_csv(tmp_csv_path, index=False)

        else:
            raise ValueError("Unsupported input format. Please use CSV, JSON or Excel .xlsx")

        return tmp_csv_path

    except Exception:
        if os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)
        raise

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
        tmp_csv_path = save_upload_as_csv(contents, suffix)

        preds = run_predictions(input_csv=tmp_csv_path)
        return {"predictions": preds, "n_predictions": len(preds)}
    
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if tmp_csv_path and os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)

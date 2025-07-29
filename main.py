from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from load_model_predictions_server import main
import pandas as pd
import uuid
import os

app = FastAPI()

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML on root
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    temp_name = f"samples_file_{uuid.uuid4().hex}.csv" # Avoid conflicts when more than one user is doing predictions
    with open(temp_name, "wb") as f:
    #with open("samples_file.csv", "wb") as f:
        f.write(contents)
        
    # Enforce 10-row limit
    df = pd.read_csv(temp_name)
    if len(df) > 10:
        return {"Error": "Maximum 10 systems allowed in CSV upload"}
    
    preds = main("norm_file.npy", input_csv=temp_name)
    os.remove(temp_name)
    return {"predictions": preds}
    #return preds[0]


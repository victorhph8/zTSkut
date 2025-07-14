from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from load_model_predictions_v7 import main

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
    with open("samples_file.csv", "wb") as f:
        f.write(contents)
    preds = main("norm_file.npy")
    #return {"predictions": preds}
    return preds[0]


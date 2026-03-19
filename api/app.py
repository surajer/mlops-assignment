from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
import numpy as np

app = FastAPI(title="ML Model API")

# ✅ Force MLflow to use correct folder
mlflow.set_tracking_uri("file:./mlruns")

MODEL_URI = "mlruns/250287894545649833/d0d3300630bd46c6976d92d39f2d0fd2/artifacts/model"

# Load model
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    input_data = np.array(data["input"])
    prediction = model.predict(input_data).tolist()
    return {"prediction": prediction}
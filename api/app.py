from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import mlflow
import mlflow.pyfunc
import numpy as np
import time   

from api.drift import detect_drift

app = FastAPI(title="ML Model API")

# MLflow to use correct folder
mlflow.set_tracking_uri("file:./mlruns")

MODEL_URI = "mlruns/250287894545649833/d0d3300630bd46c6976d92d39f2d0fd2/artifacts/model"

# Load model
model = mlflow.pyfunc.load_model(MODEL_URI)

# Metrics variables
request_count = 0
total_latency = 0.0
drift_count = 0   


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: dict):
    global request_count, total_latency, drift_count

    start_time = time.time()               

    input_data = np.array(data["input"])

    # Drift detection
    drift_flag = detect_drift(input_data)
    if drift_flag:
        drift_count += 1

    prediction = model.predict(input_data).tolist()

    latency = time.time() - start_time    

    # Update metrics
    request_count += 1
    total_latency += latency

    return {
        "prediction": prediction,
        "latency": round(latency, 4),
        "drift_detected": drift_flag
    }

# Prometheus-compatible metrics
@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    avg_latency = total_latency / request_count if request_count > 0 else 0

    return (
        f"total_requests {request_count}\n"
        f"average_latency {avg_latency}\n"
        f"drift_count {drift_count}\n"
        f'model_version{{version="v1"}} 1\n'
    )
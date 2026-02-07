from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load("model/churn_model.pkl")

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}

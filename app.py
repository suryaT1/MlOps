from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import List

app = FastAPI()
model = joblib.load("model/iris_model.pkl")

class IrisInput(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(data: IrisInput):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}

@app.post("/hello")
def hello():
    return {" Hello world !"}

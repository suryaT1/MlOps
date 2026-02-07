from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model once at startup
MODEL_PATH = "cat_dog_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI(title="Cat vs Dog Classifier")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    label = "dog" if prediction > 0.5 else "cat"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }

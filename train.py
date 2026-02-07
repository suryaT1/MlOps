from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("model", exist_ok=True)

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model/iris_model.pkl")
print("Model trained and saved")

# 🌸 Iris Classification – MLOps Project

This project demonstrates an **end-to-end MLOps workflow** using the classic **Iris dataset**, covering model training, API-based inference, containerization, and CI/CD automation.

---

## 📌 Problem Statement

Predict the **Iris flower species** based on four numerical features:
- Sepal length
- Sepal width
- Petal length
- Petal width

### Target Classes
- `0` → Setosa  
- `1` → Versicolor  
- `2` → Virginica  

---

## 🧠 Machine Learning Model

- **Algorithm:** RandomForestClassifier  
- **Library:** Scikit-learn  
- **Dataset:** Built-in Iris dataset (`sklearn.datasets.load_iris`)  

### Training Code
```python
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


```

### Prediction

POST /predict


# sample input

{
  "features": [5.1, 3.5, 1.4, 0.2]
}


# sample output

{
  "prediction": 0
}

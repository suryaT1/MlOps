import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/telecom_churn.csv")

# Fix target safely
if df["Churn"].dtype == object:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Separate features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Defensive check (VERY important)
assert y.isna().sum() == 0, "Target y contains NaN values"

# One-hot encode (if needed)
X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "model/churn_model.pkl")
print("Model saved to model/churn_model.pkl")

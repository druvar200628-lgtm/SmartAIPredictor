"""
train_model.py
--------------
Trains TWO models (Logistic Regression + Random Forest) and saves them to model/churn_model.pkl
Run:  python train_model.py
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")

# Columns used as model INPUT (features)
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "Partner", "Contract", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL


def clean_data(df):
    """Fix common data problems (the real Telco file has blank TotalCharges for new customers)."""
    df = df.copy()
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
    # SeniorCitizen can be 0/1 or Yes/No in different files -> make it 0/1
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({"Yes": 1, "No": 0}).astype(int)
    return df


def build_pipeline(model):
    """Preprocessing + model in ONE object, so new data is transformed exactly like training data."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),                           # scale numbers
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)  # text -> 0/1 columns
    ])
    return Pipeline([("prep", prep), ("model", model)])


def train():
    df = clean_data(pd.read_csv(DATA_PATH))
    X = df[FEATURES]
    y = (df["Churn"] == "Yes").astype(int)  # Yes -> 1, No -> 0

    # 80% to learn from, 20% hidden to test honestly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=1000)),
        "Random Forest": build_pipeline(RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)),
    }

    metrics = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, pred) * 100, 1),
            "precision": round(precision_score(y_test, pred, zero_division=0) * 100, 1),
            "recall": round(recall_score(y_test, pred) * 100, 1),
            "f1": round(f1_score(y_test, pred) * 100, 1),
        }
        print(f"{name:20s}", metrics[name])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"models": models, "metrics": metrics}, MODEL_PATH)
    print("Saved ->", MODEL_PATH)


if __name__ == "__main__":
    train()

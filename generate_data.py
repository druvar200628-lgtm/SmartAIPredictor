"""
generate_data.py
----------------
Creates a realistic sample Telco churn dataset (7043 rows) -> data/Telco.csv
Use this ONLY if you don't have the real Kaggle "Telco Customer Churn" file.
If you download the real file, just save it as data/Telco.csv (same column names work).
"""
import os
import string
import numpy as np
import pandas as pd


def create_dataset(path, n=7043, seed=42):
    rng = np.random.default_rng(seed)

    # --- Customer profile columns ---
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    gender = rng.choice(["Male", "Female"], n)
    senior = (rng.random(n) < 0.16).astype(int)
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21])

    # Month-to-month customers tend to have shorter tenure
    tenure = np.where(contract == "Month-to-month",
                      np.clip(rng.gamma(1.4, 13, n), 1, 72),
                      np.where(contract == "One year", rng.integers(6, 73, n), rng.integers(12, 73, n)))
    tenure = tenure.astype(int)

    monthly = np.clip(rng.normal(65, 28, n), 18.25, 118.75).round(2)
    total = (tenure * monthly * rng.uniform(0.95, 1.05, n)).round(2)

    # --- Churn is generated from a logistic formula so the ML models have real patterns to learn ---
    contract_effect = np.select([contract == "Month-to-month", contract == "One year"], [1.0, -0.4], -1.5)
    z = (-1.1 + contract_effect + 0.022 * (monthly - 65) - 0.03 * tenure
         + 0.45 * senior + 0.55 * (payment == "Electronic check") - 0.2 * (partner == "Yes"))
    prob = 1 / (1 + np.exp(-z))
    churn = np.where(rng.random(n) < prob, "Yes", "No")

    ids = [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list(string.ascii_uppercase), 5))}" for _ in range(n)]

    df = pd.DataFrame({
        "customerID": ids, "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
        "Contract": contract, "PaymentMethod": payment, "Churn": churn})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Telco.csv")
    d = create_dataset(out)
    print("Created", out, d.shape, "| churn rate:", round((d.Churn == "Yes").mean() * 100, 1), "%")

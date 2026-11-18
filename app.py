"""
SmartPredict AI - Customer Retention & Risk Intelligence Platform
Main Flask application.  Run with:  python app.py   ->  http://127.0.0.1:5000
"""
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, flash, redirect, url_for

from train_model import FEATURES, DATA_PATH, MODEL_PATH, BASE_DIR, clean_data, train

app = Flask(__name__)
app.secret_key = "smartpredict-ai-secret"   # needed for flash() messages

OUTPUT_PATH = os.path.join(BASE_DIR, "data", "bulk_predictions.csv")  # result of CSV upload

# ----------------------------------------------------------------------------
# 1. STARTUP: make sure dataset + trained model exist (so the app never crashes)
# ----------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    from generate_data import create_dataset
    create_dataset(DATA_PATH)
if not os.path.exists(MODEL_PATH):
    train()

bundle = joblib.load(MODEL_PATH)       # {"models": {...}, "metrics": {...}}
MODELS = bundle["models"]
METRICS = bundle["metrics"]
DEFAULT_MODEL = "Random Forest"        # model used for Dashboard / Customers pages

REQUIRED_COLUMNS = FEATURES            # columns a user's CSV must contain


# ----------------------------------------------------------------------------
# 2. RISK INTELLIGENCE SCORING  (Module 3)
#    Risk Score (0-100) = Churn Probability*0.6  +  Charges factor (0-20)  +  Tenure factor (0-20)
# ----------------------------------------------------------------------------
def risk_parts(prob, monthly, tenure):
    prob_part = prob * 60                                            # 0 - 60 points
    charges_part = min(max((monthly - 18) / (120 - 18), 0), 1) * 20  # higher bill -> more risk (0-20)
    tenure_part = (1 - min(tenure, 72) / 72) * 20                    # newer customer -> more risk (0-20)
    return round(prob_part, 1), round(charges_part, 1), round(tenure_part, 1)


def risk_score(prob, monthly, tenure):
    return round(sum(risk_parts(prob, monthly, tenure)), 1)


def risk_level(score):
    """0-40 Low, 41-70 Medium, 71-100 High"""
    if score <= 40:
        return "Low"
    if score <= 70:
        return "Medium"
    return "High"


# ----------------------------------------------------------------------------
# 3. RETENTION RECOMMENDATION ENGINE  (Module 5) - simple IF rules, no complex AI
# ----------------------------------------------------------------------------
def recommend(row):
    level = row["Risk_Level"]
    if level == "Low":
        return ["No action needed - customer is stable"]
    if level == "Medium":
        return ["Monitor account and send satisfaction survey"]
    tips = []                                           # High risk -> apply the rules
    if row["Contract"] == "Month-to-month":
        tips.append("Offer 1-year contract with 20% discount")
    if row["MonthlyCharges"] > 80:
        tips.append("Offer loyalty discount")
    if row["tenure"] < 12:
        tips.append("Schedule an onboarding call")
    if not tips:
        tips.append("Assign account manager for a personal follow-up call")
    return tips


# ----------------------------------------------------------------------------
# 4. CORE FUNCTION: take a DataFrame of customers -> add prediction + risk columns
# ----------------------------------------------------------------------------
def score_dataframe(df, model_name=DEFAULT_MODEL):
    model = MODELS[model_name]
    probs = model.predict_proba(df[FEATURES])[:, 1]          # probability of churn (0 to 1)
    out = df.copy()
    out["Churn_Probability"] = (probs * 100).round(1)
    out["Churn_Prediction"] = np.where(probs >= 0.5, "Will Churn", "Will Not Churn")
    out["Risk_Score"] = [risk_score(p, m, t) for p, m, t in zip(probs, out["MonthlyCharges"], out["tenure"])]
    out["Risk_Level"] = out["Risk_Score"].apply(risk_level)
    out["Recommendation"] = out.apply(lambda r: " | ".join(recommend(r)), axis=1)
    return out


# Score the whole dataset once at startup (CSV is our database)
ALL_CUSTOMERS = score_dataframe(clean_data(pd.read_csv(DATA_PATH)))


# ----------------------------------------------------------------------------
# 5. ROUTES (pages)
# ----------------------------------------------------------------------------
@app.route("/")
def dashboard():
    df = ALL_CUSTOMERS
    churned = df["Churn"] == "Yes"

    kpis = {
        "total": f"{len(df):,}",
        "churn_rate": round(churned.mean() * 100, 1),
        "avg_risk": round(df["Risk_Score"].mean(), 1),
        "high_risk": f"{(df['Risk_Level'] == 'High').sum():,}",
    }

    # Chart 1: churn % by contract type (bar)
    by_contract = (churned.groupby(df["Contract"]).mean() * 100).round(1)

    # Chart 2: churn % by tenure group (line)
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    tenure_group = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    by_tenure = (churned.groupby(tenure_group, observed=False).mean() * 100).round(1)

    # Chart 3: how many customers in each risk level (doughnut)
    risk_counts = df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)

    charts = {
        "contract": {"labels": list(by_contract.index), "values": by_contract.tolist()},
        "tenure": {"labels": labels, "values": by_tenure.tolist()},
        "risk": {"labels": list(risk_counts.index), "values": risk_counts.tolist()}   # .tolist() -> plain Python numbers (JSON-safe),
    }
    return render_template("index.html", kpis=kpis, charts=charts)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result, form = None, {}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            tenure = int(form["tenure"])
            monthly = float(form["MonthlyCharges"])
            total = float(form["TotalCharges"]) if form.get("TotalCharges") else tenure * monthly
            row = pd.DataFrame([{
                "gender": form["gender"], "SeniorCitizen": int(form["SeniorCitizen"]),
                "Partner": form["Partner"], "tenure": tenure, "MonthlyCharges": monthly,
                "TotalCharges": total, "Contract": form["Contract"], "PaymentMethod": form["PaymentMethod"],
            }])
            model_name = form.get("model", DEFAULT_MODEL)
            scored = score_dataframe(row, model_name).iloc[0]
            prob = scored["Churn_Probability"] / 100
            p, c, t = risk_parts(prob, monthly, tenure)
            result = {
                "model": model_name,
                "prediction": scored["Churn_Prediction"],
                "probability": scored["Churn_Probability"],
                "score": scored["Risk_Score"],
                "level": scored["Risk_Level"],
                "breakdown": {"prob": p, "charges": c, "tenure": t},
                "tips": recommend(scored),
            }
        except (ValueError, KeyError):
            flash("Please enter valid numbers for tenure and monthly charges.", "danger")
    return render_template("predict.html", result=result, form=form, metrics=METRICS, model_names=list(MODELS))


def load_table(source):
    """Return the table to show: all customers, or the last uploaded+predicted file."""
    if source == "upload" and os.path.exists(OUTPUT_PATH):
        return pd.read_csv(OUTPUT_PATH)
    return ALL_CUSTOMERS


@app.route("/customers")
def customers():
    source = request.args.get("source", "all")
    q = request.args.get("q", "").strip()
    risk = request.args.get("risk", "All")
    page = max(int(request.args.get("page", 1) or 1), 1)

    df = load_table(source)
    if q:                                                       # search by customer ID
        df = df[df["customerID"].astype(str).str.contains(q, case=False, na=False)]
    if risk in ("Low", "Medium", "High"):                       # filter by risk level
        df = df[df["Risk_Level"] == risk]

    per_page = 20
    pages = max((len(df) + per_page - 1) // per_page, 1)
    page = min(page, pages)
    rows = df.iloc[(page - 1) * per_page: page * per_page].to_dict("records")
    return render_template("customers.html", rows=rows, total=len(df), page=page, pages=pages,
                           q=q, risk=risk, source=source, has_upload=os.path.exists(OUTPUT_PATH))


@app.route("/upload", methods=["POST"])
def upload():
    """Bulk prediction: upload a CSV -> predict every row -> show results."""
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".csv"):
        flash("Please choose a .csv file.", "danger")
        return redirect(url_for("customers"))
    try:
        new_df = pd.read_csv(file)
    except Exception:
        flash("Could not read that file. Is it a valid CSV?", "danger")
        return redirect(url_for("customers"))

    missing = [c for c in REQUIRED_COLUMNS if c not in new_df.columns and c != "TotalCharges"]
    if missing:
        flash("Your CSV is missing columns: " + ", ".join(missing), "danger")
        return redirect(url_for("customers"))
    if "TotalCharges" not in new_df.columns:
        new_df["TotalCharges"] = np.nan                          # will be auto-filled in clean_data
    if "customerID" not in new_df.columns:
        new_df.insert(0, "customerID", [f"NEW-{i + 1:04d}" for i in range(len(new_df))])

    scored = score_dataframe(clean_data(new_df))
    scored.to_csv(OUTPUT_PATH, index=False)
    flash(f"Predicted {len(scored)} customers successfully.", "success")
    return redirect(url_for("customers", source="upload"))


@app.route("/download")
def download():
    source = request.args.get("source", "all")
    path = OUTPUT_PATH if (source == "upload" and os.path.exists(OUTPUT_PATH)) else None
    if path is None:
        path = os.path.join(BASE_DIR, "data", "all_customers_scored.csv")
        ALL_CUSTOMERS.to_csv(path, index=False)
    return send_file(path, as_attachment=True, download_name="smartpredict_results.csv")


@app.route("/analytics")
def analytics():
    df = ALL_CUSTOMERS
    churned = df["Churn"] == "Yes"
    avg_by_contract = df.groupby("Contract")["Risk_Score"].mean().round(1)
    churn_by_payment = (churned.groupby(df["PaymentMethod"]).mean() * 100).round(1)
    high = df[df["Risk_Level"] == "High"]

    charts = {
        "contract": {"labels": list(avg_by_contract.index), "values": avg_by_contract.tolist()},
        "payment": {"labels": list(churn_by_payment.index), "values": churn_by_payment.tolist()},
        "models": {"labels": ["Accuracy", "Precision", "Recall"],
                   "sets": {n: [m["accuracy"], m["precision"], m["recall"]] for n, m in METRICS.items()}},
    }
    # how many HIGH-risk customers each retention rule applies to
    actions = {
        "Offer 1-year contract (20% off)": int((high["Contract"] == "Month-to-month").sum()),
        "Offer loyalty discount": int((high["MonthlyCharges"] > 80).sum()),
        "Onboarding call": int((high["tenure"] < 12).sum()),
    }
    top10 = high.sort_values("Risk_Score", ascending=False).head(10).to_dict("records")
    levels = df["Risk_Level"].value_counts().to_dict()
    return render_template("analytics.html", charts=charts, actions=actions, top10=top10,
                           levels={k: int(levels.get(k, 0)) for k in ["Low", "Medium", "High"]})


if __name__ == "__main__":
    app.run(debug=True)

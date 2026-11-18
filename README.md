# SmartPredict AI - Customer Retention & Risk Intelligence Platform

A Flask web app that predicts which customers will leave (churn), gives each one a 0-100 **Risk Score**,
and suggests how to keep them. Works for Telecom / Banks / SaaS (demo dataset: Telco Customer Churn).

## Tech stack
Python, Flask | HTML, CSS, Bootstrap 5, Chart.js | scikit-learn, Pandas, NumPy | CSV file as database

## Folder structure
```
SmartPredictAI/
├── app.py                 # Flask app: all pages + risk score + recommendation rules
├── train_model.py         # trains Logistic Regression + Random Forest -> model/churn_model.pkl
├── generate_data.py       # makes a sample Telco.csv (7043 rows) if you don't have the real one
├── requirements.txt
├── README.md
├── model/churn_model.pkl  # saved models + accuracy/precision/recall
├── data/Telco.csv         # dataset (CSV = our database)
├── data/sample_upload.csv # example file for the bulk-prediction demo
├── static/css/style.css
└── templates/  base.html, index.html, predict.html, customers.html, analytics.html
```

## Steps to run
```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows      |   source venv/bin/activate   (Mac/Linux)

# 2. install libraries
pip install -r requirements.txt

# 3. dataset: use the real Kaggle "Telco Customer Churn" CSV saved as data/Telco.csv,
#    OR generate the sample one:
python generate_data.py

# 4. train the models  (creates model/churn_model.pkl and prints accuracy, precision, recall)
python train_model.py

# 5. start the app
python app.py
```
Open **http://127.0.0.1:5000** in your browser. (Internet is needed for Bootstrap/Chart.js CDN files.)

> If you skip steps 3-4, `app.py` creates the dataset and trains the model automatically on first run.

## Risk Score formula
```
Risk Score = (Churn Probability x 100 x 0.6)        -> 0 to 60
           + Monthly Charges factor                  -> 0 to 20  (bigger bill = more risk)
           + Short Tenure factor                     -> 0 to 20  (newer customer = more risk)
```
0-40 = Low (green) | 41-70 = Medium (amber) | 71-100 = High (red)

## Retention rules (High-risk customers only)
| Condition | Recommendation |
|---|---|
| Contract = Month-to-month | Offer 1-year contract with 20% discount |
| MonthlyCharges > 80 | Offer loyalty discount |
| Tenure < 12 months | Schedule an onboarding call |

## Bulk prediction CSV format
Required columns: `gender, SeniorCitizen, Partner, tenure, MonthlyCharges, Contract, PaymentMethod`
Optional: `customerID, TotalCharges`. Try `data/sample_upload.csv`.

## Viva - quick answers
- **What is churn?** A customer leaving the company.
- **Why two models?** Logistic Regression is simple and explainable; Random Forest (many decision trees voting) captures complex patterns. We compare them.
- **Accuracy / Precision / Recall?** Accuracy = overall correct predictions. Precision = of customers we flagged as churners, how many really were. Recall = of real churners, how many we caught. For retention, recall matters because missing a churner costs money.
- **Why a Pipeline?** It bundles scaling + one-hot encoding + model so new data is processed exactly like training data.
- **Why train/test split?** To test the model on customers it has never seen (80% train, 20% test).
- **Why Risk Score instead of just probability?** Business risk also depends on how much the customer pays and how new they are.
- **Note:** the sample dataset is synthetic; the real Kaggle file gives similar results (~80% accuracy).

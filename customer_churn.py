# -*- coding: utf-8 -*-
"""
Customer Churn Analysis
Telco Customer Churn Dataset
"""

import json
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ──────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────
df = pd.read_csv("Telco-Customer-Churn.csv")
print(df.head())
print(df.info())
print(df.describe())
print("Unique customers:", df["customerID"].nunique())

# ──────────────────────────────────────────
# 2. BASIC CLEANING
# ──────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
df = df.reset_index(drop=True)

print("Nulls after cleaning:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# ──────────────────────────────────────────
# 3. EDA — DO THIS BEFORE ENCODING
# ──────────────────────────────────────────
print("\nChurn counts:\n", df["Churn"].value_counts())
print("\nChurn rate (%):\n", df["Churn"].value_counts(normalize=True) * 100)
print("\nOverall churn rate: {:.2f}%".format((df["Churn"] == "Yes").mean() * 100))

print("\nChurn rate by Contract:")
print(df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean()))

print("\nChurn counts by Contract:")
print(df.groupby("Contract")["Churn"].value_counts())

df["Churn_binary"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\nAvg tenure by Churn:")
print(df.groupby("Churn")["tenure"].mean())

print("\nAvg MonthlyCharges by Churn:")
print(df.groupby("Churn")["MonthlyCharges"].mean())

print("\nChurn rate by InternetService:")
print(df.groupby("InternetService")["Churn_binary"].mean())

print("\nChurn rate by gender:")
print(df.groupby("gender")["Churn_binary"].mean())

print("\nChurn rate by SeniorCitizen:")
print(df.groupby("SeniorCitizen")["Churn_binary"].mean())

print("\nChurn rate by Dependents:")
print(df.groupby("Dependents")["Churn_binary"].mean())

print("\nChurn rate by PaperlessBilling:")
print(df.groupby("PaperlessBilling")["Churn_binary"].mean())

print("\nChurn rate by PaymentMethod:")
print(df.groupby("PaymentMethod")["Churn_binary"].mean())

# ──────────────────────────────────────────
# 4. VISUALIZATIONS
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.countplot(x="Churn", data=df, ax=axes[0])
axes[0].set_title("Churn Distribution")

sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=axes[1])
axes[1].set_title("Monthly Charges vs Churn")

sns.countplot(x="Contract", hue="Churn", data=df, ax=axes[2])
axes[2].set_title("Contract Type vs Churn")
axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("churn_plots.png", dpi=150)
plt.show()

# ──────────────────────────────────────────
# 5. ENCODING — AFTER EDA
# ──────────────────────────────────────────
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["gender"]       = df["gender"].map({"Female": 0, "Male": 1})
df["Churn_binary"] = df["Churn"].map({"Yes": 1, "No": 0})

df = pd.get_dummies(df, columns=[
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "PaymentMethod"
], drop_first=True)

bool_cols = df.select_dtypes("bool").columns
df[bool_cols] = df[bool_cols].astype(int)

df = df.drop(columns=["Churn"])
df = df.rename(columns={"Churn_binary": "Churn"})

print("\nFinal dataframe shape:", df.shape)
print(df.head())
print(df.info())

# ──────────────────────────────────────────
# 6. PREPARE FOR MODELING
# ──────────────────────────────────────────
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

print("\nClass distribution:\n", y.value_counts())
print("Churn rate: {:.2f}%".format(y.mean() * 100))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler   = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

print("\nTraining set size:", X_train.shape)
print("Test set size:    ", X_test.shape)

# ──────────────────────────────────────────
# 7. TRAIN MODEL
# ──────────────────────────────────────────
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ══════════════════════════════════════════════════════════════
# 8. LLM INTEGRATION (Ollama gemma3:1b)
# Requirements: pip install requests
#               ollama pull gemma3:1b
#               ollama serve  <- run in a separate terminal
# ══════════════════════════════════════════════════════════════

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

def ask_llm(prompt: str, max_tokens: int = 400) -> str:
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"num_predict": max_tokens, "temperature": 0.7},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Ollama not running. Start it with: ollama serve"
    except Exception as e:
        return f"LLM error: {e}"


# ── Build profile from highest-risk customer ─────────────────
test_df         = X_test.copy()
test_df["prob"] = y_proba                  # fix: assign y_proba before idxmax
high_risk_idx   = test_df["prob"].idxmax()
high_risk_row   = test_df.loc[high_risk_idx]
high_risk_prob  = high_risk_row["prob"]

profile = {
    "churn_probability": f"{high_risk_prob*100:.1f}%",
    "tenure_months":     int(round(high_risk_row["tenure"]     * scaler.scale_[0] + scaler.mean_[0])),
    "monthly_charges":   round(high_risk_row["MonthlyCharges"] * scaler.scale_[1] + scaler.mean_[1], 2),
    "total_charges":     round(high_risk_row["TotalCharges"]   * scaler.scale_[2] + scaler.mean_[2], 2),
    "senior_citizen":    bool(high_risk_row.get("SeniorCitizen", 0)),
    "has_partner":       bool(high_risk_row.get("Partner", 0)),
    "has_dependents":    bool(high_risk_row.get("Dependents", 0)),
    "paperless_billing": bool(high_risk_row.get("PaperlessBilling", 0)),
    "contract_type":     "Two year" if high_risk_row.get("Contract_Two year", 0)
                    else "One year" if high_risk_row.get("Contract_One year", 0)
                    else "Month-to-month",
    "fiber_optic":       bool(high_risk_row.get("InternetService_Fiber optic", 0)),
    "electronic_check":  bool(high_risk_row.get("PaymentMethod_Electronic check", 0)),
}

print("\nHigh-Risk Customer Profile:")
print(json.dumps(profile, indent=2))


# ── 1. WHY IS THIS CUSTOMER HIGH RISK? ───────────────────────
print("\n" + "-"*60)
print("LLM - Why is this customer high risk?")
print("-"*60)

risk_prompt = f"""
You are a customer success analyst at a telecom company.
A machine learning model flagged this customer as HIGH CHURN RISK.

Customer profile:
{json.dumps(profile, indent=2)}

In 3-4 sentences, explain in plain business language WHY this customer
is at high risk of churning. Reference specific profile attributes as evidence.
Be concise and direct.
"""

risk_explanation = ask_llm(risk_prompt, max_tokens=300)
print(textwrap.fill(risk_explanation, width=70))


# ── 2. RETENTION STRATEGY ────────────────────────────────────
print("\n" + "-"*60)
print("LLM - Retention strategies")
print("-"*60)

retention_prompt = f"""
You are a customer retention specialist at a telecom company.

High-risk customer profile:
{json.dumps(profile, indent=2)}

Risk reason: {risk_explanation}

Suggest 3 specific, actionable retention strategies for this customer.
Format each as:
  [Strategy Name]: one sentence on the action and expected impact.

Keep it practical - what would you actually do this week?
"""

retention_strategies = ask_llm(retention_prompt, max_tokens=350)
print(retention_strategies)


# ── 3. EXECUTIVE SUMMARY ─────────────────────────────────────
print("\n" + "-"*60)
print("LLM - Executive churn summary")
print("-"*60)

overall_churn_rate   = round(df["Churn"].mean() * 100, 2)
avg_tenure_churned   = round(df[df["Churn"]==1]["tenure"].mean(), 1)
avg_tenure_retained  = round(df[df["Churn"]==0]["tenure"].mean(), 1)
avg_monthly_churned  = round(df[df["Churn"]==1]["MonthlyCharges"].mean(), 2)
avg_monthly_retained = round(df[df["Churn"]==0]["MonthlyCharges"].mean(), 2)

exec_prompt = f"""
You are a data science lead presenting to the C-suite of a telecom company.

Key findings from a churn analysis of {len(df):,} customers:

- Overall churn rate            : {overall_churn_rate}%
- Avg tenure (churned)          : {avg_tenure_churned} months
- Avg tenure (retained)         : {avg_tenure_retained} months
- Avg monthly charge (churned)  : ${avg_monthly_churned}
- Avg monthly charge (retained) : ${avg_monthly_retained}
- Model ROC-AUC                 : {roc_auc_score(y_test, y_proba):.3f}

Write a 4-5 sentence executive summary that:
1. States the business problem and scale
2. Highlights the 2-3 most important findings
3. Ends with a clear strategic recommendation

Write for a non-technical CEO. Flowing prose only, no bullet points.
"""

exec_summary = ask_llm(exec_prompt, max_tokens=450)
print(textwrap.fill(exec_summary, width=70))


# ── Save report ───────────────────────────────────────────────
report = f"""
TELCO CHURN - LLM INTELLIGENCE REPORT
{'='*60}

HIGH-RISK CUSTOMER PROFILE
{json.dumps(profile, indent=2)}

WHY HIGH RISK
{risk_explanation}

RETENTION STRATEGIES
{retention_strategies}

EXECUTIVE SUMMARY
{exec_summary}
"""

with open("churn_llm_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("\nReport saved -> churn_llm_report.txt")

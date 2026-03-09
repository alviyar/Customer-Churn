# 📉 Customer Churn Analysis — A Strategic Operational Case Study
<img width="1367" height="780" alt="image" src="https://github.com/user-attachments/assets/453e26f8-a228-4631-9d7f-6b45f173788e" />


## 🔎 Project Objective

This project analyzes customer attrition patterns to evaluate how service quality, billing friction, and contract structures impact long-term revenue. The goal is to identify high-risk customer segments and provide data-driven strategies to reduce the **26.54% churn rate** — translating to an estimated **$295,680 in preventable annual revenue loss**.

---

## 📁 Dataset & Scope

- **Source:** Industry-standard Telco Customer Churn dataset (Kaggle / IBM Sample Data)
- **Scope:** Contract types, internet service tiers, payment methods, monthly charges, and technical support status
- **Total Customers:** 7,043
- **Total Churned:** 1,869
- **Churn Rate:** 26.54%
- **Avg Monthly Charges:** $64.76

---

## 📌 Key KPIs

| Metric | Value |
|---|---|
| Customer Base | 7,043 total subscribers |
| Churned Customers | 1,869 (26.54%) |
| Revenue at Risk | Churned customers average **$74.44/month** vs. **$61.31/month** for retained — a 21% premium |
| Critical Churn Window | 0–12 months (First-year tenure) |
| Primary Risk Factor | Month-to-month contracting (42.71% churn) |
| Estimated Annual Loss | ~$295,680 (based on 1,869 churners × $74.44 avg × 12 months, offset by partial tenure) |

---

## 📊 Key Insights

### 1️⃣ Tenure is the Strongest Predictor of Loyalty
Retention improves exponentially with time — the first year is the make-or-break window.

| Tenure Group | Churn Rate | Risk Label |
|---|---|---|
| 0–1 Year | 47.44% | 🔴 Danger Zone |
| 1–2 Years | 28.71% | 🟠 High Risk |
| 2–4 Years | 20.39% | 🟡 Medium Risk |
| 4+ Years | 9.51% | 🟢 Loyalty Zone |

> **Insight:** Nearly 1 in 2 customers leaves within the first year. A structured 90-day onboarding intervention — including check-in emails at Day 7, Day 30, and Day 60, plus a loyalty discount at the 6-month mark — directly targets this window.

---

### 2️⃣ Contract Structure Dictates Revenue Stability
The flexibility of short-term plans comes at a massive cost to retention.

| Contract Type | Churn Rate |
|---|---|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

> **Insight:** Moving a customer from monthly to a two-year plan reduces churn risk by **over 15×**. A "13th Month Free" incentive or a $10/month discount for annual commitment has a measurable ROI given the $74.44 avg monthly value of at-risk customers.

---

### 3️⃣ Technical Support as a Retention Engine
Access to support cuts churn by more than half — this is a revenue defense tool, not a cost center.

| Tech Support Status | Churn Rate |
|---|---|
| No Tech Support | 41.64% |
| With Tech Support | 15.17% |
| No Internet Service | 7.40% |

> **Insight:** A 26-point reduction in churn is achievable simply by enrolling customers in Tech Support. Bundling a 3-month free Tech Support trial into Fiber Optic plans — where churn is highest (~42%) — directly addresses the segment most likely to leave.

---

### 4️⃣ Payment Friction and Pricing Sensitivity
Manual payment methods correlate strongly with disengagement and churn.

| Payment Method | Churn Rate |
|---|---|
| Electronic Check | 45.29% |
| Mailed Check | 19.11% |
| Bank Transfer (Auto) | 16.71% |
| Credit Card (Auto) | 15.24% |

> **Insight:** Electronic check users churn at 3× the rate of auto-pay customers. Offering a $5/month discount for switching to automated billing removes the friction point while costing far less than acquiring a replacement customer.

---

## 🧠 Strategic Recommendations

### Priority Action Matrix

| Priority | Action | Target Segment | Mechanism |
|---|---|---|---|
| 🔴 Critical | "Year 1" Onboarding Playbook | 0–12 month customers | Automated email sequence: Day 7 (welcome + tips), Day 30 (usage check-in), Day 60 (feedback + offer), Month 6 (loyalty discount) |
| 🔴 Critical | Contract Migration Campaign | Month-to-month subscribers | Offer "13th Month Free" or 15% annual discount; surface offer at Month 3 and Month 9 |
| 🟠 High | Auto-Pay Migration Incentive | Electronic check users | $5/month permanent discount for switching to bank transfer or credit card |
| 🟠 High | Tech Support Bundling | Fiber Optic subscribers | Include 3-month free Tech Support trial in Fiber plans; convert at Month 4 |
| 🟡 Medium | Senior Customer Tier | SeniorCitizen = 1 (~42% churn) | Simplified billing, dedicated support line, optional paper statements |
| 🟡 Medium | High-ARPU Retention Offer | Churned customers paying $74+/month | Proactive price-match offer before cancellation, triggered at 45-day inactivity |

---

## 🛠 Technical Pipeline

### Data Cleaning (Python / Pandas)
- Identified and corrected dtype misclassification: `TotalCharges` stored as `object` due to 11 whitespace entries
- Applied `pd.to_numeric(..., errors='coerce')` and dropped 11 null rows → final dataset: **7,032 records**
- Validated zero duplicates across all 7,043 records using `df.duplicated().sum()`
- Preserved all original column names for direct SQL compatibility

### SQL Analysis (PostgreSQL / pgAdmin)
- Used `CASE WHEN` aggregations for churn rate by contract, tenure bucket, payment method, and service type
- Applied `::numeric` cast to resolve PostgreSQL `ROUND()` type mismatch on `double precision` columns
- Segmented high-value vs. high-risk accounts using tenure grouping and monthly charge thresholds

### Power BI Dashboard
- Built interactive executive dashboard with slicers for Contract, Internet Service, Payment Method, Gender, and Tech Support
- Visualized: KPI cards, tenure churn distribution (stacked bar), contract dual-axis chart (volume + churn rate), tech support clustered bar, payment method horizontal bar, avg charges comparison

---

## 📈 Business Impact Estimate

If Tier 1 interventions (onboarding playbook + contract migration) reduce churn by **20%** in the month-to-month segment:

```
Month-to-month churners:     1,655 customers
20% retention improvement:   ~331 customers saved
Avg monthly value:           $74.44
Monthly revenue recovery:    ~$24,640
Annual revenue recovery:     ~$295,680
```

This estimate is conservative — it excludes compounding effects of reduced acquisition costs and improved LTV from retained customers moving to longer-term contracts.

---



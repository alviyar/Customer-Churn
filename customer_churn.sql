select * from customer

-- 1)What is the overall churn rate?
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer;

--2)How many customers are on each contract type, and what is the churn rate per contract?
SELECT 
    "Contract",
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer
GROUP BY "Contract"
ORDER BY churn_rate_pct DESC;

--3)What is the average monthly charges for churned vs non-churned customers?
SELECT 
    "Churn",
    ROUND(AVG("MonthlyCharges")::numeric, 2) AS avg_monthly_charges,
    ROUND(AVG("TotalCharges")::numeric, 2) AS avg_total_charges
FROM customer
GROUP BY "Churn";

--4) What is the average tenure of churned vs non-churned customers?
SELECT 
    "Churn",
    ROUND(AVG("tenure"), 2) AS avg_tenure_months
FROM customer_churn
GROUP BY "Churn";

--5)  Which internet service type has the highest churn rate?
SELECT 
    "InternetService",
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer
GROUP BY "InternetService"
ORDER BY churn_rate_pct DESC;

--6) Does having a partner or dependents affect churn?
SELECT 
    "Partner",
    "Dependents",
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer
GROUP BY "Partner", "Dependents"
ORDER BY churn_rate_pct DESC;

--7)Which payment method has the highest churn rate?
SELECT 
    "PaymentMethod",
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer
GROUP BY "PaymentMethod"
ORDER BY churn_rate_pct DESC;

--8)What percentage of senior citizens churn?
SELECT 
    "SeniorCitizen",
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer
GROUP BY "SeniorCitizen";

--9) What is the churn rate by tenure group (bucketed)?
SELECT 
    CASE 
        WHEN "tenure" BETWEEN 0 AND 12  THEN '0-12 months'
        WHEN "tenure" BETWEEN 13 AND 24 THEN '13-24 months'
        WHEN "tenure" BETWEEN 25 AND 48 THEN '25-48 months'
        WHEN "tenure" BETWEEN 49 AND 60 THEN '49-60 months'
        ELSE '60+ months'
    END AS tenure_group,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer
GROUP BY tenure_group
ORDER BY churn_rate_pct DESC;

--10)Which combination of contract type and payment method has the highest churn?
SELECT 
    "Contract",
    "PaymentMethod",
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer
GROUP BY "Contract", "PaymentMethod"
ORDER BY churn_rate_pct DESC;
Payment Fraud Detection & Transaction Risk Analysis

From 7.48M transactions to an explainable fraud investigation system

I built this project to understand fraud detection as a complete analytics problem, not just a classification exercise. The goal was to take a large transaction dataset, reconstruct customer behavior over time, detect suspicious activity without leaking future information, and turn model scores into something a fraud analyst could actually investigate.

The result is an end-to-end workflow using Python, SQL, DuckDB, LightGBM, SHAP, Plotly, and Streamlit.



Demo: The Streamlit investigation app is included in this repository and can be launched locally with streamlit run app/app.py.

Tech Stack











Project at a Glance

Metric

Result

Transactions processed

7,483,766

Unique source transaction IDs

7,477,306

Customers

4,869

Cards

5,000

Merchants

105

Fraud rate

19.97%

Final test transactions

1,206,491

ROC-AUC

0.9969

PR-AUC

0.9907

Precision

96.99%

Recall

93.36%

F1 Score

95.14%

Labeled fraud transaction amount captured at 0.75 threshold*

96.53%

Monitoring days evaluated

5

Performance drift alerts

0

*The dataset contains multiple currencies. This percentage uses the source amount field as provided and is not a currency-normalized monetary estimate.

End-to-End Architecture

flowchart LR
    A["Raw Transactions<br/>7.48M rows"] --> B["DuckDB<br/>Ingestion"]
    B --> C["Cleaning +<br/>Data Quality"]
    C --> D["Transaction ID<br/>Integrity Repair"]
    D --> E["Behavioral +<br/>Velocity Features"]
    E --> F["Relationship<br/>Features"]
    F --> G["Chronological<br/>Train / Validation / Test"]
    G --> H["Logistic Regression<br/>Baseline"]
    G --> I["LightGBM<br/>Risk Model"]
    I --> J["Threshold +<br/>Business Logic"]
    J --> K["SHAP<br/>Explainability"]
    K --> L["Streamlit<br/>Investigation App"]
    I --> M["Drift + Performance<br/>Monitoring"]

1. Data Engineering

The source covers transactions from September 30 to October 30, 2024. With more than 7.48 million rows, I used DuckDB and SQL for ingestion, profiling, transformations, and large window calculations instead of repeatedly loading the full dataset into pandas.

One of the first issues I found was 6,453 reused transaction IDs. They looked like duplicates at first, but deeper checks showed that the same IDs could have different customers, cards, timestamps, amounts, devices, IP addresses, and even fraud labels.

Rather than deleting valid observations, I preserved the original source ID and generated a new unique analytical transaction_key.

Result: all 7,483,766 transactions were retained without treating distinct transactions as duplicates.

2. Behavioral & Relationship Features

Fraud is difficult to understand from a single transaction. I rebuilt the customer context that existed before each transaction, including:

prior transaction count

minutes since previous transaction

historical average spend

amount vs customer average

1-hour transaction velocity

24-hour transaction velocity

prior-hour and prior-day spending

new merchant activity

overnight behavior

I also built historical relationships across customers, cards, devices, and merchants.

The important part was keeping the model time-safe. Historical averages and velocity calculations use only earlier transactions, so future activity does not leak backward into the prediction.

Some device-network signals were almost perfectly associated with fraud. Because the dataset is synthetic, I treated those as potential data-generation shortcuts and kept the near-deterministic signals for investigation and benchmarking, not the main ML model.

3. Modeling & Validation

I trained Logistic Regression first as an interpretable baseline, then compared it with LightGBM.

Instead of randomly mixing transactions from across the month, I used a chronological split:

Dataset

Period

Train

Sep 30 – Oct 20

Validation

Oct 21 – Oct 25

Final Test

Oct 26 – Oct 30

This tests a more realistic question: can a model trained on past activity detect fraud in future transactions?

Model Comparison

Model

ROC-AUC

PR-AUC

Best F1

Logistic Regression

0.9705

0.9233

0.8365

LightGBM

0.9968

0.9906

0.9508

LightGBM captured the nonlinear transaction patterns better and became the final model.

The validation set was also used to select a 0.75 decision threshold, which was locked before the final five-day test period was evaluated.

Final Untouched Test

Metric

Result

Transactions

1,206,491

ROC-AUC

0.9969

PR-AUC

0.9907

Precision

96.99%

Recall

93.36%

F1

95.14%

True Positives

225,252

False Positives

6,994

False Negatives

16,019

True Negatives

958,226

Validation and final-test performance stayed very close, which gave me more confidence that the model generalized across the later time period.

4. Explainable Risk Decisions

A fraud score is much more useful when an investigator can understand what caused it.

I used SHAP to calculate transaction-level feature contributions and surfaced them directly inside the investigation app.



The strongest global drivers included:

distance from home · card presence · transaction amount · currency · merchant · transaction hour · payment channel · card type · amount vs customer history · country

This lets the system answer both:

How risky is this transaction?

and

Why did the model consider it risky?

5. Business Decision Layer

A probability alone does not tell a fraud team what action to take.

I evaluated thresholds using:

fraud transactions caught

fraud missed

false alerts

review volume

transaction recall

labeled fraud transaction amount captured

At the selected 0.75 threshold, the model achieved 96.99% precision, 93.36% recall, and captured 96.53% of the labeled fraud transaction amount in the test set using the source amount field.

Rules vs Machine Learning

Approach

Precision

Recall

Fraud Amount Captured*

Behavioral Rules

58.07%

41.44%

73.05%

ML Only

96.99%

93.36%

96.53%

ML + Behavioral Rules

75.60%

95.48%

98.91%

*Based on the source amount field without currency normalization.

The hybrid approach caught more fraud, but it also produced substantially more false alerts. That tradeoff matters when investigation teams have limited review capacity.

6. Fraud Investigation App

I built a Streamlit workspace so the project did not end with a prediction CSV.

An analyst can:

open the highest-risk transaction queue

inspect false positives

inspect false negatives

search individual transactions

view a 0–100 risk score

review the recommended action

understand model risk drivers

inspect customer behavior

investigate linked cards, devices, and customers

Customer Intelligence



This view gives the investigator context around historical spending, transaction velocity, previous transactions, and recent customer behavior.

Relationship Analysis



The relationship layer helps surface shared devices, cards, and customers that may deserve additional investigation.

Full-period relationship tables are used for investigation only. Future relationship information is not used as a predictive feature by the main model.

7. Monitoring

The final stage asks what happens after the model is deployed.

I built monitoring for:

daily precision and recall

alert rate

fraud rate

prediction-score drift

numeric feature drift

categorical feature drift

Across the five-day monitoring period:

Monitoring Result

Outcome

Precision

96.95% – 97.04%

Recall

93.26% – 93.52%

Score drift

None

Categorical drift

None

Performance deterioration

None

Five high-PSI alerts were generated for customer_prior_transactions. That feature naturally increases as customer histories accumulate over time.

Because model scores, precision, and recall stayed stable, I treated this as structural time drift, not automatic evidence that retraining was required.

What Made This Project Difficult

Challenge

How I handled it

7.48M transactions

Moved large-scale profiling, filtering, aggregations, and historical window calculations into DuckDB and SQL.

Reused transaction IDs

Investigated conflicts field by field and created a unique analytical transaction key instead of deleting valid rows.

Data leakage risk

Built historical averages, velocity, and behavioral signals using ordered windows so each row only uses prior information.

Synthetic shortcuts

Excluded near-deterministic device-network signals from the main model rather than using them to inflate performance.

Business tradeoffs

Compared precision, recall, false alerts, review volume, rule systems, and threshold behavior instead of optimizing ROC-AUC alone.

Drift interpretation

Separated feature movement from true model deterioration by checking score distributions and daily predictive performance.

Real Results vs Scenario Assumptions

The transaction counts, model metrics, fraud rates, model outputs, and monitoring results above are calculated from the project dataset.

The underlying dataset is synthetic, so the results should be interpreted as portfolio experimentation rather than real banking performance.

For business-cost experiments, I also tested assumptions such as:

Scenario Variable

Assumption

Manual review cost

$5

False-positive friction cost

$15

Fraud recovery rate

90%

These are scenario assumptions only. They are not actual bank costs, recovery rates, realized savings, or claims about real-world financial impact.

Reproducibility & Pipeline

The project is organized as a sequence of reusable scripts rather than one large notebook.

Ingestion
   ↓
Cleaning
   ↓
Data Quality Investigation
   ↓
Behavioral Features
   ↓
Relationship Features
   ↓
Model Dataset
   ↓
Logistic Regression Baseline
   ↓
LightGBM
   ↓
Final Test
   ↓
Business Decisioning
   ↓
Explainability
   ↓
Monitoring
   ↓
Streamlit Investigation App

I also used deterministic development samples, chronological validation, integrity checks, and saved intermediate reports so the workflow can be reviewed and reproduced.

Repository Structure

Payment_Fraud_Analysis/
│
├── app/
│   └── app.py
│
├── assets/
│   ├── fraud_app_overview.png
│   ├── fraud_app_risk_drivers.png
│   ├── fraud_app_customer_intelligence.png
│   └── fraud_app_relationship_analysis.png
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   ├── analysis/
│   ├── business_results/
│   ├── data_quality/
│   ├── explainability/
│   ├── model_results/
│   └── monitoring/
│
├── src/
│   ├── ingestion/
│   ├── cleaning/
│   ├── features/
│   ├── models/
│   ├── risk/
│   └── monitoring/
│
├── requirements.txt
├── .gitignore
└── README.md

Large raw datasets, DuckDB databases, and model binaries are excluded from Git where appropriate.

Run Locally

Create and activate the environment:

python -m venv .venv

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

After generating the required pipeline and model artifacts:

streamlit run app/app.py

What I Learned

This project strengthened both my technical and business approach to analytics.

The most useful lessons came from decisions around data integrity, leakage, suspiciously strong predictors, threshold tradeoffs, explainability, and model drift rather than simply training the highest-scoring model.

It reinforced the way I like to work: build the analytics carefully, question the result, and then turn it into something useful for the person making the decision.

About Me

Hi, I'm Nisha Rajkumar, an M.S. Business Analytics candidate at the University of Rochester's Simon Business School, after completing my B.Tech in Biotechnology at SRM Institute of Science and Technology.

I enjoy working with messy data, SQL/Python workflows, business analysis, visualization, and machine learning when it genuinely improves the decision being made.

I'm interested in opportunities across Business Analytics, Data Analytics, BI, Risk Analytics, Operations Analytics, Product Analytics, and Strategy Analytics.
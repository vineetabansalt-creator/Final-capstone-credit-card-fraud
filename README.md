### Detecting Credit Card Fraud with Machine Learning

**[Your Name]**

#### Executive summary

**The problem.** Credit card fraud costs banks, merchants and consumers billions every year, but fraud is extremely rare: in this project's data, only about 1 in every 600 transactions is fraudulent. A fraud system has to find those few cases without blocking honest customers, because both missed fraud and false alarms cost money.

**What I did.** Using 283,726 real card transactions, I trained and compared seven different models, including two that learn without being told which transactions are fraud. I then turned the best model into a practical business decision: how sensitive the alert system should be to give the lowest total cost, counting both fraud losses and the cost of reviewing alerts.

**What I found.** The best model (XGBoost) caught **77% of fraud** in transactions it had never seen, while flagging only **11 genuine customers out of 56,672**. More than 5 out of every 6 alerts it raised were real fraud. Compared with having no fraud model, it cut the combined cost of fraud losses and alert reviews by **65%**.

**The surprising part.** Two "common sense" approaches, flagging every large purchase and using a model tuned to catch almost all fraud, both ended up **costing more than doing nothing at all**, because they buried the review team in false alarms.

**Recommendation.** Score transactions with a gradient-boosting model rather than simple amount rules, set the alert level based on how many alerts the review team can handle, and don't chase catching every last fraud: beyond about 75–80% capture, each extra fraud caught costs a disproportionate number of false alarms.

![Total cost and fraud caught by strategy](images/business_comparison.png)

#### Rationale

A fraud system can fail in two ways, and both are expensive:

- **Missed fraud.** The bank or merchant usually absorbs the loss, and the customer loses trust.
- **False alarms.** Genuine purchases get declined or held for review. Customers get frustrated, call support or switch cards, and analysts waste time on transactions that were fine.

Without a data-driven approach, organizations often fall back on simple rules such as "flag anything over a certain amount." This project shows how much better a machine learning model can do, and, just as importantly, how to set it up so that it saves money overall instead of simply catching more fraud at any cost.

#### Research Question

Can a machine learning model flag fraudulent credit card transactions accurately enough to catch most fraud while keeping false alarms on legitimate customers low, and which approach gives the best balance of fraud losses prevented versus customers wrongly inconvenienced?

#### Data Sources

**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains 284,807 transactions made by European cardholders over two days in September 2013, of which 492 (0.17%) are fraud.

Each transaction has:
- **Time:** seconds since the first transaction in the dataset.
- **Amount:** the transaction amount. The source doesn't state the currency; I show amounts in euros (€) because the cardholders were European, but this is an assumption and doesn't affect the results.
- **V1–V28:** 28 numeric features that the bank anonymized to protect cardholder privacy. They help the models, but their real-world meaning (merchant, location and so on) is hidden.
- **Class:** whether the transaction was fraud (1) or legitimate (0).

**Cleaning:** there were no missing values. I removed 1,081 exact duplicate records, leaving **283,726 transactions with 473 frauds**.

**What the data showed:**

*Fraud is extremely rare.* A model that simply labels everything "legitimate" would be 99.8% accurate while catching no fraud at all, so accuracy is useless for judging these models.

![Transactions by class](images/class_distribution.png)

*Fraud happens disproportionately at night.* Normal shopping drops sharply in the early hours, but fraud doesn't, so the share of transactions that are fraud jumps to as much as 1.45% in the quietest hour, nearly 9 times the overall rate.

![Fraud rate by hour](images/fraud_by_hour.png)

*Fraudsters often start small.* Over a third of fraud (36%) is €1 or less, and transactions of €1 or less are more than three times as likely to be fraud as average. This fits a known tactic called card testing, where criminals make a tiny charge to check whether a stolen card works. Most fraud is small, which is why a "flag large purchases" rule fails.

![Fraud rate by amount](images/fraud_by_amount.png)

#### Methodology

1. **Fair testing that mimics real life.** I sorted transactions by time, trained models on the earliest 80% and kept the latest 20% aside as a final test, used only once at the end. A bank trains on past data and uses the model on future transactions, so this is a more honest test than mixing the data randomly.
2. **Seven models compared:**
   - **Logistic Regression** (a simple, transparent model) in three versions, each handling the rarity of fraud differently: giving fraud extra weight, creating synthetic fraud examples (SMOTE), or discarding some legitimate transactions (undersampling).
   - **Random Forest** and **XGBoost**: models that combine many decision trees and can pick up more complex patterns.
   - **Isolation Forest** and **One-Class SVM**: "anomaly detectors" that never see which transactions are fraud. They flag anything unusual, which could in theory catch new types of fraud.
3. **Tuning each model's settings** with grid search and time-aware cross-validation: each model was repeatedly trained on earlier data and checked on the period immediately after, and the best settings were kept.
4. **Choosing the right yardstick.** I compared models using **PR-AUC**, which measures how well a model balances two things that matter to the business:
   - **Recall:** what share of all fraud the model catches.
   - **Precision:** what share of the model's alerts are real fraud.

   PR-AUC runs from 0 to 1 (higher is better); random guessing scores about 0.001.
5. **Turning the model into a business decision.** A model gives each transaction a fraud score; someone has to decide how high the score must be to raise an alert. I picked the alert level that minimized **total cost = losses from missed fraud + €5 for every alert reviewed**. The €5 is an assumed cost of an analyst's time and customer inconvenience, so I also tested values from €1 to €25. The alert level was chosen on the training period, then checked on the test period.

#### Results

**1. Tree-based models perform best.**

| Model | PR-AUC on test data |
|---|---|
| **XGBoost** (recommended) | **0.802** |
| Random Forest | 0.802 |
| Logistic Regression, extra weight on fraud | 0.769 |
| Logistic Regression, synthetic fraud (SMOTE) | 0.759 |
| Logistic Regression, undersampling | 0.753 |
| One-Class SVM (no fraud labels) | 0.063 |
| Isolation Forest (no fraud labels) | 0.044 |

XGBoost and Random Forest tied on the test data. I recommend XGBoost because it scored slightly higher during tuning, had the highest precision, and trained about six times faster, which matters because fraud models need regular retraining.

![Model comparison](images/model_comparison.png)

**2. The simplest way of handling rare fraud worked best.** Giving fraud extra weight did as well as or better than generating synthetic fraud or throwing away legitimate data.

**3. Anomaly detection doesn't work well here.** Models that look for "unusual" transactions performed poorly, because plenty of legitimate transactions are unusual too. Even as a backup to XGBoost, Isolation Forest would have caught only 2 of the 17 frauds XGBoost missed, at the cost of 568 extra reviews.

**4. What the best model would have meant for the business.** Over the test period (56,746 transactions, 74 frauds worth €7,728):

| Strategy | Alerts | Fraud caught | Genuine customers flagged | Total cost* |
|---|---|---|---|---|
| No fraud model | 0 | 0 of 74 | 0 | €7,728 |
| Flag every purchase of €500 or more | 1,613 | 4 of 74 | 1,609 | €11,560 |
| Simple model tuned to catch most fraud | 2,472 | 69 of 74 | 2,403 | €13,709 |
| **XGBoost (recommended)** | **68** | **57 of 74** | **11** | **€2,741** |

*Total cost = fraud not prevented + €5 per alert reviewed (assumed).

The recommended model reduces total cost by **65%** compared with doing nothing. Notice that the amount rule and the "catch almost everything" model both cost **more** than having no model, because of the review workload their false alarms create.

![Results of the recommended model](images/final_confusion_matrix.png)

**5. The exact alert level matters little for XGBoost.** The model is very decisive: almost every transaction gets a fraud score near 0 or near 1. As a result, a wide range of alert levels gives a total cost within about 6% of the best. The business can therefore set the alert level based on review-team capacity without losing much.

![Cost by alert level](images/cost_curve.png)

**6. What drives the model.** A handful of the anonymized features, led by V14, do most of the work. The night-time and small-charge patterns are real (about 4 times riskier than average), but the anonymized features already capture this information, so the model doesn't need them as separate inputs.

![Feature importance](images/feature_importance.png)

**Limitations.** The data covers only two days in 2013, and the test period contains 74 frauds, so a figure like "77% of fraud caught" could easily vary by several percentage points on other periods. The €5 review cost and the euro currency are assumptions. Because the features are anonymized, the model can't explain *why* a particular transaction looks suspicious.

#### Next steps

- **Test on a longer period** (weeks or months) to confirm the model holds up as fraud patterns change.
- **Replace the assumed €5 review cost with real figures** from a fraud operations team, including the cost of losing customers after false declines.
- **Add customer-history features**, such as how a purchase compares with that cardholder's usual spending, location and merchants. These usually help with the roughly 1 in 4 frauds that look completely normal on their own.
- **Explain each alert.** With the original, non-anonymized data, tools such as SHAP could show analysts why each transaction was flagged.
- **Monitor and retrain regularly,** tracking the share of fraud caught and the share of alerts that are real fraud each week.

#### Outline of project

- [Notebook 1: Data cleaning, exploratory data analysis and baseline model](notebooks/1_eda_and_baseline.ipynb)
- [Notebook 2: Model tuning, comparison, interpretation and business recommendation](notebooks/2_modeling_and_evaluation.ipynb)

**Repository structure**

```
├── README.md
├── requirements.txt
├── images/                          charts used in this README
└── notebooks/
    ├── 1_eda_and_baseline.ipynb
    ├── 2_modeling_and_evaluation.ipynb
    └── data/creditcard.csv          download from Kaggle (not stored in the repo)
```

**To run the notebooks:** install the packages in `requirements.txt`, download `creditcard.csv` from the Kaggle link above into `notebooks/data/`, then run notebook 1 followed by notebook 2. Notebook 2 takes about 15 minutes, mostly for the Random Forest tuning.

##### Contact and Further Information

[Your Name]
Email: [your email]
[LinkedIn](https://www.linkedin.com/in/your-profile)

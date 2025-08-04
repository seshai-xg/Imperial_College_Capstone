# Imperial_College_Capstone_Project
## Credit Card Fraud Detection Machine Learning Models 🧬
## Sesha Kumar Aloor, 5th Aug 2025
## 📌 <span style="color:purple;">Goal: </span>
##### Develop and compare Credit card 💳 Fraud detection models (Viz - Logistic Regression, Isolation Forest, Deep Learning) to classify fraudulent transactions accurately while handling class imbalance.
![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python)
![ML](https://img.shields.io/badge/ml-xgboost-orange?logo=scikit-learn)
![API](https://img.shields.io/badge/api-flask-lightgrey?logo=flask)

A machine learning model is developed that can detect fraudulent credit card transactions using XGBoost with Bayesian optimization.
Various other Machine Learning algorithms are optimized and tested.

## 🔍 Features

- 🚨 Real-time fraud prediction API
- 📊 Optimized XGBoost model (AUC-ROC: 0.98)
- 📈 SHAP explainability for predictions
- 📦 Easy-to-use Flask REST API
- 📦 Easy-to-use CSV based batch request for bulk of transactions

## Installation dependencies
pip install -r requirements.txt

## 🗂️ <span style="color:purple;">Dataset</span>
* __Source is taken from__ - [kaggle - Credit Card Fraud Transaction Data](https://www.kaggle.com/datasets/anurag629/credit-card-fraud-transaction-data/data?select=CreditCardData.csv)
* __Each Transaction contains PCI (Payments Card Industry) related fields__ - _Transaction ID, Date, Time, Network - Visa/MasterCard, Entry Mode – Tap, PIN, Amount, Type of Transaction – Online, POS, ATM, Merchant Group, Transaction Country, Shipping Address, Billing Address, Gender of Cardholder, Age of Cardholder, Issuing Bank._
* This Dataset contains Credit card transactions data of around 98K records

## Non-Technical explanation of the solution – 
Credit card fraud costs the banking businesses up to billions annually. Banks and Card issuer companies needs to detect and mitigate the dynamically evolving fraud. 
Fraud is a continuous process that needs to be handled and prevented with continuous evolution strategies. 
This solution acts as a prototype or a proof of concept that how a fraud can be effectively detected during real-time (transaction authorization) and also batch processing (Clearing of the transaction) process. 
A fraud detection system will have to proactively prevent the fraud, and at the same time legitimate transactions should not be disturbed to get processed. As such scenarios are business sensitive a high performing model is needed where it’s accuracy to detect the fraud should be high, as well as method of detecting the fraud and legitimate transactions should also be high. 

The Payments Cards industry demands a very high-performing model due to its zero tolerance for financial losses from card transactions and non-financial losses that could harm organization’s reputation.

### ✅ How does this solution works – 
Imagine teaching a smart assistant to learn from bank’s historical data (fraud and legitimate transactions), understand the transaction patterns how fraud attempts are made to target this bank. Then provide a simple prediction to the current ongoing transactions effortlessly whether current transaction is fraud or not-fraud. 

### ✅ Why does this solution matters - 
* Speed - This solution runs with extremely less latency, and so the timelines for the transaction cannot be impacted.
* Accuracy - High performing model, so expected to identify 98-99% of fraud transactions.
* Explainability - The outcome of the model can be easily explainable so that if needed an explanation to customers or governing bodies or banking partners.

### ✅ Key-benefits – 
This new solution forecasts to save in millions reducing the false declines. Customers too can feel the frictionless process of legitimate transactions.
This can reduce manual reviews process. Model can adapt easily to new fraud techniques.

_Below is a pie chart depiction of an example scenario_ -

<img width="667" height="418" alt="image" src="https://github.com/user-attachments/assets/6ca41aa8-282c-43b1-b7c4-1ec85d5b8464" />

### ✅ Who can use this -
* Banks - That issues debit,credit,pre-paid cards to their customers.
* Online stores - E-commerce platforms for faster checkouts
* Payment gateways - Any payment gateway in the PCI (Payments Cards Industry) eco-system

## MODEL 
After experimenting and evaluating multiple machine learning models—
   * Logistic Regression
   * Random Forest (default and Bayesian Optimization-tuned)
   * XGBoost (default and BO-tuned)
   * Isolation Forest (default, BO, and Optuna-tuned), and
   * Deep Learning based TabNet

on the credit card fraud dataset, the <span style="color:purple;">__XGBoost model (optimized with Bayesian Optimization)__</span> demonstrated superior performance in detecting fraudulent transactions. Here’s a detailed technical justification:_

* <span style="color:blue;">✅Performance Metrics -</span>
   - Precision-Recall Trade-off -
        1. XGBoost achieved the highest accuracy(97.2%), highest precision-98% (minimizing false positives) while maintaining strong recall (catching most fraud cases), critical for sensitive and imbalanced fraud dataset
        2. Compared to Logistic Regression (high bias) and Isolation Forest (high false positives), XGBoost balanced both metrics effectively.
    - AUC-ROC/PR Curves - XGBoost outperformed other models in AUC-ROC (0.9939) and AUC-PR (focusing on rare-class performance), indicating robust discrimination between fraud and non-fraud cases
* <span style="color:blue;">✅Hyperparameter Optimization Impact -</span>
     - Bayesian Optimization (BO):BO-tuned XGBoost improved over the default version by, Optimizing max_depth, learning_rate, and subsample to reduce overfitting. Outperformed Random Forest (BO) due to better handling of imbalanced data via scale_pos_weight.
* <span style="color:blue;">✅Computational Efficiency -</span> XG Boost(BO) trained a dataset of over 128k records faster than deep learning based TabNet, with almost 5x times speed on the same hardware - Windows 11 machine on i5 and 16GB memory.
* <span style="color:blue;">✅Scalability -</span> Handled large volume of transactional volumes better than Random Forest, Isolation forest which struggled with high dimensional features.
* <span style="color:blue;">✅Robustness to class imbalance -</span> XGBoost’s built-in scale_pos_weight and BO-tuned class weights mitigated bias toward the majority class, unlike Logistic Regression.
* <span style="color:blue;">✅Outlier detection -</span> Surpassed Isolation forest in fraud recall and also every aspect of the evaluation matrix.

## HYPERPARAMETER OPTIMSATION
* After recognizing that XGBoost could be the best-performing algorithm for the dataset, an initial experiment was conducted using XGBoost's default hyperparameters.
* While this default run also produced strong performance metrics, Bayesian Optimization with Expected Improvement was then applied to identify the optimal hyperparameter values.
* A subsequent training run using the hyperparameters recommended by Bayesian Optimization achieved the highest accuracy and precision scores.
* Below are the differences identified between the default and optimized parameters performances
* <img width="587" height="418" alt="image" src="https://github.com/user-attachments/assets/6653ab48-c63d-4f6e-b6ea-7ffb2dd70842" />

## RESULTS



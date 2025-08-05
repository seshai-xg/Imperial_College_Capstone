# Datasheet
This dataset has been downloaded from the Kaggle platform - https://www.kaggle.com/datasets/anurag629/credit-card-fraud-transaction-data/data?select=CreditCardData.csv

## Motivation
This dataset has been created for the experimental purpose for identifying Fraud transactions in Credit card data.
Owner of the data is Anurag Verma (https://www.kaggle.com/anurag629)
Few other datasets were also created by the author to cater other kind of experimental purpose.

## Composition
### About Dataset
Every year, credit card fraud results in the loss of billions of pounds. While fraud itself presents a significant expense to the financial system, considerable costs are also associated with efforts to detect and prevent such fraud. In a competitive banking environment, institutions must carefully weigh the cost of fraud against the potential negative effects on customer experience caused by strict security measures. Specifically, false positives in fraud detection—when legitimate transactions are mistakenly declined—can inconvenience customers and harm the bank’s reputation. This underscores the importance of deploying highly effective fraud detection systems that maintain low error rates.
Each instance of the sample contains the Payments Card Industry replicated data and contains the features - Transaction ID, Date, Time, Type of Card – Visa, MasterCard, Entry Mode – Tap, PIN, Amount, Type of Transaction – Online, POS, ATM, Merchant Group, Transaction Country, Shipping Address, Billing Address, Gender of Cardholder, Age of Cardholder, Issuing Bank.
Some of the features are identified as unwanted and excluded from the model training (Ex - Gender and age of cardholder)
There are upto 98K samples in the available data. To balance the dataset with Fraud samples SMOTE method has been applied.
Missing data has been identified compared to ISO 8583 standard, yet this dataset is sufficient for the Proof of concept and experimental purpose on the Machine Learning algorithms.
No confidential or Personally Identifiable Information (PII) has been identified in this publicy available dataset.

## Collection process
Collection process or sampling methods are not available online for this dataset. As this is test and expreimental purpose dataset , no such information is made available to users.

## Preprocessing/cleaning/labelling
Several Exploratory Data Analysis has been made on this dataset.
Duplicates has been handled with python libraries. 
Outliers has been handled with python libraries.
Created couple of categorical variables from the raw dataset.
Applied feature engineering (No PCA is applied), added couple of features and excluded unwanted features (Gender, Age of the cardholder)
Applied encoding on the dataset. 
Class distribution analysis was made to identify the imbalanced class.
Fraud transactions are identified imbalance and minority class, and so SMOTE methods are applied to balance the class and over sampled the dataset.
Raw dataset is untouched and the new cleaned dataset is obtained and saved for further Machine Learning related experimental purpose.


## Uses
During the development of the model, I have identified number of other conclusions can also be drawn using this dataset, such as bank risk score,  are there any targets made based on the gender/age of the card holder etc.

Are there tasks for which the dataset should not be used? Yes, as this dataset is built with test data, the conclusions or predictions drawn out of this cannot be applied to PCI customers data. 

## Distribution
No distribution methods has been specified by the author. The dataset is made publicly available for any user to use.
No copyright or license is required to access the dataset. 

## Maintenance
No maintenance activity is observed since 2 years for this dataset.



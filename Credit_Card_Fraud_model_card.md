# Model Card
Model Name: fraud_model.xgb
Version: NA
Date: 05th Aug 2025
Author: Sesha Kumar Aloor

## Model Description
This XG Boost model with its hyperparameters optimized by Bayesian Optimization has been built to detect the Fraud transactions in the Payments Card Industry(PCI)
The model has been built in an intention to use anywhere in the PCI ecosystem. 
This can be used during a transaction authorization process to detect the fraud, or during the overnight batch process of clearing and settlement bulk transaction processing. 
(An example Implementation code is provided to call the model via API, Single observation and Batch processor)
This model can be used by Card Issuing banks(Eg- HSBC, Santander et ), Merchants(High street / E-commerce), or Acquirers (Eg- Worldpay, Sagepay etc), Payment Gateway providers or any other third party payment processors.

**Input:** This model is built as a prototype or Proof of Concept to demonstrate the detection of fraud in the Card transactions. A Credit Card test dataset available publicly in kaggle.com has been chosen to build the prototype. The number of samples in the raw dataset were around 98K and balanced the dataset using SMOTE measures for better results. Several data engineering measures has been conducted to obtain a clean dataset that can be used for training a model.

**Output:** The outcomes of the model has been carefully observed for higher performance without exhiting bias or over-fitting.
In all the modes of call to the model (Viz- API/Single observation/ Batch) The model outputs the classification of the transaction as Fraud or Non-Fraud.
Disclaimer - The model outputs based on the patterns in the training data only, and sometimes may go wrong based on unseen situations.

**Model Architecture:** Several other algorithms (Such as Random Forest, Isolation Forest, Deep learning for tabular data TabNet) were also tested before concluding with the BO optimized XG Boost model.
This model works with XG Boost (eXtreme Gradient Boosting) algorithm with the hyperparameters optimized with Bayesian Optimization method.
The best hyperparameters has been recommended by the BayesSearchCV algorithm with 5-fold cross validation method. Below are the recommendations of hyperparameters made by the Bayesian Optimization algorithm - 
Best parameters found:
* colsample_bytree: 0.9086743462993765
* gamma: 0.008365555755750488
* learning_rate: 0.025652837853875416
* max_depth: 12
* min_child_weight: 4
* n_estimators: 456
* scale_pos_weight: 1.3208758693958391
* subsample: 0.6323936187106906
This model is built on CPU without any use of GPU.

## Performance
The model exhibited greater performance over the other similar tests conducted with different algorithms. 
The model has been evaluated for its bias and overfitting and has been concluded post satisfactory results.
The model exhibited extremely high accuracy that is upto 97% which is outstanding point of this model characteristic. Which means out of 100, 97 transactions can be correctly distinguished as fraud or not-fraud.
Another outstanding characteristic point of the model is it's ability to discriminate the class and with its score of AUC-ROC 0.99.
Its precision to predict a non-fraud transaction as truely non-fraud is 98%. Which means out of 100 non-fraud transactions, 98 transactions can definitely be detected as non-fraud. 
Its precision to predict a fraud transaction as truely fraud is also very high at 97%. Which means out of 100 fraud transactions, 97 transactions can definitely be detected as fraud.

## Limitations
* This model is being proposed as a prototype or Proof of concept, which means this can cater a minor or micro functionality of vast PCI fraud area. 
* This model is trained with the publicy available test data (The integrity of the training data used is unknown). 
* Large and whole features of the PCI vast characteristics, doesn't resemble on this dataset. A part of the PCI standard data features (ISO 8583 / ISO 20022) has been used in this model. And limitations are identified to implement the model for real time applications.

## Trade-offs

Outline any trade-offs of your model, such as any circumstances where the model exhibits performance issues. 

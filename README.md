<h3><img align="center" src="https://github.com/shorthillstech/Pybanking/blob/main/logo.png"> Banking Machine Learning - Pybanking is an open source library</h3>

## Banking Project
Pybanking is an open source library that has the state of the art machine learning models for the banking industry. [Documentation](https://pybanking.gitbook.io/pybanking-shorthillstech/) can be found here along with tutorials and sample datasets. To [contribute](https://github.com/shorthillstech/Pybanking/) to the project please feel free to send a pull request and our team will review is at soon as possible. Machine Learning can help banks and financial institutions save money by automating and improving their processes. 
The AI journey for Banks and Financial institutions start with customer segmentation and understanding customer behaviour. Various statistical tools and exploratory analysis can be used to segment and understand customers and build their profile [Building 360 Customer Profile](https://towardsdatascience.com/enabling-data-ai-in-retail-banking-part-1-customer-analytics-journey-54a7ce7d2a81).

<img align="center" src="https://github.com/shorthillstech/Pybanking/blob/main/images/cust360.png">

On top of the customer profile, Machine learning can help banks to identify multiple revenue enhancement opportunities. Various predictive models like Revenue Prediction, Churn Prediction, Cross Selling Opportunities, Sales Funnel Analysis can be built to identify revenue opportunities. Models can also be built to prevent fraud, better assess risk, and to make better lending and investment decisions.

This is an opensource library which aims to create state of the art machine learning models to help all financial institutions deploy technology at scale. Multiple parts of the project use open source data available from different projects.

- Churn Model
- Marketing Prediction
- Transaction Prediction

<img align="center" src="https://github.com/shorthillstech/Pybanking/blob/main/images/model.png">

If you want to use your own data for training/ prediction functions are implemented for the same.

The project is being maintained by [Shorthills Tech](https://www.shorthillstech.com/about), which is a leading data engineering services provider.

## Installing

```bash
    pip install pybanking
```

## Usage

```python
from pybanking.example import custom_sklearn
custom_sklearn.get_sklearn_version()
'0.24.2'
```

## Churn Prediction

Title: Credit Card Customers. Name: Sakshi Goyal. Link: [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers?datasetId=982921&sortBy=voteCount)

The dataset has 10,127 rows and 20 columns, namely, Attrition_Flag, Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status, Income_Category, Card_Category, Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio.

The model predicts whether a credit card customer will churn (1) or not (0). It can help a bank to take proactive measures to provide customers better services and and turn their decision around.

```python
from pybanking.churn_prediction import model_churn
df = model_churn.get_data()
model = model_churn.pretrained("Logistic_Regression")
X, y = model_churn.preprocess_inputs(df)
model_churn.predict(X, model)
```   

## Marketing Prediction

Title: Banking Dataset - Marketing Targets. Name: Prakhar Rathi. Link: [Kaggle](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)

The dataset has 45,211 rows and 16 columns, namely, Job, Marital, Education, Default, Balance, Housing, Loan, Contact, Day, Month, Duration, Campaign, Pdays, Previous, Poutcome.

The model predicts whether a customer would subscribe for a term deposit in a direct marketing campaign. It can help the bank optimise their marketing spend and improve the ROI.

```python
from pybanking.deposit_prediction import model_banking_deposit
df_train, df_test = model_banking_deposit.get_data()
model = model_banking_deposit.pretrained("Logistic_Regression")
X, y = model_banking_deposit.preprocess_inputs(df_train, df_test)
model_banking_deposit.predict(X, model)
```
    
## Transaction Prediction

Title: Santander Customer Transaction Prediction. Name: Banco Santander. Link: [Kaggle](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview)

The dataset has 15,000 rows and 201 columns, namely, target, var_0 ... var_199. The data is encrpyted to safeguard the privacy of customer.

The model predicts whether a customer will make a transaction in the future. It can help banks incentivce inactive customers.

```python
from pybanking.transaction_prediction import model_transaction
df_train, df_test = model_transaction.get_data()
model = model_transaction.pretrained("Logistic_Regression")
X, y = model_transaction.preprocess_inputs(df_train, df_test)
model_transaction.predict(X, model)
```

## Hugging Face

We have hosted the [Churn Prediction model on Hugging Face](https://huggingface.co/spaces/shorthillstech/pybanking_churn) along with the same data. If you would like to upload custom data, please design it in a similar format to sample data and upload it.

![Hugging Face](https://github.com/shorthillstech/Pybanking/blob/main/images/Pybanking_Churn.gif)

## Contributing to Pybanking

We would love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

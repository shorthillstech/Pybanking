<h3><img align="center" src="https://github.com/shorthillstech/Pybanking/blob/main/logo.png"> Banking Machine Learning - Pybanking is an open source library </h3>

## Banking Project
Pybanking is an open source library that has the state of the art machine learning models for the banking industry. [Documentation](https://pybanking.gitbook.io/pybanking-shorthillstech/) can be found here along with tutorials and sample datasets. To [contribute](https://github.com/shorthillstech/Pybanking/) to the project please feel free to send a pull request and out will review is at soon as possible. Machine Learning can help banks and financial institutions save money by automating and improving their processes. 
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

```python
    from pybanking.churn_prediction import model_churn
    df = model_churn.get_data()
    model = model_churn.pretrained("Logistic_Regression")
    X, y = model_churn.preprocess_inputs(df)
    model_churn.predict(X, model)
```   

## Marketing Prediction


```python
    from pybanking.deposit_prediction import model_banking_deposit
    df = model_banking_deposit.get_data()
    model = model_banking_deposit.pretrained("Logistic_Regression")
    X, y = model_banking_deposit.preprocess_inputs(df)
    model_banking_deposit.predict(X, model)
```
    
## Transaction Prediction

```python
    from pybanking.transaction_prediction import model_transaction
    df = model_transaction.get_data()
    model = model_transaction.pretrained("Logistic_Regression")
    X, y = model_transaction.preprocess_inputs(df)
    model_transaction.predict(X, model)
```
## Contributing to Pybanking

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

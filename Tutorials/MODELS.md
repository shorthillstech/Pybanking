## Tutorial

## Churn Prediction

In this module, we have a dataset that consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

Challenge: We have only 16.07% of customers who have churned. Thus, it's a bit difficult to train our model to predict churning customers.

Use case: A manager at the bank is disturbed with more and more customers leaving their credit card services. With the help of pybanking now they can predict who is going to get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction.

Solution:

```python
    from pybanking.churn_prediction import model_churn
    df = model_churn.get_data(dataset)
    model_name = "Logistic_Regression"
    model = model_churn.pretrained(model_name)
    X, y = model_churn.preprocess_inputs(df, model_name)
    model_churn.predict(X, model)
```   

## Marketing Prediction

In this module, the data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (variable y).

Term deposits are a major source of income for a bank. A term deposit is a cash investment held at a financial institution. Your money is invested for an agreed rate of interest over a fixed amount of time, or term. The bank has various outreach plans to sell term deposits to their customers such as email marketing, advertisements, telephonic marketing, and digital marketing.

Telephonic marketing campaigns still remain one of the most effective way to reach out to people. However, they require huge investment as large call centers are hired to actually execute these campaigns. Hence, it is crucial to identify the customers most likely to convert beforehand so that they can be specifically targeted via call.

Solution:
```python
    from pybanking.deposit_prediction import model_banking_deposit
    train, test = model_banking_deposit.get_data(dataset)
    model_name = "Logistic_Regression"
    model = model_banking_deposit.pretrained(model_name)
    X, y = model_banking_deposit.preprocess_inputs(train, test)
    model_banking_deposit.predict(X, model)
```
    
## Transaction Prediction

In this module, we help you identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this module has the same structure as the real data banks have available to solve this problem.

Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?

Solution:
```python
    from pybanking.transaction_prediction import model_transaction
    train, test = model_transaction.get_data(train_dataset, test_dataset)
    model_name = "Logistic_Regression"
    model = model_transaction.pretrained(model_name)
    X, y = model_transaction.preprocess_inputs(train, test)
    model_transaction.predict(X, model)
```
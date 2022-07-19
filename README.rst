

Banking Project
===============
This is the banking project contains models on customer chrun, revenue prediction, and more. For queries contact apurv@shorthillstech.com.

Installing
============

.. code-block:: bash

    pip install pybanking

Usage
=====

.. code-block:: bash

    >>> from pybanking.example import custom_sklearn
    >>> custom_sklearn.get_sklearn_version()
    '0.24.2'

.. code-block:: bash

    >>> from pybanking.churn_prediction import model_churn
    >>> df = get_data()
    >>> model = pretrained("Logistic_Regression")
    >>> X, y = preprocess_inputs(df)
    >>> predict(X, model)
import sys
import os
import pandas as pd
import subprocess
import collections.abc
collections.Iterable = collections.abc.Iterable

# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization, UtilityFunction # scipy==1.7.3
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import sklearn.metrics as metrics

import warnings
warnings.filterwarnings("ignore")

def get_data(train = 'https://raw.githubusercontent.com/shorthills-tech/open-datasets/main/transaction_dataset_train.csv', test = 'https://raw.githubusercontent.com/shorthills-tech/open-datasets/main/transaction_dataset_test.csv'):
    df_train = pd.read_csv(train, index_col=0, sep=',')
    df_test = pd.read_csv(test, index_col=0, sep=',')
    return df_train, df_test

def analysis(df, input = "dataprep"):
    script_dir = os.path.dirname( __file__ )
    mymodule_dir = os.path.join( script_dir, '..', 'EDA' )
    sys.path.append( mymodule_dir )
    from data_analysis import Analysis
    da = Analysis()

    if input == "dataprep":
        return da.dataprep_analysis(df)
    elif input == "profiling":
        return da.pandas_analysis(df)
    elif input == "sweetviz":
        return da.sweetviz_analysis(df)
    else:
        return "Wrong Input"

# Preprocessing
def preprocess_inputs(df_train, df_test, model_name = "Logistic_Regression"):
    y = df_train["target"]
    X = df_train.drop("target", axis = 1)
    return X, y

def train(df_train, df_test, model_name = "Logistic_Regression"):
    X, y = preprocess_inputs(df_train, df_test)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)

    models = [
        LogisticRegression(),
        SVC(),
        DecisionTreeClassifier(),
        MLPClassifier(),
        RandomForestClassifier()
    ]
    
    model_names = [
        "Logistic_Regression",
        "Support_Vector_Machine",
        "Decision_Tree",
        "Neural_Network",
        "Random_Forest"
    ]

    if model_name == "Support_Vector_Machine_Optimized":
        # Define the black box function to optimize.
        def black_box_function(C):
            # C: SVC hyper parameter to optimize for.
            model = SVC(C = C)
            model.fit(X_train, y_train)
            y_score = model.decision_function(X_test)
            f = roc_auc_score(y_test, y_score)
            return f

        # Set range of C to optimize for.
        # bayes_opt requires this to be a dictionary.
        pbounds = {"C": [0.1, 10]}
        
        # Create a BayesianOptimization optimizer,
        # and optimize the given black_box_function.
        optimizer = BayesianOptimization(f = black_box_function, pbounds = pbounds, verbose = 2, random_state = 4)
        optimizer.maximize(init_points = 5, n_iter = 10)
        
        models = [
            LogisticRegression(),
            SVC(),
            SVC(C = optimizer.max["params"]['C']),
            DecisionTreeClassifier(),
            MLPClassifier(),
            RandomForestClassifier()
        ]

        model_names = [
            "Logistic_Regression",
            "Support_Vector_Machine",
            "Support_Vector_Machine_Optimized",
            "Decision_Tree",
            "Neural_Network",
            "Random_Forest"
        ]

    for model,name in zip(models,model_names):
         if name == model_name:
            model.fit(X_train, y_train)
            return model
    
    if model_name == "Pycaret_Best":
        subprocess.run(['pip', 'install', '--pre', 'pycaret'])
        from pycaret.classification import setup, compare_models
        exp_name = setup(data = df_train,  target = 'target')
        best = compare_models(exclude = ['lr', 'svm', 'rbfsvm', 'dt', 'rf'])
        return best
    else:
        return "Model Urecognized"


def pretrained(model_name = "Logistic_Regression"):
    df_train, df_test = get_data()
    return train(df_train, df_test, model_name)


def predict(test_X, model):
    if model == "Model Urecognized":
        return "Prediction Failed"
    else:
        return model.predict(test_X)   


if __name__ == '__main__':
    tr, ts = get_data()
    model = "Random_Forest"
    m = pretrained(model)
    print(m)
    X, y = preprocess_inputs(tr, ts)
    print(predict(ts, m))
    analysis(tr, "profiling")

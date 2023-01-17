import sys
import os
import pandas as pd
import numpy as np
import subprocess
import collections.abc
collections.Iterable = collections.abc.Iterable
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization # scipy==1.7.3
from sklearn.metrics import classification_report, confusion_matrix, precision_score,recall_score

import warnings
warnings.filterwarnings("ignore")

def get_data(url = 'https://raw.githubusercontent.com/shorthills-tech/open-datasets/main/BankChurners.csv'):
    df = pd.read_csv("/home/dyms/Pybanking/src/pybanking/credit_score/hmeq.csv")
    df = df.drop(df.columns[-2:], axis=1)
    return df

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

# Defining Functions for encoding

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

# Preprocessing
def preprocess_inputs(df, model_name = "Logistic_Regression"):
    df = df.copy()
    # Split df into X and y
    df_train, df_test = train_test_split(df, test_size = 0.2, stratify = df['BAD'])
    X_train = df_train.copy()
    y_train = X_train.pop('BAD')

    X_test = df_test.copy()
    y_test = X_test.pop('BAD')
    
    numeric_feats = list(X_train.select_dtypes(include = ['float', 'int']).columns)
    categorical_feats = list(X_train.select_dtypes('O').columns)
    cat_pl = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'most_frequent')),  #Fill missing data with the most frequent
            ('onehot', OneHotEncoder())  #One hot encoding for categorical features
        ]
    )
    num_pl = Pipeline(
        steps = [
                ('imputer', KNNImputer()),
                ('scaler', RobustScaler())
        ]
    )
    preprocessor = ColumnTransformer(
        transformers = [
                    ('numerical', num_pl, numeric_feats),       #Apply transformers for numerical features
                    ('categorical', cat_pl, categorical_feats)  #Apply transfomers for categorical features
        ]
    )

    
    if model_name == "Pycaret_Best":
        X = df
    return X_train, y_train,X_test,y_test,preprocessor



def train(df, model_name = "Logistic_Regression"):
    X_train, y_train,X_test,y_test,preprocessor = preprocess_inputs(df)
    def fbeta(y_test, y_pred):
        return fbeta_score(y_test, y_pred, beta = np.sqrt(1))
    metrics = make_scorer(fbeta)
    models = [RandomForestClassifier(), LogisticRegression(), KNeighborsClassifier(), MLPClassifier()]
    all_scores = []
    #Initial K-Fold
    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 42)

    #Evaluate model
    for model in models:
        complete_pl = Pipeline(
        steps = [
                ('preprocessing', preprocessor),
                ('classifier', model)
        ]
    )
    scores = cross_val_score(complete_pl, X_train, y_train, scoring = metrics, cv = cv, n_jobs = -1)
    all_scores.append(scores)
    print('Mean fbeta: {:.03f}, STD fbeta: {:.03f}'.format(np.mean(all_scores), np.std(all_scores)))
    model = Pipeline(
        steps = [
                ('pre', preprocessor),
                ('rdfr', RandomForestClassifier())
        ]
    )

    model.fit(X_train, y_train)
    return model


    """models = [
        LogisticRegression(),
        SVC(),
        DecisionTreeClassifier(),
        MLPClassifier(),
        RandomForestClassifier(),
        KNeighborsClassifier()
    ]
    
    model_names = [
        "Logistic_Regression",
        "Support_Vector_Machine",
        "Decision_Tree",
        "Neural_Network",
        "Random_Forest"
        "KNN"
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
            RandomForestClassifier(),
            KNeighborsClassifier()
        ]

        model_names = [
            "Logistic_Regression",
            "Support_Vector_Machine",
            "Support_Vector_Machine_Optimized",
            "Decision_Tree",
            "Neural_Network",
            "Random_Forest"
            "KNN"
        ]
    
   

    for model,name in zip(models,model_names):
         if name == model_name:
            model.fit(X_train, y_train)
            return model
    
    if model_name == "Pycaret_Best":
        subprocess.run(['pip', 'install', '--pre', 'pycaret'])
        from pycaret.classification import setup, compare_models
        exp_name = setup(data = df,  target = '0')
        best = compare_models(exclude = ['lr', 'svm', 'rbfsvm', 'dt', 'rf','knn'])
        return best
    else:
        return "Model Urecognized"""


def pretrained(model_name = "Logistic_Regression"):
    df = get_data()
    return train(df, model_name)


def evaluate_classifier(y_test, y_pred):
  print('Precision: {:.03f}'.format(precision_score(y_test, y_pred, pos_label = 1)))
  print('Recall: {:.03f}'.format(recall_score(y_test, y_pred, pos_label = 1)))
  cfm = confusion_matrix(y_test, y_pred, normalize = 'true')
  sns.heatmap(cfm, annot = True, cmap = 'Blues')
  print(classification_report(y_test, y_pred))


def predict(test_X, model):
    if model == "Model Urecognized":
        return "Prediction Failed"
    else:
        return model.predict(test_X)   


if __name__ == '__main__':
    df = get_data()
    print(df.head())
    model = "Random_Forest"
    m = pretrained(model)
    print(m)
    X_train, y_train,X_test,y_test,preprocessor = preprocess_inputs(df)
    p=predict(X_test, m)
    print(p)
    evaluate_classifier(y_test,p)
    analysis(df, "profiling")
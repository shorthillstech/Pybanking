import pandas as pd
import numpy as np
# from pycaret.classification import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings("ignore")

def get_data(url = 'https://raw.githubusercontent.com/nikhil0nk/credit_card_customer_churning/main/BankChurners.csv'):
    # url = 'https://raw.githubusercontent.com/nikhil0nk/credit_card_customer_churning/main/BankChurners.csv'
    df = pd.read_csv(url,index_col=0)
    # Drop last two columns (unneeded)
    # df = df.drop(df.columns[-2:], axis=1)

    # Drop CLIENTNUM columns
    # df = df.drop('CLIENTNUM', axis=1)
    return df


# Defining Functions for encoding
def binary_encode(df, column, positive_value):
    df = df.copy()
    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df

def ordinal_encode(df, column, ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x: ordering.index(x))
    return df

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

# Preprocessing
def preprocess_inputs(df):
    df = df.copy()
    
    # Encode unknown values as np.NaN
    df = df.replace('Unknown', np.NaN)
    
    # Fill ordinal missing values with modes (Education_Level and Income_Category columns)
    df['Education_Level'] = df['Education_Level'].fillna('Graduate')
    df['Income_Category'] = df['Income_Category'].fillna('Less than $40K')
    
    # Encode binary columns
    df = binary_encode(df, 'Attrition_Flag', positive_value='Attrited Customer')
    df = binary_encode(df, 'Gender', positive_value='M')
    
    # Encode ordinal columns
    education_ordering = [
        'Uneducated',
        'High School',
        'College',
        'Graduate',
        'Post-Graduate',
        'Doctorate'
    ]
    income_ordering = [
        'Less than $40K',
        '$40K - $60K',
        '$60K - $80K',
        '$80K - $120K',
        '$120K +'
    ]
    df = ordinal_encode(df, 'Education_Level', ordering=education_ordering)
    df = ordinal_encode(df, 'Income_Category', ordering=income_ordering)
    
    # Encode nominal columns
    df = onehot_encode(df, 'Marital_Status', prefix='MS')
    df = onehot_encode(df, 'Card_Category', prefix='CC')
    
    # Split df into X and y
    y = df['Attrition_Flag'].copy()
    X = df.drop('Attrition_Flag', axis=1).copy()
    
    # Scale X with a standard scaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y



def train(df, model_name = "Logistic_Regression"):
    X, y = preprocess_inputs(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    
    n_cols = df.shape[1]
    df.columns = [str(i) for i in range(n_cols)]
    df['0'] = df['0'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)


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
        "Support_Vector Machine",
        "Support_Vector Machine (Optimized)",
        "Decision_Tree",
        "Neural_Network",
        "Random_Forest"
    ]

    for model,name in zip(models,model_names):
         if name == model_name:
            model.fit(X_train, y_train)
            return model
    
    # if model_name == "pycaret_best":
    #         exp_name = setup(data = df,  target = '0')
    #         best = compare_models()
    #         return best
    # else:
    #     return "Model Urecognized"


def pretrained(model_name = "Logistic_Regression"):
    df = get_data()
    return train(df, model_name)


def predict(X_test, model):
    # X, y = preprocess_inputs(df)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
        
    # n_cols = df.shape[1]
    # df.columns = [str(i) for i in range(n_cols)]
    # df['0'] = df['0'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)


    # # Define the black box function to optimize.
    # def black_box_function(C):
    #     # C: SVC hyper parameter to optimize for.
    #     model = SVC(C = C)
    #     model.fit(X_train, y_train)
    #     y_score = model.decision_function(X_test)
    #     f = roc_auc_score(y_test, y_score)
    #     return f

    # # Set range of C to optimize for.
    # # bayes_opt requires this to be a dictionary.
    # pbounds = {"C": [0.1, 10]}
    
    # # Create a BayesianOptimization optimizer,
    # # and optimize the given black_box_function.
    # optimizer = BayesianOptimization(f = black_box_function, pbounds = pbounds, verbose = 2, random_state = 4)
    # optimizer.maximize(init_points = 5, n_iter = 10)
    return model.predict(X_test)
    
    # if model_name == "pycaret_best":
    #         res = predict_model(train(df, model_name), X_test)
    #         lbl = res['Label']
    #         print(train(df, model_name))
    #         print(confusion_matrix(y_test,lbl))
    #         print(classification_report(y_test,lbl))
    #         print(accuracy_score(y_test, lbl))
    #         print("\n")
    # else:
    #     return "Model Urecognized"
    
if __name__ == '__main__':
    # df = get_data('https://raw.githubusercontent.com/nikhil0nk/credit_card_customer_churning/main/BankChurners.csv')
    # model = pretrained("Logistic_Regression")
    # X, y = preprocess_inputs(df)
    # predict(X, model)
    pass
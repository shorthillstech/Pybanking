from ssl import VERIFY_ALLOW_PROXY_CERTS
import sys
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from pybanking.churn_prediction.model_churn import preprocess_inputs
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso
import subprocess
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
#Read Data
def get_data():
    train_df = pd.read_csv('https://raw.githubusercontent.com/shorthills-tech/open-datasets/main/Value_prediction_train.csv',index_col=0)
    test_df = pd.read_csv('https://raw.githubusercontent.com/shorthills-tech/open-datasets/main/Value_prediction_test.csv',index_col=0)
    return train_df,test_df
#Exctract Important Features
def important_feat(train_df,test_df,model_name):
    train_X,test_X,train_y,dev_X,val_X,dev_y,val_y= preprocess_inputs(train_df,test_df,model_name)
    if model_name=="Lasso":
        return train_df,test_df
    else:
        model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0)
        model.fit(train_X, train_y)
        feat_names = train_X.columns.values
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:50]

        tr_df=train_df[train_df.columns[train_df.columns.isin(feat_names[indices])]]
        te_df=test_df[test_df.columns[test_df.columns.isin(feat_names[indices])]]
        if model_name != "Pycaret_Best":
            tr_df.insert(2,"target",train_df["target"].values)
        return tr_df,te_df


#Training
def train(tr_df,te_df,model_name):
    train_X,test_X,train_y,dev_X,val_X,dev_y,val_y= preprocess_inputs(tr_df,te_df,model_name)
    models = [
            LogisticRegression(),
            SVR(),
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
    if model_name=="LGBM":
        params = {
            "objective" : "regression",
            "metric" : "rmse",
            "num_leaves" : 30,
            "learning_rate" : 0.01,
            "bagging_fraction" : 0.7,
            "feature_fraction" : 0.7,
            "bagging_frequency" : 5,
            "bagging_seed" : 2018,
            "verbosity" : -1
        }
    
        lgtrain = lgb.Dataset(train_X, label=train_y)
        lgval = lgb.Dataset(val_X, label=val_y)
        evals_result = {}
        model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=200, evals_result=evals_result)
        return model

    if model_name=="Lasso":
        model = Lasso(alpha=0.0000001, max_iter = 10000)
        model.fit(train_X,train_y)
        return model
    if model_name == "Support_Vector_Machine_Optimized":
        #building a radial basis function kernel
        estimator=SVR(kernel='rbf')
        #arbitary param values to optimize
        param_grid={
            'C': [1.1, 5.4, 170, 1001],
            'epsilon': [0.0003, 0.007, 0.0109, 0.019, 0.14, 0.05, 8, 0.2, 3, 2, 7],
            'gamma': [0.7001, 0.008, 0.001, 3.1, 1, 1.3, 5]
        }
        grid = GridSearchCV(estimator,param_grid,cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        grid.fit(train_X,train_y)
        best=grid.best_params_




        
        models = [
            LogisticRegression(),
            SVR(),
            SVR(C = best['C'],epsilon=best['epsilon'],gamma=best['gamma']),
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
            lab_enc = preprocessing.LabelEncoder()
            enc_y = lab_enc.fit_transform(train_y)
            model.fit(train_X, enc_y)
            return model
    if model_name == "Pycaret_Best":
        subprocess.run(['pip', 'install', '--pre', 'pycaret'])
        from pycaret.regression import setup, compare_models
        exp_name = setup(data = train_X,  target = 'target')
        best = compare_models(exclude = ['lr', 'svm', 'rbfsvm', 'dt', 'rf','lightgbm','lasso'])
        return best
    else:
        return "Model Unrecognized"
  
    
    


#Analysis
import warnings
warnings.filterwarnings("ignore")
def analysis(train_df, input = "dataprep"):
    script_dir = os.path.dirname( __file__ )
    mymodule_dir = os.path.join( script_dir, '..', 'EDA' )
    sys.path.append( mymodule_dir )
    from data_analysis import Analysis
    da = Analysis()

    if input == "dataprep":
        return da.dataprep_analysis(train_df)
    elif input == "profiling":
        return da.pandas_analysis(train_df)
    elif input == "sweetviz":
        return da.sweetviz_analysis(train_df)
    else:
        return "Wrong Input"
#Preprocessing
def preprocess_inputs(train_df,test_df,model_name):
    unique_df = train_df.nunique().reset_index()
    unique_df.columns = ["col_name", "unique_count"]
    constant_df = unique_df[unique_df["unique_count"]==1]
    index = pd.Index(range(0, 1784, 1))
    train_df=train_df.set_index(index)
    train_X = train_df.drop(constant_df.col_name.tolist() + [ "target"], axis=1)
    test_X = test_df.drop(constant_df.col_name.tolist(), axis=1)
    train_y = np.log1p(train_df["target"].values)
    dev_X=0
    val_X=0
    dev_y=0
    val_y=0
    if model_name=="Lasso" :
        feat = SelectKBest(mutual_info_regression,k=200)
        X_tr = feat.fit_transform(train_X,train_y)
        X_te = feat.transform(test_X)
        return X_tr,X_te,train_y,dev_X,val_X,dev_y,val_y
    if model_name=="LGBM":
        kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
        for dev_index, val_index in kf.split(train_X):
            dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
            dev_y, val_y = train_y[dev_index], train_y[val_index]
        return train_X,test_X,train_y,dev_X,val_X,dev_y,val_y
    if model_name == "Pycaret_Best":
        train_X = train_df
        return train_X,test_X,train_y,dev_X,val_X,dev_y,val_y       
    return train_X,test_X,train_y,dev_X,val_X,dev_y,val_y


#Prediction
def predict(train_y, test_X, model):
    if model == "Model Urecognized":
        return "Prediction Failed"
    elif model == "LGBM":
        pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
        return pred_test_y, model
    else:
        pred_test_y=model.predict(test_X)
        return pred_test_y,model
#Pretraining
def pretrained(tr_df,te_df,model_name):
    return train(tr_df,te_df,model_name)

if __name__ == '__main__':
    train_df,test_df = get_data()
    model_name="Support_Vector_Machine_Optimized"
    tr_df,te_df = important_feat(train_df,test_df,model_name)
    m=pretrained(tr_df,te_df,model_name)
    train_X,test_X,train_y,dev_X,val_X,dev_y,val_y=preprocess_inputs(tr_df,te_df,model_name)
    pred_test_y,m=predict(train_y,test_X,m)
    analysis(train_df,"sweetviz")










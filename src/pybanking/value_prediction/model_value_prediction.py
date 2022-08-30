import sys
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from pybanking.churn_prediction.model_churn import preprocess_inputs
import lightgbm as lgb
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso
import subprocess
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
#Read Data
def get_data(url1='https://raw.githubusercontent.com/shorthills-tech/open-datasets/main/Value_prediction_train.csv'):
    train_df = pd.read_csv(url1,index_col=0)
    return train_df
#Exctract Important Features
def important_feat(train_df,model_name="LGBM"):
    train_X,test_X,train_y,dev_X,val_X,dev_y,val_y,y_test= preprocess_inputs(train_df,model_name)
    if model_name=="Lasso":
        return train_df
    else:
        model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0)
        model.fit(train_X, train_y)
        feat_names = train_X.columns.values
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:50]

        tr_df=train_df[train_df.columns[train_df.columns.isin(feat_names[indices])]]
        if model_name != "Pycaret_Best":
            tr_df.insert(2,"target",train_df["target"].values)
        return tr_df


#Training
def train(tr_df,model_name="LGBM"):
    train_X,test_X,train_y,dev_X,val_X,dev_y,val_y,y_test= preprocess_inputs(tr_df,model_name)
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
        from pycaret.regression import setup, compare_models,predict_model
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
def preprocess_inputs(train_df,model_name="LGBM"):
    unique_df = train_df.nunique().reset_index()
    unique_df.columns = ["col_name", "unique_count"]
    constant_df = unique_df[unique_df["unique_count"]==1]
    index = pd.Index(range(0, 1784, 1))
    train_df=train_df.set_index(index)
    train_X = train_df.drop(constant_df.col_name.tolist() + [ "target"], axis=1)
    train_y = np.log1p(train_df["target"].values)
    dev_X=0
    val_X=0
    dev_y=0
    val_y=0
    #Splitting Data
    X_tr_sp, X_te_sp, y_tr_sp, y_te_sp = train_test_split(train_X, train_y, test_size=0.2, random_state=0)
    if model_name=="Lasso" :
        feat = SelectKBest(mutual_info_regression,k=200)
        X_tr = feat.fit_transform(X_tr_sp,y_tr_sp)
        X_te = feat.transform(X_te_sp)
        return X_tr,X_te,y_tr_sp,dev_X,val_X,dev_y,val_y,y_te_sp
    if model_name=="LGBM":
        kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
        for dev_index, val_index in kf.split(X_tr_sp):
            dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
            dev_y, val_y = train_y[dev_index], train_y[val_index]
        #Doesnt Require splitting
        return train_X,X_te_sp,train_y,dev_X,val_X,dev_y,val_y,y_te_sp
    if model_name == "Pycaret_Best":
        train_X=train_df
        #Cant split for Pycaret
        return train_X,X_te_sp,train_y,dev_X,val_X,dev_y,val_y,y_te_sp       
    return X_tr_sp,X_te_sp,y_tr_sp,dev_X,val_X,dev_y,val_y,y_te_sp


#Prediction
def predict(test_X, model,model_name):
    if model_name == "Model Urecognized":
        return "Prediction Failed"
    elif model_name == "LGBM":
        pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
        #f=model.score(test_y,pred_test_y)
        return pred_test_y
    else:
        pred_test_y=model.predict(test_X)
        #f=metrics.mean_squared_log_error(test_y, pred_test_y,squared=False)
        return pred_test_y
#Pretraining
def pretrained(model_name="LGBM"):
    train_df=get_data()
    tr_df=important_feat(train_df,model_name)
    return train(tr_df,model_name)

if __name__ == '__main__':
    train_df=get_data()
    model_name="LGBM"
    tr_df=important_feat(train_df,model_name)
    m=pretrained(model_name)
    train_X,test_X,train_y,dev_X,val_X,dev_y,val_y,test_y=preprocess_inputs(tr_df,model_name)
    pred_test_y=predict(test_X,m,model_name)
    analysis(train_df,"sweetviz")










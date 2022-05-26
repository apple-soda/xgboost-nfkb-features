import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import ensemble

from core.dataset import *
from core.dprocessing import *

def single_fit(features, time_steps, feature_list, bound, dataset, labels, display=True, random_state=42):
    ligands = ['CpG', 'FLA', 'FSL', 'LPS', 'P3K', 'PIC', 'R84', 'TNF']
    df = partition_features(features, time_steps, feature_list, bound, dataset)
    data = df.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy()
        labels = labels.reshape((-1, ))
        
    if random_state:
        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=random_state)
    else:
        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1)
        
    model = xgb.XGBClassifier(use_label_encoder=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    cr = classification_report(y_val, y_pred, target_names=ligands)
    cr_dic = classification_report(y_val, y_pred, target_names=ligands, output_dict=True)
    
    if display:
        print(cr)
    
    return [df, cr_dic]

def one_fit(features, dataset, target_names, random_state, verbose=True, ret=False, ret_labels=False):
    X, labels = dataset.loc[:, features].to_numpy(), dataset.iloc[:, [984]].to_numpy().reshape(-1, )
    model = xgb.XGBClassifier(use_label_encoder=False, tree_method='gpu_hist')
    
    if random_state:
        X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.1, random_state=random_state)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.1)
    
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    cr = pd.DataFrame(classification_report(y_val, pred, target_names=target_names, output_dict=True)).transpose()
    
    if verbose:
        print(cr)
        
    if ret == True and ret_labels == False:
        return cr
    
    if ret == True and ret_labels == True:
        return cr, y_val, pred
    
    
def one_fit_rf(features, dataset, target_names, random_state, verbose=True, ret=False):
    X, labels = dataset.loc[:, features].to_numpy(), dataset.iloc[:, [984]].to_numpy().reshape(-1, )
    model = xgb.XGBClassifier(use_label_encoder=False, tree_method='gpu_hist')
    
    if random_state:
        X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.1, random_state=random_state)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.1)
    
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    cr = pd.DataFrame(classification_report(y_val, pred, target_names=target_names, output_dict=True)).transpose()
    
    if verbose:
        print(cr)
        
    if ret:
        return cr
    
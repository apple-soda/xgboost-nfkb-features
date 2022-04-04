import numpy as np
import pandas as pd
import xgboost as xgb

from core.dataset import *
from core.dprocessing import *
from core.optimizers import *
from core.utils import *

class Optimizer:
    def __init__(self, model, features, time_steps, feature_importance, labels, dataset, ligands):
        self.model = model
        self.features = features
        self.time_steps = time_steps
        self.feature_importance = feature_importance
        self.labels = labels
        self.dataset = dataset
        self.ligands = ligands
        
    def fit(self, bound=1, random_state=42, display=True):
        """
        This function is essentially the same as single_fit from core.utils.py but modified for this class
        """
        df = partition_features(self.features, self.time_steps, self.feature_importance, bound, self.dataset)
        data = df.to_numpy()
        if isinstance(self.labels, pd.DataFrame):
            labels = self.labels.to_numpy()
            labels = labels.reshape((-1, ))
            
        if random_state:
            X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=random_state)
        else:
            X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        cr = classification_report(y_val, y_pred, target_names=self.ligands)
        cr_dic = classification_report(y_val, y_pred, target_names=self.ligands, output_dict=True)
        
        if display:
            print(cr)
            
        self.df = df
        self.cr_dic = cr_dic

    
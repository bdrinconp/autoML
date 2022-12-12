import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

class ClassifierModels:
    def __init__(self, data, X, Y,model_type = None, param_grid = dict,scoring = None):
        '''
        data: datos con los que se van a ajustar los modelos
        model_type: tipo de modelo svc: SVC() , logit: LogisticRegression() , 
                    rf: RandomForestClassifier(), xgb: XGBClassifier() 
        X: lista de variables explicativas
        Y: variable objetivo 
        param_grid: diccionario de parametros para gridsearch
        scoring: score de evaluación para gridsearch
        '''
        self._data = data
        self._model_type = model_type
        self._X = X
        self._Y = Y
        self._param_grid = param_grid
        
        if scoring:
            self._scoring = scoring
        else:
            self._scoring = "roc_auc"
            
    def split(self, test_size, seed = None):
        '''
        test_size: tamaño del conjunto de test
        seed: semilla 
        '''
        if seed: 
            self._seed = seed
        else:
            self._seed = 50
        X = np.array(self._data[self._X])
        y = np.array(self._data[self._Y])
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size = test_size, random_state = self._seed)
      
    def fit(self):
        if self._model_type == 'svc':
            self._user_defined_model = SVC(probability=True)
            self._grid = GridSearchCV(estimator = self._user_defined_model,
                                      param_grid = self._param_grid ,
                                      scoring = self._scoring)
            self._grid.fit(self._X_train, self._y_train)
            self.model = SVC(**self._grid.best_params_)
            self.model = self._user_defined_model.fit(self._X_train, self._y_train)
            
        if self._model_type == 'rf':
            self._user_defined_model = RandomForestClassifier()
            self._grid = GridSearchCV(estimator = self._user_defined_model,
                                      param_grid = self._param_grid ,
                                      scoring = self._scoring)
            self._grid.fit(self._X_train, self._y_train)
            self.model = RandomForestClassifier(**self._grid.best_params_)
            self.model = self._user_defined_model.fit(self._X_train, self._y_train)
            
        if self._model_type == 'logit':
            self._user_defined_model = LogisticRegression()
            self._grid = GridSearchCV(estimator = self._user_defined_model,
                                      param_grid = self._param_grid ,
                                      scoring = self._scoring)
            self._grid.fit(self._X_train, self._y_train)
            self.model = LogisticRegression(**self._grid.best_params_)
            self.model = self._user_defined_model.fit(self._X_train, self._y_train)
            
        if self._model_type == 'xgb':
            self._user_defined_model = XGBClassifier()
            self._grid = GridSearchCV(estimator = self._user_defined_model,
                                      param_grid = self._param_grid ,
                                      scoring = self._scoring)
            self._grid.fit(self._X_train, self._y_train)
            self.model = XGBClassifier(**self._grid.best_params_)
            self.model = self._user_defined_model.fit(self._X_train, self._y_train)

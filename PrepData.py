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

class DataPrep:
    '''
    outcome: variable objetivo
    features: listado de las variables (columnas) que vamos a utilizar en el proceso de preparación
    categorical_features: listado de las variables categoricas (columnas) que vamos a dumificar
    numerical_features: listado de las variables numericas (columnas) que vamos a estandarizar
    one_hot_encoder: metodo de dumificación (opcional)
    scaler_method: metodo de escalamiento (opcional)
    '''
    def __init__(self, outcome , features=None, categorical_features=None, numerical_features = None, one_hot_encoder=None, scaler_method = None):
        self._outcome = outcome
        self._features = features
        self._categorical_features = categorical_features
        self._numerical_features = numerical_features
        self._one_hot_encoder = one_hot_encoder
        self._scaler_method = scaler_method
 
    def get_data(self,data_dir,file, sep):
        '''
        data_dir: directorio de trabajo
        file: nombre del archivo con extensión archivo.csv
        sep: separado ej: sep = ','
        '''
        path = os.path.join(data_dir, file)
        data = pd.read_csv(path, sep= sep)
        if self._features:
            data = data[self._features]
        print(f'Data read from {path}')
        return data
            
    def encode_data(self, data):
        ''' 
        data: dataframe sobre el que se va a aplicar la transformación one_hot_encoder y el escalamiento
        '''
        ###################################################################################
        if self._numerical_features:
            numerical = data[self._numerical_features]
        else:
            numerical = data.select_dtypes(include=numerics)
            numerical_features = numerical.columns
        
        if self._scaler_method:
            scaler_method = self._scaler_method
        else:
            scaler_method = preprocessing.StandardScaler()
        
        numerical = scaler_method.fit_transform(numerical)
        numerical_columns = scaler_method.get_feature_names_out(self._numerical_features)
        numerical = pd.DataFrame(numerical, columns = numerical_columns)
        ##################################################################################
        if self._categorical_features:
            categorical = data[self._categorical_features]
        else:
            categorical = data.select_dtypes(exclude=np.number)
            categorical_features = categorical.columns
            
        if self._one_hot_encoder:
            one_hot_encoder = self._one_hot_encoder
        else:
            one_hot_encoder = preprocessing.OneHotEncoder(
                sparse=False,
                drop='first'
                )
             
        categorical = one_hot_encoder.fit_transform(categorical)
        categorical_columns = one_hot_encoder.get_feature_names_out(self._categorical_features)
        categorical = pd.DataFrame(categorical, columns=categorical_columns)
        ##################################################################################
        non_transform = data[list(set(self._features) - set(self._categorical_features) - set(self._numerical_features))].reset_index(drop = True)
        ##################################################################################
        data = pd.concat([non_transform, categorical, numerical ], axis = 1)
        return data, one_hot_encoder, scaler_method
 
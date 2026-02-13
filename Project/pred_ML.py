import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import numpy as np
import random
import os
from matplotlib.dates import (
    DateFormatter, AutoDateLocator, AutoDateFormatter, datestr2num)
from sklearn.multioutput import MultiOutputRegressor
import shap
import seaborn as sns
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from collections import defaultdict
import warnings
warnings.simplefilter("ignore", FutureWarning)
 
class Prediction_ML():
    
    def __init__(self, dirname, filename, feature_names,target, ahead, offset, size_split_valtest, size_split_test):
        self.dirname = dirname
        self.filename = filename
        self.feature_names = feature_names
        self.target = target
        self.ahead = ahead
        self.offset = offset
        self.size_split_valtest = size_split_valtest
        self.size_split_test = size_split_test
        self._prepare_data = self.prepare_data()
        self.x_train, self.y_train, self.x_val,self.y_val, self.x_test, self.y_test = self.features()
        self._train_model = self.train_model()

    def prepare_data(self):
        data = pd.read_csv(self.dirname + self.filename)
        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True)
        data.sort_index(inplace = True)
        data_for_pred = data[[self.target] + sorted(self.feature_names) ]
        return data_for_pred
    
    def split(self):
        train, val_test = train_test_split(self._prepare_data, test_size = self.size_split_valtest, shuffle=False)
        val, test = train_test_split(val_test, test_size =self.size_split_test, shuffle = False)
        return train, val, test
    
    # def custom_split(self):
    #     train = self._prepare_data[self._prepare_data.index.year.isin([2016, 2017])]
    #     val = self._prepare_data[self._prepare_data.index.year.isin([2018])]
    #     test = self._prepare_data[self._prepare_data.index.year.isin([2019])]
    #     return train, val, test
    
    def features(self):
        train, val, test = self.split()
        
        offset_range = np.arange(1, 1+self.offset)
        ahead_range = np.arange(0, self.ahead)
        features_col = self._prepare_data.columns
        
        def input_output(data):
            new_cols, target_cols = [], []
            for col in features_col: #generate lag features
                for day in offset_range:
                    new_col_name = f'{col}_day_-{day}'
                    new_col = data[col].shift(day)
                    new_cols.append(pd.Series(new_col, name=new_col_name))
            for day in ahead_range:
                new_col_name = f'target_day_{day+1}' #generate target values
                new_col = data[self.target].shift(-day)
                target_cols.append(pd.Series(new_col, name=new_col_name))
            # Concatenate all new columns
            all_new_cols = pd.concat(new_cols + target_cols, axis=1)
            # Prepare features and target
            data_augmented = pd.concat([data, all_new_cols], axis=1).dropna()

            x = data_augmented[[col for col in data_augmented.columns if '_day_-' in col]]
            y = data_augmented[[col for col in data_augmented.columns if col.startswith('target_day_')]]
            return x, y
        
        x_train, y_train =   input_output(train)         
        x_val, y_val     =   input_output(val)         
        x_test, y_test   =   input_output(test)         

        return x_train, y_train, x_val, y_val, x_test, y_test
    
    def train_model(self):
        model_base = CatBoostRegressor(
                verbose=0,
                allow_writing_files=False,
                n_estimators=600,
                l2_leaf_reg=10,
                depth = 3,
                colsample_bylevel=0.9,
                )
        model = MultiOutputRegressor(model_base)
        model.fit(self.x_train, self.y_train)
        return model
    
    def predict(self):
        y_pred = self._train_model.predict(self.x_test)
        return pd.DataFrame(y_pred)
        
    def metrics(self):
        y_pred = self.predict()
        pe_results, rmse_results = [], []
        for day in range(self.ahead):
            pe_results.append(np.around(r2_score(self.y_test.iloc[:, day], pd.DataFrame(y_pred).iloc[:, day]), 2))
            rmse_results.append(np.around(mean_squared_error(self.y_test.iloc[:, day], pd.DataFrame(y_pred).iloc[:, day]), 2))
        
        print(f'{self.filename[:-3]} \nR^2 up to 10 min:', pe_results)
        return pe_results, rmse_results
        
 
 

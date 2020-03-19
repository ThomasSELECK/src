#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the M5 Forecasting Accuracy competition                  #
#                                                                             #
# This file contains the code needed for the preprocessing step.              #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2020-03-15                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import time
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from datetime import datetime
from datetime import timedelta
import pickle

from m5_forecasting_accuracy.data_loading.data_loader import DataLoader

class PreprocessingStep(BaseEstimator, TransformerMixin):
    """
    This class defines the first preprocessing step.
    """

    def __init__(self, test_days, dt_col, keep_last_train_days = 0):
        """
        This is the class' constructor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.test_days = test_days
        self.dt_col = dt_col
        self.keep_last_train_days = keep_last_train_days # Number of rows at the end of the train set to keep for appending at the beginning of predict data
        self._last_train_rows = None
        self._orig_earliest_date = None
        self._is_train_data = False # Whether we are processing train data
        self._cols_dtype_dict = None

        self.features = [
            "item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI", "sell_price",
            # demand features.
            "shift_t28", "shift_t29", "shift_t30", "rolling_std_t7", "rolling_std_t30", "rolling_std_t60", "rolling_std_t90", "rolling_std_t180", "rolling_mean_t7", "rolling_mean_t30",
            "rolling_mean_t60", "rolling_mean_t90", "rolling_mean_t180", "rolling_skew_t30", "rolling_kurt_t30", 
            # price features
            "price_change_t1", "price_change_t365", "rolling_price_std_t7", "rolling_price_std_t30", 
            # time features.
            "year", "month", "week", "day", "dayofweek", "is_year_end", "is_year_start", "is_quarter_end", "is_quarter_start", "is_month_end", "is_month_start", "is_weekend"
        ]
                                          
    def fit(self, X, y):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        if self.keep_last_train_days > 0:
            tmp = X[["date"]]
            tmp["date"] = pd.to_datetime(tmp["date"])
            earliest_date = tmp["date"].max() - timedelta(days = self.keep_last_train_days)
            self._last_train_rows = X.loc[tmp["date"] > earliest_date]

        self._is_train_data = True
        
        return self
    
    def transform(self, X):
        """
        This method is called transform the data given in argument.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
                
        Returns
        -------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
        """
                            
        st = time.time()
        print("Preprocessing data...")   
        print("X.shape:", X.shape)

        # If doing predictions, append the train rows at the beginning
        if self._last_train_rows is not None and not self._is_train_data:
            print("self._last_train_rows.shape:", self._last_train_rows.shape)
            self._orig_earliest_date = pd.to_datetime(X["date"]).min()
            X = pd.concat([self._last_train_rows, X], axis = 0).reset_index(drop = True)
                
        # Rolling demand features
        for diff in [0, 1, 2]:
            shift = self.test_days + diff
            X[f"shift_t{shift}"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(shift))
            
        for size in [7, 30, 60, 90, 180]:
            X[f"rolling_std_t{size}"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(self.test_days).rolling(size).std())
            
        for size in [7, 30, 60, 90, 180]:
            X[f"rolling_mean_t{size}"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(self.test_days).rolling(size).mean())
            
        X["rolling_skew_t30"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(self.test_days).rolling(30).skew())
        X["rolling_kurt_t30"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(self.test_days).rolling(30).kurt())
        
        # Price features
        X["shift_price_t1"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1))
        X["price_change_t1"] = (X["shift_price_t1"] - X["sell_price"]) / (X["shift_price_t1"])
        X["rolling_price_max_t365"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1).rolling(365).max())
        X["price_change_t365"] = (X["rolling_price_max_t365"] - X["sell_price"]) / (X["rolling_price_max_t365"])
        
        X["rolling_price_std_t7"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(7).std())
        X["rolling_price_std_t30"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(30).std())
   
        X.drop(["rolling_price_max_t365", "shift_price_t1"], axis = 1, inplace = True)
        
        # Time-related features
        X[self.dt_col] = pd.to_datetime(X[self.dt_col])
        attrs = ["year", "quarter", "month", "week", "day", "dayofweek", "is_year_end", "is_year_start", "is_quarter_end", "is_quarter_start", "is_month_end", "is_month_start"]

        for attr in attrs:
            dtype = np.int16 if attr == "year" else np.int8
            X[attr] = getattr(X[self.dt_col].dt, attr).astype(dtype)

        X["is_weekend"] = X["dayofweek"].isin([5, 6]).astype(np.int8)
           
        if self._last_train_rows is not None and not self._is_train_data:
            X = X.loc[pd.to_datetime(X["date"]) >= self._orig_earliest_date]

            # Optimizations
            for col in X.columns.tolist():
                X[col] = X[col].astype(self._cols_dtype_dict[col])
        else:  
            # Optimizations
            X = DataLoader.reduce_mem_usage(X, "data_df") # Need to take same dtypes as train
            self._cols_dtype_dict = {col: X[col].dtype for col in X.columns.tolist()}
            self._is_train_data = False

        print("out X.shape:", X.shape)
    
        print("Preprocessing data... done in", round(time.time() - st, 3), "secs")
        
        return X
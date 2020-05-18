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
# Date: 2020-04-20                                                            #
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

class PreprocessingStep3(BaseEstimator, TransformerMixin):
    """
    This class defines the first preprocessing step.
    """

    def __init__(self, dt_col, keep_last_train_days = 0):
        """
        This is the class' constructor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.dt_col = dt_col
        self.keep_last_train_days = keep_last_train_days # Number of rows at the end of the train set to keep for appending at the beginning of predict data
        self._last_train_rows = None
        self._orig_earliest_date = None
        self._is_train_data = False # Whether we are processing train data
        self._cols_dtype_dict = None

    def _max_consecutive_nonzero(self, x):
        a_ext = np.concatenate(( [0], x, [0] ))
        idx = np.flatnonzero(a_ext[1:] != a_ext[:-1])
        a_ext[1:][idx[1::2]] = idx[::2] - idx[1::2]
        return a_ext.cumsum()[1:-1].max()

    def _max_consecutive_nonzero_at_end(self, x):
        a_ext = np.concatenate(( [0], x, [0] ))
        idx = np.flatnonzero(a_ext[1:] != a_ext[:-1])
        a_ext[1:][idx[1::2]] = idx[::2] - idx[1::2]
        a_cum = a_ext.cumsum()
        return int(a_cum[1:-1].max() == a_cum[-2])
                                                  
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

        # If doing predictions, append the train rows at the beginning
        if self._last_train_rows is not None and not self._is_train_data:
            self._orig_earliest_date = pd.to_datetime(X["date"]).min()
            X = pd.concat([self._last_train_rows, X], axis = 0).reset_index(drop = True)
                
        # Doing feature engineering
        lags = [7, 28]
        lag_cols = [f"lag_{lag}" for lag in lags]
        for lag, lag_col in zip(lags, lag_cols):
            X[lag_col] = X[["id", "shifted_demand"]].groupby("id")["shifted_demand"].shift(lag - 1)

        wins = [7, 28]
        for win in wins:
            for lag,lag_col in zip(lags, lag_cols):
                X[f"rmean_{lag}_{win}"] = X[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
                #X[f"rstd_{lag}_{win}"] = X[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).std())
                #X[f"rmax_{lag}_{win}"] = X[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).max())
                        
        """for lag, lag_col in zip(lags, lag_cols):
            X["monthly_demand_by_id_" + str(lag) + "_days_ago"] = X.groupby(["id", X["date"].dt.to_period("M")])[lag_col].cumsum()
            X["weekly_demand_by_id_" + str(lag) + "_days_ago"] = X.groupby(["id", X["date"].dt.to_period("W")])[lag_col].cumsum()"""

        date_features = {
            "wday": "weekday",
            "week": "weekofyear",
            "month": "month",
            "quarter": "quarter",
            "year": "year",
            "mday": "day",
        }
    
        for date_feat_name, date_feat_func in date_features.items():
            if date_feat_name in X.columns:
                X[date_feat_name] = X[date_feat_name].astype("int16")
            else:
                X[date_feat_name] = getattr(X["date"].dt, date_feat_func).astype("int16")

        """
        # Price features
        X["shift_price_t1"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1))
        X["price_change_t1"] = ((X["shift_price_t1"] - X["sell_price"]) / (X["shift_price_t1"])).fillna(0)
        X.drop(["shift_price_t1"], axis = 1, inplace = True)
        
        X["rolling_price_std_t7"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(7).std())
        X["rolling_price_std_t30"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(30).std())
        """

        X.drop(["event_type_2", "event_name_2"], axis = 1, inplace = True)
                
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
                
        print("Preprocessing data... done in", round(time.time() - st, 3), "secs")
        
        return X
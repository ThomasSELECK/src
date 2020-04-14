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

class PreprocessingStep2(BaseEstimator, TransformerMixin):
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

        self.test_days = 0 #test_days
        self.dt_col = dt_col
        self.keep_last_train_days = keep_last_train_days # Number of rows at the end of the train set to keep for appending at the beginning of predict data
        self._last_train_rows = None
        self._orig_earliest_date = None
        self._is_train_data = False # Whether we are processing train data
        self._cols_dtype_dict = None
                                                          
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
                
        return self
    
    def transform(self, X, start_date = None, end_date = None):
        """
        This method is called transform the data given in argument.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.

        start_date: datetime object (default = None)
                If not None, the data whose date is prior `start_date`will be discarded.

        end_date: datetime object (default = None)
                If not None, the data whose date is posterior `end_date`will be discarded.
                
        Returns
        -------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
        """
                            
        st = time.time()
        print("Preprocessing data...")
        
        # Clip data to required time window if needed
        if start_date is not None:
            X = X[X["date"] >= start_date]

        if end_date is not None:
            X = X[X["date"] <= end_date]

        # Doing feature engineering
        lags = [7, 28]
        lag_cols = [f"lag_{lag}" for lag in lags]
        for lag, lag_col in zip(lags, lag_cols):
            X[lag_col] = X[["id", "sales"]].groupby("id")["sales"].shift(lag)

        wins = [7, 28]
        for win in wins:
            for lag,lag_col in zip(lags, lag_cols):
                X[f"rmean_{lag}_{win}"] = X[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

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
                
        print("Preprocessing data... done in", round(time.time() - st, 3), "secs")
        
        return X
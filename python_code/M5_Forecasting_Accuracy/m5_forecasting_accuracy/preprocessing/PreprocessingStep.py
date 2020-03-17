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

from m5_forecasting_accuracy.data_loading.data_loader import DataLoader

class PreprocessingStep(BaseEstimator, TransformerMixin):
    """
    This class defines the first preprocessing step.
    """

    def __init__(self, test_days, dt_col):
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

        # Optimizations
        X = DataLoader.reduce_mem_usage(X, "data_df")
        X.sort_values("date", inplace = True)
    
        print("Preprocessing data... done in", round(time.time() - st, 3), "secs")
        
        return X
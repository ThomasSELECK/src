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

class PreprocessingStep(BaseEstimator, TransformerMixin):
    """
    This class defines the first preprocessing step.
    """

    def __init__(self):
        """
        This is the class' constructor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        pass
                                          
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
        X["lag_t28"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(28))
        X["lag_t29"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(29))
        X["lag_t30"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(30))
        X["rolling_mean_t7"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(7).mean())
        X["rolling_std_t7"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(7).std())
        X["rolling_mean_t30"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(30).mean())
        X["rolling_mean_t90"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(90).mean())
        X["rolling_mean_t180"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(180).mean())
        X["rolling_std_t30"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(30).std())
        X["rolling_skew_t30"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(30).skew())
        X["rolling_kurt_t30"] = X.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(30).kurt())
        
        # Price features
        X["lag_price_t1"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1))
        X["price_change_t1"] = (X["lag_price_t1"] - X["sell_price"]) / (X["lag_price_t1"])
        X["rolling_price_max_t365"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1).rolling(365).max())
        X["price_change_t365"] = (X["rolling_price_max_t365"] - X["sell_price"]) / (X["rolling_price_max_t365"])
        X["rolling_price_std_t7"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(7).std())
        X["rolling_price_std_t30"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(30).std())
        X.drop(["rolling_price_max_t365", "lag_price_t1"], inplace = True, axis = 1)
    
        # Time features
        X["date"] = pd.to_datetime(X["date"])
        X["year"] = X["date"].dt.year
        X["month"] = X["date"].dt.month
        X["week"] = X["date"].dt.week
        X["day"] = X["date"].dt.day
        X["dayofweek"] = X["date"].dt.dayofweek
    
        print("Preprocessing data... done in", round(time.time() - st, 3), "secs")
        
        return X
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
# Date: 2020-05-23                                                            #
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
import multiprocessing as mp
from joblib import Parallel, delayed

from m5_forecasting_accuracy.data_loading.data_loader import DataLoader

class PreprocessingStep4(BaseEstimator, TransformerMixin):
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

        self.num_cores = 8 #mp.cpu_count()

        self.dt_col = dt_col
        self.keep_last_train_days = keep_last_train_days # Number of rows at the end of the train set to keep for appending at the beginning of predict data
        self._last_train_rows = None
        self._orig_earliest_date = None
        self._is_train_data = False # Whether we are processing train data
        self._cols_dtype_dict = None

    def generate_lags(self, data_df, d_shift, d_window, agg_type = "mean"):
        if agg_type == "mean":
            res =  data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)
            res = res.rename("rolling_mean_" + str(d_shift) + "_" + str(d_window))
        elif agg_type == "std":
            res = data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift).rolling(d_window).std()).astype(np.float16)
            res = res.rename("rolling_std_" + str(d_shift) + "_" + str(d_window))
        elif agg_type == "sum":
            res = data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift).rolling(d_window).sum()).astype(np.float16)
            res = res.rename("rolling_sum_" + str(d_shift) + "_" + str(d_window))

        return res

    def create_lag_features(self, X):
        features_args_lst = [(28, i, "mean") for i in [7, 14, 30, 60, 180]] 
        features_args_lst += [(28, i, "std") for i in [7, 14, 30, 60, 180]]
        features_args_lst += [(28, i, "sum") for i in [7, 14, 30, 60, 180]]
        features_args_lst += [(d_shift, d_window, "mean") for d_shift in [1, 7, 14] for d_window in [7, 14, 30, 60]]
        data_df = X[["id", "shifted_demand"]].copy()

        lag_features_df = pd.concat(Parallel(n_jobs = self.num_cores, max_nbytes = None)(delayed(self.generate_lags)(data_df.copy(), d_shift, d_window, agg_type) for d_shift, d_window, agg_type in features_args_lst), axis = 1)
        X = pd.concat([X, lag_features_df], axis = 1)
            
        return X

    def create_prices_features(self, X):
        # We can do some basic aggregations
        X["price_max"] = X.groupby(["store_id", "item_id"])["sell_price"].transform("max")
        X["price_min"] = X.groupby(["store_id", "item_id"])["sell_price"].transform("min")
        X["price_std"] = X.groupby(["store_id", "item_id"])["sell_price"].transform("std")
        X["price_mean"] = X.groupby(["store_id", "item_id"])["sell_price"].transform("mean")

        # and do price normalization (min/max scaling)
        X["price_norm"] = X["sell_price"] / X["price_max"]

        # Some items are can be inflation dependent
        # and some items are very "stable"
        X["price_nunique"] = X.groupby(["store_id", "item_id"])["sell_price"].transform("nunique")
        X["item_nunique"] = X.groupby(["store_id", "sell_price"])["item_id"].transform("nunique")

        # Now we can add price "momentum" (some sort of)
        # Shifted by week 
        # by month mean
        # by year mean
        X["price_momentum"] = X["sell_price"] / X.groupby(["store_id", "item_id"])["sell_price"].transform(lambda x: x.shift(1))
        X["price_momentum_m"] = X["sell_price"] / X.groupby(["store_id", "item_id", "month"])["sell_price"].transform("mean")
        X["price_momentum_y"] = X["sell_price"] / X.groupby(["store_id", "item_id", "year"])["sell_price"].transform("mean")

        """
        # Price features
        X["shift_price_t1"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1))
        X["price_change_t1"] = (X["shift_price_t1"] - X["sell_price"]) / (X["shift_price_t1"])
        X["rolling_price_max_t365"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1).rolling(365).max())
        X["price_change_t365"] = (X["rolling_price_max_t365"] - X["sell_price"]) / (X["rolling_price_max_t365"])
        
        X["rolling_price_std_t7"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(7).std())
        X["rolling_price_std_t30"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(30).std())
   
        X.drop(["rolling_price_max_t365", "shift_price_t1"], axis = 1, inplace = True)

        X = DataLoader.reduce_mem_usage(X, "data_df") # Need to take same dtypes as train
        """

        return X

    def generate_target_encodings(self, data_df, cols, agg_type = "mean"):
        col_name = "_" + "_".join(cols) + "_"

        if agg_type == "mean":
            res = data_df.groupby(cols)["shifted_demand"].transform("mean").astype(np.float16)
            res = res.rename("enc" + col_name + "mean")
        elif agg_type == "std":
            res = data_df.groupby(cols)["shifted_demand"].transform("std").astype(np.float16)
            res = res.rename("enc" + col_name + "std")

        return res

    def create_target_encoding_features(self, X):
        group_ids = [
            ["state_id"],
            ["store_id"],
            ["cat_id"],
            ["dept_id"],
            ["state_id", "cat_id"],
            ["state_id", "dept_id"],
            ["store_id", "cat_id"],
            ["store_id", "dept_id"],
            ["item_id"],
            ["item_id", "state_id"],
            ["item_id", "store_id"]
        ]

        data_df = X[["id", "shifted_demand", "dept_id", "state_id", "cat_id", "item_id", "store_id"]].copy()
        features_args_lst = [(cols, "mean") for cols in group_ids] 
        features_args_lst += [(cols, "std") for cols in group_ids]

        target_encoding_features_df = pd.concat(Parallel(n_jobs = self.num_cores, max_nbytes = None)(delayed(self.generate_target_encodings)(data_df.copy(), cols, agg_type) for cols, agg_type in features_args_lst), axis = 1)
        X = pd.concat([X, target_encoding_features_df], axis = 1)

        return X
                                                  
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
                
        print("    Creating prices features...")
        X = self.create_prices_features(X)

        print("    Creating lag features...")
        X = self.create_lag_features(X)

        lag_days_lst = [col for col in range(28, 28 + 15)]
        for l in lag_days_lst:
            training_set_df["sales_lag_" + str(l)] = training_set_df.groupby("id")["shifted_demand"].transform(lambda x: x.shift(l - 1))
        
        #print("    Creating target encoding features...")
        X = self.create_target_encoding_features(X)

        # Make some features from date
        X["tm_d"] = X["date"].dt.day.astype(np.int8)
        X["tm_w"] = X["date"].dt.week.astype(np.int8)
        X["tm_m"] = X["date"].dt.month.astype(np.int8)
        X["tm_y"] = X["date"].dt.year
        X["tm_y"] = (X["tm_y"] - X["tm_y"].min()).astype(np.int8)
        X["tm_wm"] = X["tm_d"].apply(lambda x: ceil(x / 7)).astype(np.int8)
        X["tm_dw"] = X["date"].dt.dayofweek.astype(np.int8)
        X["tm_w_end"] = (X["tm_dw"] >= 5).astype(np.int8)
                
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

        X = X[["item_id", "dept_id", "cat_id", "release", "sell_price", "price_max", "price_min", "price_std", "price_mean", "price_norm", "price_nunique", "item_nunique", "price_momentum", "price_momentum_m", "price_momentum_y", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI", "tm_d", "tm_w", "tm_m", "tm_y", "tm_wm", "tm_dw", "tm_w_end", "enc_cat_id_mean", "enc_cat_id_std", "enc_dept_id_mean", "enc_dept_id_std", "enc_item_id_mean", "enc_item_id_std", "sales_lag_28", "sales_lag_29", "sales_lag_30", "sales_lag_31", "sales_lag_32", "sales_lag_33", "sales_lag_34", "sales_lag_35", "sales_lag_36", "sales_lag_37", "sales_lag_38", "sales_lag_39", "sales_lag_40", "sales_lag_41", "sales_lag_42", "rolling_mean_28_7", "rolling_std_28_7", "rolling_mean_28_14", "rolling_std_28_14", "rolling_mean_28_30", "rolling_std_28_30", "rolling_mean_28_60", "rolling_std_28_60", "rolling_mean_28_180", "rolling_std_28_180", "rolling_mean_1_7", "rolling_mean_1_14", "rolling_mean_1_30", "rolling_mean_1_60", "rolling_mean_7_7", "rolling_mean_7_14", "rolling_mean_7_30", "rolling_mean_7_60", "rolling_mean_14_7", "rolling_mean_14_14", "rolling_mean_14_30", "rolling_mean_14_60"]]
        
        return X
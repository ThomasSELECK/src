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

        self.num_cores = 5 #mp.cpu_count()

        self.dt_col = dt_col
        self.keep_last_train_days = keep_last_train_days # Number of rows at the end of the train set to keep for appending at the beginning of predict data
        self._last_train_rows = None
        self._orig_earliest_date = None
        self._is_train_data = False # Whether we are processing train data
        self._cols_dtype_dict = None

    def generate_lags(self, data_df, d_shift, d_window, agg_type = "mean"):
        if agg_type == "mean":
            res =  data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float32)
            res = res.rename("rolling_mean_" + str(d_shift) + "_" + str(d_window))
        elif agg_type == "std":
            res = data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift).rolling(d_window).std()).astype(np.float32)
            res = res.rename("rolling_std_" + str(d_shift) + "_" + str(d_window))
        elif agg_type == "sum":
            res = data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift).rolling(d_window).sum()).astype(np.float32)
            res = res.rename("rolling_sum_" + str(d_shift) + "_" + str(d_window))
        elif agg_type == "shift":
            res = data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift)).astype(np.float32)
            res = res.rename("sales_lag_" + str(d_shift))

        return res

    def create_lag_features(self, X):
        features_args_lst = [(28, i, "mean") for i in [7, 14, 30, 60, 180]] 
        features_args_lst += [(28, i, "std") for i in [7, 14, 30, 60, 180]]
        features_args_lst += [(28, i, "sum") for i in [7, 14, 30, 60, 180]]
        features_args_lst += [(d_shift, d_window, "mean") for d_shift in [1, 7, 14] for d_window in [7, 14, 30, 60]]
        features_args_lst += [(d_shift, 0, "shift") for d_shift in range(28, 28 + 15)]
        data_df = X[["id", "shifted_demand"]].copy()

        lag_features_df = pd.concat(Parallel(n_jobs = self.num_cores, backend = "threading", max_nbytes = None)(delayed(self.generate_lags)(data_df.copy(), d_shift, d_window, agg_type) for d_shift, d_window, agg_type in features_args_lst), axis = 1)
        X = pd.concat([X, lag_features_df], axis = 1)
            
        return X

    def create_prices_features(self, X):
        prices_df = pd.read_csv("D:/Projets_Data_Science/Competitions/Kaggle/M5_Forecasting_Accuracy/data/raw/sell_prices.csv")
        calendar_df = pd.read_csv("D:/Projets_Data_Science/Competitions/Kaggle/M5_Forecasting_Accuracy/data/raw/calendar.csv")

        prices_df["price_max"] = prices_df.groupby(["store_id", "item_id"])["sell_price"].transform("max").astype(np.float32)
        prices_df["price_min"] = prices_df.groupby(["store_id", "item_id"])["sell_price"].transform("min").astype(np.float32)
        prices_df["price_std"] = prices_df.groupby(["store_id", "item_id"])["sell_price"].transform("std").astype(np.float32)
        prices_df["price_mean"] = prices_df.groupby(["store_id", "item_id"])["sell_price"].transform("mean").astype(np.float32)

        # and do price normalization (min/max scaling)
        prices_df["price_norm"] = (prices_df["sell_price"] / prices_df["price_max"]).astype(np.float32)

        # Some items are can be inflation dependent
        # and some items are very "stable"
        prices_df["price_nunique"] = prices_df.groupby(["store_id", "item_id"])["sell_price"].transform("nunique").astype(np.int32)
        prices_df["item_nunique"] = prices_df.groupby(["store_id", "sell_price"])["item_id"].transform("nunique").astype(np.int8)

        # I would like some "rolling" aggregations
        # but would like months and years as "window"
        calendar_prices = calendar_df[["wm_yr_wk", "month", "year"]]
        calendar_prices = calendar_prices.drop_duplicates(subset = ["wm_yr_wk"])
        prices_df = prices_df.merge(calendar_prices[["wm_yr_wk", "month", "year"]], on = ["wm_yr_wk"], how = "left")
        del calendar_prices

        # Now we can add price "momentum" (some sort of)
        # Shifted by week by month mean by year mean
        prices_df["price_momentum"] = (prices_df["sell_price"] / prices_df.groupby(["store_id", "item_id"])["sell_price"].transform(lambda x: x.shift(1))).astype(np.float32)
        prices_df["price_momentum_m"] = (prices_df["sell_price"] / prices_df.groupby(["store_id", "item_id", "month"])["sell_price"].transform("mean")).astype(np.float32)
        prices_df["price_momentum_y"] = (prices_df["sell_price"] / prices_df.groupby(["store_id", "item_id", "year"])["sell_price"].transform("mean")).astype(np.float32)

        prices_df.drop(["month", "year", "sell_price"], axis = 1, inplace = True)
        prices_df["store_id"] = prices_df["store_id"].astype("category")
        prices_df["item_id"] = prices_df["item_id"].astype("category")
        
        X = X.merge(prices_df, on = ["store_id", "item_id", "wm_yr_wk"], how = "left")

        #X = DataLoader.reduce_mem_usage(X, "data_df") # Need to take same dtypes as train

        return X

    def create_prices_features_old(self, X):
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
   
        X.drop(["rolling_price_max_t365", "shift_price_t1"], axis = 1, inplace = True)"""

        #X = DataLoader.reduce_mem_usage(X, "data_df") # Need to take same dtypes as train

        return X

    def create_target_encoding_features(self, X):
        target_encoding_group_ids = [
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

        data_df = X[["id", "d", "shifted_demand", "dept_id", "state_id", "cat_id", "item_id", "store_id"]].copy()
        data_df["d"] = data_df["d"].str.replace("d_", "").apply(lambda x: int(x))
        data_df = data_df.loc[data_df["d"] <= (1913 - 28)] # to be sure we don't have leakage in our validation set
        data_df.drop("d", axis = 1, inplace = True)

        features_args_lst = [(cols, "mean") for cols in target_encoding_group_ids] 
        features_args_lst += [(cols, "std") for cols in target_encoding_group_ids]

        target_encoding_features_lst = []
        for cols, agg_type in features_args_lst:
            col_name = "_" + "_".join(cols) + "_"

            if agg_type == "mean":
                res = data_df.groupby(cols)["shifted_demand"].mean().reset_index()
                res = res.rename(columns = {"shifted_demand": "enc" + col_name + "mean"})
            elif agg_type == "std":
                res = data_df.groupby(cols)["shifted_demand"].std().reset_index()
                res = res.rename(columns = {"shifted_demand": "enc" + col_name + "std"})
            target_encoding_features_lst.append((cols, res))

        return target_encoding_features_lst
                                                  
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

        print("    Creating target encoding features...")
        self.target_encoding_features_lst = self.create_target_encoding_features(X)
        
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
            X["d"] = X["d"].str.replace("d_", "").apply(lambda x: int(x))
            X.sort_values(["id", "d"], ascending = True, inplace = True)
            X = X.reset_index(drop = True)
        else:
            X["d"] = X["d"].str.replace("d_", "").apply(lambda x: int(x))
                
        print("    Creating prices features...")
        X = self.create_prices_features(X)

        print("    Creating lag features...")
        X = self.create_lag_features(X)
                        
        print("    Adding target encoding features...")
        for cols, target_features_df in self.target_encoding_features_lst:
            X = X.merge(target_features_df, how = "left", on = cols)

        # Make some features from date
        X["tm_d"] = X["date"].dt.day.astype(np.int8)
        X["tm_w"] = X["date"].dt.week.astype(np.int8)
        X["tm_m"] = X["date"].dt.month.astype(np.int8)
        X["tm_y"] = X["date"].dt.year.astype(np.int16)
        X["tm_wm"] = X["tm_d"].apply(lambda x: np.ceil(x / 7)).astype(np.int8)
        X["tm_dw"] = X["date"].dt.dayofweek.astype(np.int8)
        X["tm_w_end"] = (X["tm_dw"] >= 5).astype(np.int8)
                                        
        if self._last_train_rows is not None and not self._is_train_data:
            X = X.loc[pd.to_datetime(X["date"]) >= self._orig_earliest_date]

            # Optimizations
            """for col in X.columns.tolist():
                X[col] = X[col].astype(self._cols_dtype_dict[col])"""
        else:  
            # Optimizations
            #X = DataLoader.reduce_mem_usage(X, "data_df") # Need to take same dtypes as train

            """for col in ["snap_CA", "snap_TX", "snap_WI"]:
                X[col] = X[col].astype(np.int8)"""

            self._cols_dtype_dict = {col: X[col].dtype for col in X.columns.tolist()}
            self._is_train_data = False

        X = X[["id", "store_id", "d", "item_id", "dept_id", "cat_id", "release", "sell_price", "price_max", "price_min", "price_std", "price_mean", "price_norm", "price_nunique", "item_nunique", "price_momentum", "price_momentum_m", "price_momentum_y", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI", "tm_d", "tm_w", "tm_m", "tm_y", "tm_wm", "tm_dw", "tm_w_end", "enc_cat_id_mean", "enc_cat_id_std", "enc_dept_id_mean", "enc_dept_id_std", "enc_item_id_mean", "enc_item_id_std", "sales_lag_28", "sales_lag_29", "sales_lag_30", "sales_lag_31", "sales_lag_32", "sales_lag_33", "sales_lag_34", "sales_lag_35", "sales_lag_36", "sales_lag_37", "sales_lag_38", "sales_lag_39", "sales_lag_40", "sales_lag_41", "sales_lag_42", "rolling_mean_28_7", "rolling_std_28_7", "rolling_mean_28_14", "rolling_std_28_14", "rolling_mean_28_30", "rolling_std_28_30", "rolling_mean_28_60", "rolling_std_28_60", "rolling_mean_28_180", "rolling_std_28_180", "rolling_mean_1_7", "rolling_mean_1_14", "rolling_mean_1_30", "rolling_mean_1_60", "rolling_mean_7_7", "rolling_mean_7_14", "rolling_mean_7_30", "rolling_mean_7_60", "rolling_mean_14_7", "rolling_mean_14_14", "rolling_mean_14_30", "rolling_mean_14_60"]]
              
        print("Preprocessing data... done in", round(time.time() - st, 3), "secs")

        return X
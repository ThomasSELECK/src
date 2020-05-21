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
import multiprocessing as mp
from joblib import Parallel, delayed

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

    def generate_lags(self, data_df, d_shift, d_window, agg_type = "mean"):
        if agg_type == "mean":
            res =  data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift - 1).rolling(d_window).mean()).astype(np.float16)
            res = res.rename("rolling_mean_" + str(d_shift) + "_" + str(d_window))
        elif agg_type == "std":
            res = data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift - 1).rolling(d_window).std()).astype(np.float16)
            res = res.rename("rolling_std_" + str(d_shift) + "_" + str(d_window))
        elif agg_type == "sum":
            res = data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift - 1).rolling(d_window).sum()).astype(np.float16)
            res = res.rename("rolling_sum_" + str(d_shift) + "_" + str(d_window))

        return res

    def create_lag_features(self, X):
        features_args_lst = [(28, i, "mean") for i in [7, 14, 30, 60, 180]] 
        features_args_lst += [(28, i, "std") for i in [7, 14, 30, 60, 180]]
        features_args_lst += [(28, i, "sum") for i in [7, 14, 30, 60, 180]]
        features_args_lst += [(d_shift, d_window, "mean") for d_shift in [1, 7, 14] for d_window in [7, 14, 30, 60]]
        num_cores = mp.cpu_count()
        data_df = X[["id", "shifted_demand"]].copy()

        lag_features_df = pd.concat(Parallel(n_jobs = num_cores)(delayed(self.generate_lags)(data_df, d_shift, d_window, agg_type) for d_shift, d_window, agg_type in features_args_lst), axis = 1)
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

        # Price features
        X["shift_price_t1"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1))
        X["price_change_t1"] = (X["shift_price_t1"] - X["sell_price"]) / (X["shift_price_t1"])
        X["rolling_price_max_t365"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1).rolling(365).max())
        X["price_change_t365"] = (X["rolling_price_max_t365"] - X["sell_price"]) / (X["rolling_price_max_t365"])
        
        X["rolling_price_std_t7"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(7).std())
        X["rolling_price_std_t30"] = X.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(30).std())
   
        X.drop(["rolling_price_max_t365", "shift_price_t1"], axis = 1, inplace = True)

        X = DataLoader.reduce_mem_usage(X, "data_df") # Need to take same dtypes as train

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

        num_cores = mp.cpu_count()
        data_df = X[["id", "shifted_demand", "dept_id", "state_id", "cat_id", "item_id", "store_id"]].copy()
        features_args_lst = [(cols, "mean") for cols in group_ids] 
        features_args_lst += [(cols, "std") for cols in group_ids]

        target_encoding_features_df = pd.concat(Parallel(n_jobs = num_cores)(delayed(self.generate_target_encodings)(data_df, cols, agg_type) for cols, agg_type in features_args_lst), axis = 1)
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

        #print("    Creating target encoding features...")
        #X = self.create_target_encoding_features(X)
        print("Creating target encoding features DISABLED! To reduce memory usage.")

        """
        print("    Creating lags and rolling window features...")
        # Rolling demand features
        for shift in [0, 1, 2, 6, 7, 14]:
            X[f"shift_t{shift}"] = X.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(shift))
            
        for size in [7, 30, 60, 90, 180]:
            X[f"rolling_mean_t{size}"] = X.groupby(["id"])["shifted_demand"].transform(lambda x: x.rolling(size).mean())
            X[f"rolling_std_t{size}"] = X.groupby(["id"])["shifted_demand"].transform(lambda x: x.rolling(size).std())
            X[f"rolling_sum_t{size}"] = X.groupby(["id"])["shifted_demand"].transform(lambda x: x.rolling(size).sum())
            
        X["rolling_skew_t30"] = X.groupby(["id"])["shifted_demand"].transform(lambda x: x.rolling(30).skew())
        X["rolling_kurt_t30"] = X.groupby(["id"])["shifted_demand"].transform(lambda x: x.rolling(30).kurt())
        """

        # Time-related features
        X[self.dt_col] = pd.to_datetime(X[self.dt_col])
        attrs = ["year", "quarter", "month", "week", "day", "dayofweek", "is_year_end", "is_year_start", "is_quarter_end", "is_quarter_start", "is_month_end"]

        for attr in attrs:
            dtype = np.int16 if attr == "year" else np.int8
            X[attr] = getattr(X[self.dt_col].dt, attr).astype(dtype)

        X["is_weekend"] = X["dayofweek"].isin([5, 6]).astype(np.int8)
                
        ### New features
        """
        for size in [7, 30]:
            X[f"store_rolling_sum_t{size}"] = X.groupby(["store_id"])["shifted_demand"].transform(lambda x: x.rolling(size).sum())
            X[f"cat_rolling_sum_t{size}"] = X.groupby(["cat_id"])["shifted_demand"].transform(lambda x: x.rolling(size).sum())
            
        X["monthly_demand"] = X.groupby(pd.to_datetime(X["date"]).dt.to_period("M"))["shifted_demand"].cumsum()
        X["weekly_demand"] = X.groupby(pd.to_datetime(X["date"]).dt.to_period("W"))["shifted_demand"].cumsum()

        tmp = X[["id", "shifted_demand"]]
        tmp["shifted_demand"] = tmp["shifted_demand"].apply(lambda x: int(x == 0))

        for size in [7, 30]:
            X[f"nb_zeros_rolling_t{size}"] = tmp.groupby(["id"])["shifted_demand"].transform(lambda x: x.rolling(size).sum())

        # Bin sell_price
        X["binned_sell_price"] = 0
        X["binned_sell_price"].loc[X["sell_price"] <= 0.80] = 6
        X["binned_sell_price"].loc[(X["sell_price"] > 0.80) & (X["sell_price"] <= 1)] = 5
        X["binned_sell_price"].loc[(X["sell_price"] > 1) & (X["sell_price"] <= 10)] = 4
        X["binned_sell_price"].loc[(X["sell_price"] > 10) & (X["sell_price"] <= 20)] = 2
        X["binned_sell_price"].loc[(X["sell_price"] > 20) & (X["sell_price"] <= 40)] = 3
        X["binned_sell_price"].loc[X["sell_price"] > 40] = 1
        """

        X["monthly_demand_by_id"] = X.groupby(["id", pd.to_datetime(X["date"]).dt.to_period("M")])["shifted_demand"].cumsum()
        X["weekly_demand_by_id"] = X.groupby(["id", pd.to_datetime(X["date"]).dt.to_period("W")])["shifted_demand"].cumsum()
        X["monthly_demand_by_store"] = X.groupby(["store_id", pd.to_datetime(X["date"]).dt.to_period("M")])["shifted_demand"].cumsum()
        X["weekly_demand_by_store"] = X.groupby(["store_id", pd.to_datetime(X["date"]).dt.to_period("W")])["shifted_demand"].cumsum()
        X["monthly_demand_by_item_id"] = X.groupby(["item_id", pd.to_datetime(X["date"]).dt.to_period("M")])["shifted_demand"].cumsum()
        X["weekly_demand_by_item_id"] = X.groupby(["item_id", pd.to_datetime(X["date"]).dt.to_period("W")])["shifted_demand"].cumsum()

        # Max consecutive zeros length
        tmp = X[["id", "shifted_demand"]]
        tmp["shifted_demand"] = tmp["shifted_demand"].apply(lambda x: int(x == 0))
        tmp2 = tmp.groupby(["id"])["shifted_demand"].apply(self._max_consecutive_nonzero).reset_index()
        tmp3 = tmp.groupby(["id"])["shifted_demand"].apply(self._max_consecutive_nonzero_at_end).reset_index()
        tmp2 = tmp2.merge(tmp3, how = "left", on = "id")
        tmp2.columns = ["id", "max_consecutive_zeros", "max_consecutive_zeros_at_end"]
        X = X.merge(tmp2, how = "left", on = "id")

        X.drop(["is_quarter_start", "event_type_2", "is_month_end", "is_quarter_end", "event_name_2", "is_year_start", "max_consecutive_zeros_at_end"], axis = 1, inplace = True)
                
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
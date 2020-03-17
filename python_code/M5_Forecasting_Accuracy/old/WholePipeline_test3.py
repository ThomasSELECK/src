#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the M5 Forecasting Accuracy competition                  #
#                                                                             #
# This file contains all files paths of the datasets.                         #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2020-03-14                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import os
import time
import numpy as np
import pandas as pd
import pickle
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from datetime import datetime
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

from dev.files_paths import *
from m5_forecasting_accuracy.data_loading.data_loader import DataLoader
from m5_forecasting_accuracy.preprocessing.PreprocessingStep import PreprocessingStep
from m5_forecasting_accuracy.models.lightgbm_wrapper import LGBMRegressor

pd.set_option("display.max_columns", 100)

def create_dt(is_train = True, h = 28, max_lags = 400, tr_last = 1913):  
    prices = pd.read_csv(SELL_PRICES_PATH_str)
    cal = pd.read_csv(CALENDAR_PATH_str)
    cal.drop("weekday", axis = 1, inplace = True)
    cal["date"] = pd.to_datetime(cal["date"], format = "%Y-%m-%d")
  
    if is_train:
        dt = pd.read_csv(SALES_TRAIN_PATH_str)
    else:
        dt = pd.read_csv(SALES_TRAIN_PATH_str)
        dt.drop(["d_" + str(i) for i in range(1, tr_last - max_lags + 1)], axis = 1, inplace = True)
        for c in ["d_" + str(i) for i in range(tr_last + 1, tr_last + 2 * h + 1)]:
            dt[c] = np.nan

    dt = pd.melt(dt, id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name = "d", value_name = "sales")
    dt = pd.merge(dt, cal[["d", "date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]], how = "left", on = "d")
    dt = dt.merge(prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]], on = ["store_id", "item_id", "wm_yr_wk"], how = "left")

    return dt

def create_fea(dt):
    for lag in [7, 28, 29]:
        feature = "lag_" + str(lag)
        dt[feature] = dt.groupby(["id"])["sales"].transform(lambda x: x.shift(lag))

    for win in [7, 30, 90, 180]:
        feature = "roll_mean_28_" + str(win)
        dt[feature] = dt.groupby(["id"])["lag_28"].transform(lambda x: x.rolling(win).mean())

    for win in [7, 30]:
        feature = "roll_max_28_" + str(win)
        dt[feature] = dt.groupby(["id"])["lag_28"].transform(lambda x: x.rolling(win).max())
        
    dt["price_change_1"] = dt.groupby(["id"])["sell_price"].transform(lambda x: x / x.shift(1) - 1)
    dt["price_change_365"] = dt.groupby(["id"])["sell_price"].transform(lambda x: x / x.shift(1).rolling(365).max() - 1)
    
    dt["event_name_1"].fillna("unknown", inplace = True)
            
    cat = ["item_id", "state_id", "dept_id", "cat_id", "event_name_1"]
    for feature in cat:
        encoder = LabelEncoder()
        dt[feature] = encoder.fit_transform(dt[feature])
  
    dt.drop(["store_id", "d", "wm_yr_wk"], axis = 1, inplace = True)

    dt["wday"] = dt["date"].dt.weekday
    dt["mday"] = dt["date"].dt.day
    dt["week"] = dt["date"].dt.week
    dt["month"] = dt["date"].dt.month
    dt["year"] = dt["date"].dt.year

    return dt


# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
        
    enable_validation = True

    h = 28 
    max_lags = 400
    tr_last = 1913
    fday = "2016-04-25"
    df_sales = pd.read_csv(SALES_TRAIN_PATH_str)

    print("Creating training set with features...")

    tr = create_dt()
    gc.collect()

    tr = create_fea(tr)
    gc.collect()

    tr.dropna(inplace = True)
    y = tr["sales"]

    idx = tr.loc[tr["date"] <= tr["date"].max() - np.timedelta64(28, "D")].index

    tr.drop(["id", "sales", "date"], axis = 1, inplace = True)
    gc.collect()
    
    print("Constructing training and validation sets for GBM...")
    
    cats = ["item_id", "state_id", "dept_id", "cat_id", "wday", "mday", "week", "month", "year", "snap_CA", "snap_TX", "snap_WI"]

    xtr = lgb.Dataset(tr.loc[tr.index.isin(idx)], y.loc[y.index.isin(idx)], categorical_feature = cats)
    xval = lgb.Dataset(tr.loc[~tr.index.isin(idx)], y.loc[~y.index.isin(idx)], categorical_feature = cats)

    print("Training model...\n")

    lgb_params = {
        "boosting_type": "gbdt",
        "metric": "rmse",
        "objective": "regression",
        "n_jobs": -1,
        "learning_rate": 0.05,
        "bagging_fraction": 0.75,
        "bagging_freq": 10, 
        "colsample_bytree": 0.75,
        "max_depth": -1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "device": "cpu",
        "verbosity": -1
    }
    
    m_lgb = lgb.train(lgb_params, xtr, num_boost_round = 2500, early_stopping_rounds = 400, valid_sets = [xtr, xval], verbose_eval = 100)
        
    te = create_dt(False)

    days_lst = ["2016-04-25", "2016-04-26", "2016-04-27", "2016-04-28", "2016-04-29", "2016-04-30", "2016-05-01", "2016-05-02", "2016-05-03", "2016-05-04", "2016-05-05", 
                "2016-05-06", "2016-05-07", "2016-05-08", "2016-05-09", "2016-05-10", "2016-05-11", "2016-05-12", "2016-05-13", "2016-05-14", "2016-05-15", "2016-05-16", 
                "2016-05-17", "2016-05-18", "2016-05-19", "2016-05-20", "2016-05-21", "2016-05-22", "2016-05-23", "2016-05-24", "2016-05-25", "2016-05-26", "2016-05-27",
                "2016-05-28", "2016-05-29", "2016-05-30", "2016-05-31", "2016-06-01", "2016-06-02", "2016-06-03", "2016-06-04", "2016-06-05", "2016-06-06", "2016-06-07", 
                "2016-06-08", "2016-06-09", "2016-06-10", "2016-06-11", "2016-06-12", "2016-06-13", "2016-06-14", "2016-06-15", "2016-06-16", "2016-06-17", "2016-06-18", 
                "2016-06-19"]

    for day in days_lst:
        print(day)

        tst = te.loc[(te["date"] >= datetime.strptime(day, "%Y-%m-%d") - timedelta(days = max_lags)) & (te["date"] <= datetime.strptime(day, "%Y-%m-%d"))]
        tst = create_fea(tst)
        tst = tst.loc[tst["date"] == day]
        tst.drop(["id", "sales", "date"], axis = 1, inplace = True)
        te["sales"].loc[te["date"] == day] = m_lgb.predict(tst)

    tmp = te.loc[te["date"] >= fday]
    tmp["id"].loc[te["date"] >= datetime.strptime(fday, "%Y-%m-%d") + timedelta(days = h)] = tmp["id"].loc[te["date"] >= datetime.strptime(fday, "%Y-%m-%d") + timedelta(days = h)].str.replace("validation", "evaluation")

    predictions = tmp[["id", "d", "sales"]]
    predictions["d"] = predictions["d"].apply(lambda x: "F" + str(((int(x.replace("d_", "")) - 1914) % 28) + 1))
    predictions = predictions.set_index(["id", "d"])["sales"].unstack().reset_index()
    predictions = predictions[["id"] + ["F" + str(i) for i in range(1, 29)]]

    predictions2 = predictions.groupby("id")["d"].apply(lambda x: pd.Series(["F" + str(i) for i in range(1, x.shape[0] + 1)])).reset_index().drop("level_1", axis = 1)
    predictions.drop("d", axis = 1, inplace = True)
    predictions = pd.pivot(predictions, index = "id", columns = "date", values = "sales").reset_index()
    predictions.columns = ["id"] + ["F" + str(i + 1) for i in range(28)]
    predictions.to_csv(PREDICTIONS_DIRECTORY_PATH_str + "sub_dt_lgb_v3.csv", index = False)
    
    # Stop the timer and print the exectution time
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")

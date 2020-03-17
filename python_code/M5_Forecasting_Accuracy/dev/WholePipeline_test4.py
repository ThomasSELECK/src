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
from m5_forecasting_accuracy.model_utils.CustomTimeSeriesSplitter import CustomTimeSeriesSplitter

pd.set_option("display.max_columns", 100)

def add_demand_features(df, DAYS_PRED):
    for diff in [0, 1, 2]:
        shift = DAYS_PRED + diff
        df[f"shift_t{shift}"] = df.groupby(["id"])["demand"].transform(lambda x: x.shift(shift))

    for size in [7, 30, 60, 90, 180]:
        df[f"rolling_std_t{size}"] = df.groupby(["id"])["demand"].transform(lambda x: x.shift(DAYS_PRED).rolling(size).std())

    for size in [7, 30, 60, 90, 180]:
        df[f"rolling_mean_t{size}"] = df.groupby(["id"])["demand"].transform(lambda x: x.shift(DAYS_PRED).rolling(size).mean())

    df["rolling_skew_t30"] = df.groupby(["id"])["demand"].transform(lambda x: x.shift(DAYS_PRED).rolling(30).skew())
    df["rolling_kurt_t30"] = df.groupby(["id"])["demand"].transform(lambda x: x.shift(DAYS_PRED).rolling(30).kurt())

    return df


def add_price_features(df):
    df["shift_price_t1"] = df.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1))
    df["price_change_t1"] = (df["shift_price_t1"] - df["sell_price"]) / (df["shift_price_t1"])
    df["rolling_price_max_t365"] = df.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1).rolling(365).max())
    df["price_change_t365"] = (df["rolling_price_max_t365"] - df["sell_price"]) / (df["rolling_price_max_t365"])

    df["rolling_price_std_t7"] = df.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(7).std())
    df["rolling_price_std_t30"] = df.groupby(["id"])["sell_price"].transform(lambda x: x.rolling(30).std())

    return df.drop(["rolling_price_max_t365", "shift_price_t1"], axis=1)


def add_time_features(df, dt_col):
    df[dt_col] = pd.to_datetime(df[dt_col])
    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
        "is_year_end",
        "is_year_start",
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    return df

def plot_cv_indices(cv, X, y, dt_col, lw=10):
    n_splits = cv.get_n_splits()
    _, ax = plt.subplots(figsize=(20, n_splits))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            X[dt_col],
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.coolwarm,
            vmin=-0.2,
            vmax=1.2,
        )

    # Formatting
    MIDDLE = 15
    LARGE = 20
    ax.set_xlabel("Datetime", fontsize=LARGE)
    ax.set_xlim([X[dt_col].min(), X[dt_col].max()])
    ax.set_ylabel("CV iteration", fontsize=LARGE)
    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels(list(range(n_splits)))
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=MIDDLE)
    ax.set_title("{}".format(type(cv).__name__), fontsize=LARGE)
    return ax

def train_lgb(bst_params, fit_params, X, y, cv, drop_when_train=None):
    models = []

    if drop_when_train is None:
        drop_when_train = []

    for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y)):
        print(f"\n---------- Fold: ({idx_fold + 1} / {cv.get_n_splits()}) ----------\n")

        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]
        train_set = lgb.Dataset(X_trn.drop(drop_when_train, axis=1), label=y_trn)
        val_set = lgb.Dataset(X_val.drop(drop_when_train, axis=1), label=y_val)

        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            **fit_params,
        )
        models.append(model)

        del idx_trn, idx_val, X_trn, X_val, y_trn, y_val
        gc.collect()

    return models

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def make_submission(test, submission):
    preds = test[["id", "date", "demand"]]
    preds = preds.pivot(index="id", columns="date", values="demand").reset_index()
    preds.columns = ["id"] + ["F" + str(d + 1) for d in range(DAYS_PRED)]

    evals = submission[submission["id"].str.endswith("evaluation")]
    vals = submission[["id"]].merge(preds, how="inner", on="id")
    final = pd.concat([vals, evals])

    assert final.drop("id", axis=1).isnull().sum().sum() == 0
    assert final["id"].equals(submission["id"])

    final.to_csv(PREDICTIONS_DIRECTORY_PATH_str + "submission_kaggle_17032020.csv", index=False)

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
        
    enable_validation = True

    if not os.path.exists("E:/M5_Forecasting_Accuracy_cache/cached_data.pkl"):
        dl = DataLoader()
        data, sample_submission_df = dl.load_data(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, enable_validation = True)

        DAYS_PRED = sample_submission_df.shape[1] - 1  # 28
        dt_col = "date"

        data = add_demand_features(data, DAYS_PRED)
        data = add_price_features(data)
        data = add_time_features(data, dt_col)
        data = dl.reduce_mem_usage(data, "data")
        data = data.sort_values("date")

        print("start date:", data[dt_col].min())
        print("end date:", data[dt_col].max())
        print("data shape:", data.shape)

        cv_params = {
            "n_splits": 3,
            "train_days": 365 * 2,
            "test_days": DAYS_PRED,
            "dt_col": dt_col,
        }

        cv = CustomTimeSeriesSplitter(**cv_params)
        # Plotting all the points takes long time.
        #plot_cv_indices(cv, data.iloc[::1000][[dt_col]].reset_index(drop=True), None, dt_col)

        features = [
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
            "sell_price",
            # demand features.
            "shift_t28",
            "shift_t29",
            "shift_t30",
            "rolling_std_t7",
            "rolling_std_t30",
            "rolling_std_t60",
            "rolling_std_t90",
            "rolling_std_t180",
            "rolling_mean_t7",
            "rolling_mean_t30",
            "rolling_mean_t60",
            "rolling_mean_t90",
            "rolling_mean_t180",
            "rolling_skew_t30",
            "rolling_kurt_t30",
            # price features
            "price_change_t1",
            "price_change_t365",
            "rolling_price_std_t7",
            "rolling_price_std_t30",
            # time features.
            "year",
            "month",
            "week",
            "day",
            "dayofweek",
            "is_year_end",
            "is_year_start",
            "is_quarter_end",
            "is_quarter_start",
            "is_month_end",
            "is_month_start",
            "is_weekend",
        ]

        # prepare training and test data.
        # 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913
        # 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)
        # 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)

        mask = data["date"] <= "2016-04-24"

        # Attach "date" to X_train for cross validation.
        X_train = data[mask][["date"] + features].reset_index(drop=True)
        y_train = data[mask]["demand"].reset_index(drop=True)
        X_test = data[~mask][features].reset_index(drop=True)

        # keep these two columns to use later.
        id_date = data[~mask][["id", "date"]].reset_index(drop=True)

        del data
        gc.collect()

        with open("E:/M5_Forecasting_Accuracy_cache/cached_data.pkl", "wb") as f:
            pickle.dump((X_train, X_test, y_train, cv, dt_col, id_date, sample_submission_df), f)
    else:
        with open("E:/M5_Forecasting_Accuracy_cache/cached_data.pkl", "rb") as f:
            X_train, X_test, y_train, cv, dt_col, id_date, sample_submission_df = pickle.load(f)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    bst_params = {
        "boosting_type": "gbdt",
        "metric": "rmse",
        "objective": "regression",
        "n_jobs": -1,
        "seed": 42,
        "learning_rate": 0.05,
        "bagging_fraction": 0.75,
        "bagging_freq": 10,
        "colsample_bytree": 0.55,
        "max_depth": -1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbosity": -1
    }

    fit_params = {
        "num_boost_round": 10000,
        "early_stopping_rounds": 100,
        "verbose_eval": 100,
    }

    models = train_lgb(bst_params, fit_params, X_train, y_train, cv, drop_when_train=[dt_col])

    del X_train, y_train
    gc.collect()

    imp_type = "gain"
    importances = np.zeros(X_test.shape[1])
    preds = np.zeros(X_test.shape[0])

    features_names = []
    for model in models:
        preds += model.predict(X_test)
        importances += model.feature_importance(imp_type)
        features_names = model.feature_name()

    preds = preds / cv.get_n_splits()
    make_submission(id_date.assign(demand=preds), sample_submission_df)

    importances = importances / cv.get_n_splits()
    feature_importance_df = pd.DataFrame({"feature": features_names, "importance": importances}).sort_values(by = "importance", ascending = False).reset_index(drop = True)
    print(feature_importance_df)
    feature_importance_df.to_excel("E:/importances.xlsx")

    

    # Stop the timer and print the exectution time
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")

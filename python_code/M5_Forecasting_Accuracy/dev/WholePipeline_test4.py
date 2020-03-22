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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
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
from m5_forecasting_accuracy.preprocessing.categorical_encoders import CategoricalFeaturesEncoder, OrdinalEncoder, GroupingEncoder, LeaveOneOutEncoder, TargetAvgEncoder

pd.set_option("display.max_columns", 100)

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

def make_submission(test, submission, DAYS_PRED):
    preds = test[["id", "date", "demand"]]
    preds = preds.pivot(index="id", columns="date", values="demand").reset_index()
    preds.columns = ["id"] + ["F" + str(d + 1) for d in range(DAYS_PRED)]

    evals = submission[submission["id"].str.endswith("evaluation")]
    vals = submission[["id"]].merge(preds, how="inner", on="id")
    final = pd.concat([vals, evals])

    assert final.drop("id", axis=1).isnull().sum().sum() == 0
    assert final["id"].equals(submission["id"])

    final.to_csv(PREDICTIONS_DIRECTORY_PATH_str + "submission_kaggle_22032020.csv", index=False)

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
        
    enable_validation = True

    dl = DataLoader()
    training_set_df, target_sr, testing_set_df, sample_submission_df = dl.load_data(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, "2016-04-24", enable_validation = True)

    print("Training set shape:", training_set_df.shape)
    print("Testing set shape:", testing_set_df.shape)

    #categorical_columns_to_be_encoded_lst = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    #categorical_encoders_lst = [OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder()]

    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint1_v3.pkl", "wb") as f:
        pickle.dump((training_set_df, target_sr, testing_set_df, sample_submission_df), f)

    """categorical_columns_to_be_encoded_lst = ["item_id", "dept_id", "dept_id", "store_id", "store_id", 
                                             "cat_id", "cat_id", "state_id", "state_id", "event_name_1", 
                                             "event_type_1", "event_name_2", "event_type_2", "weekday", "weekday"]

    categorical_encoders_lst = [TargetAvgEncoder(), LabelBinarizer(), TargetAvgEncoder(), TargetAvgEncoder(), LabelBinarizer(), 
                                LabelBinarizer(), TargetAvgEncoder(), LabelBinarizer(), TargetAvgEncoder(), LabelBinarizer(), 
                                LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), TargetAvgEncoder()]"""
    
    """cat_enc = CategoricalFeaturesEncoder(categorical_columns_to_be_encoded_lst, categorical_encoders_lst)
    training_set_df = cat_enc.fit_transform(training_set_df, training_set_df["demand"]) # y is not used here; think to generate y_lag using shifts
    testing_set_df = cat_enc.transform(testing_set_df)"""
    
    prp = PreprocessingStep(test_days = 28, dt_col = "date", keep_last_train_days = 366) # 366 = shift + max rolling (365)
    training_set_df = prp.fit_transform(training_set_df, training_set_df["demand"]) # y is not used here; think to generate y_lag using shifts
    testing_set_df = prp.transform(testing_set_df)

    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint2_v3.pkl", "wb") as f:
        pickle.dump((training_set_df, testing_set_df, target_sr), f)

    print("Training set shape after preprocessing:", training_set_df.shape)
    print("Testing set shape after preprocessing:", testing_set_df.shape)

    dt_col = "date"            
    id_date = testing_set_df[["id", "date"]].reset_index(drop = True) # keep these two columns to use later.
    DAYS_PRED = sample_submission_df.shape[1] - 1  # 28

    # Attach "date" to X_train for cross validation.
    useless_features_lst = ["wm_yr_wk", "quarter", "id", "demand", "shifted_demand"]
    y_train = target_sr.reset_index(drop = True)
    training_set_df.drop(useless_features_lst, axis = 1, inplace = True)
    testing_set_df.drop(["date"] + useless_features_lst, axis = 1, inplace = True)
    X_train = training_set_df.reset_index(drop = True)
    X_test = testing_set_df.reset_index(drop = True)

    gc.collect()
    
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

    # Training models
    cv = CustomTimeSeriesSplitter(n_splits = 3, train_days = 365 * 2, test_days = DAYS_PRED, dt_col = "date")
    models = train_lgb(bst_params, fit_params, X_train, y_train, cv, drop_when_train = ["date"])

    del X_train, y_train
    gc.collect()

    # Making predictions
    imp_type = "gain"
    importances = np.zeros(X_test.shape[1])
    preds = np.zeros(X_test.shape[0])

    features_names = []
    for model in models:
        preds += model.predict(X_test)
        importances += model.feature_importance(imp_type)
        features_names = model.feature_name()

    preds = preds / cv.get_n_splits()
    make_submission(id_date.assign(demand=preds), sample_submission_df, DAYS_PRED)

    # Feature importance
    importances = importances / cv.get_n_splits()
    feature_importance_df = pd.DataFrame({"feature": features_names, "importance": importances}).sort_values(by = "importance", ascending = False).reset_index(drop = True)
    print(feature_importance_df)
    feature_importance_df.to_excel("E:/importances.xlsx")

    # Stop the timer and print the exectution time
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")

    # 17/03:
    # [897]   train's rmse: 2.12774   valid's rmse: 2.22693
    # [833]   train's rmse: 2.12977   valid's rmse: 2.1448
    # [1175]  train's rmse: 2.09825   valid's rmse: 2.12569
    # Public LB score: 0.62366 - File: submission_kaggle_17032020_LB_0.62366.csv

    # 18/03:
    # [1481]  train's rmse: 2.07964   valid's rmse: 2.2132
    # [1142]  train's rmse: 2.10411   valid's rmse: 2.14399
    # [965]   train's rmse: 2.11633   valid's rmse: 2.13404
    # Public LB score: 0.62222 - File: submission_kaggle_18032020_LB_0.62222.csv

    # 19/03:
    # [1481]  train's rmse: 2.07964   valid's rmse: 2.2132
    # [1142]  train's rmse: 2.10411   valid's rmse: 2.14399
    # [965]   train's rmse: 2.11633   valid's rmse: 2.13404
    # Public LB score: 0.62369 - File: submission_kaggle_19032020_LB_0.62369.csv

    # 21/03:
    # [939]   train's rmse: 2.1221    valid's rmse: 2.22015
    # [1260]  train's rmse: 2.09468   valid's rmse: 2.14285
    # [970]   train's rmse: 2.11698   valid's rmse: 2.13183
    # Public LB score: 0.62338 - File: submission_kaggle_21032020_LB_0.62338.csv

    # 22/03:
    # [1145]  train's rmse: 2.10359   valid's rmse: 2.22309
    # [1998]  train's rmse: 2.0497    valid's rmse: 2.13933
    # [915]   train's rmse: 2.12153   valid's rmse: 2.13127
    # Public LB score: 0.62285 - File: submission_kaggle_22032020_LB_0.62285.csv

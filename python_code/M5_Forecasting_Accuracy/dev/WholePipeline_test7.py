#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the M5 Forecasting Accuracy competition                  #
#                                                                             #
# This file is the entry point of the solution.                               #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2020-04-20                                                            #
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
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

from dev.files_paths import *
from m5_forecasting_accuracy.data_loading.data_loader import DataLoader
from m5_forecasting_accuracy.models_evaluation.wrmsse_metric_fast import WRMSSEEvaluator
from m5_forecasting_accuracy.models_evaluation.metric_dashboard import WRMSSEDashboard
from m5_forecasting_accuracy.models.lgbm_store_model import LGBMStoreModel

pd.set_option("display.max_columns", 100)

def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

def worker(store_id, store_training_set_df, store_target_df, store_testing_set_df):
    start_time2 = time.time()
    print("Training model for store_id:", store_id)
    print("    Current training set shape:", store_training_set_df.shape)
    gc.collect()

    first_store_model = LGBMStoreModel(train_test_date_split, eval_start_date, store_id)
    first_store_model.fit(store_training_set_df, store_target_df)
    preds = first_store_model.predict(store_testing_set_df)

    print("Model for store_id:", store_id, "done in:", time.time() - start_time2, "seconds")

    return preds

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
        
    enable_validation = False

    if enable_validation:
        date_to_predict = "2016-03-28"
        train_test_date_split = "2016-03-27"
        eval_start_date = "2016-02-28"
    else:
        date_to_predict = "2016-04-25"
        train_test_date_split = "2016-04-24"
        eval_start_date = "2016-03-27"

    dl = DataLoader()
    training_set_df, target_df, testing_set_df, truth_df, orig_target_df, sample_submission_df = dl.load_data_v3(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, 28, train_test_date_split, enable_validation = False, first_day = 1, max_lags = 57, shift_target = False)
    id_date_df = testing_set_df[["id", "date"]].copy()

    print("Statistics for all data:")
    print("    Training set shape:", training_set_df.shape)
    print("    Testing set shape:", testing_set_df.shape)

    all_preds_lst = Parallel(n_jobs = 10, max_nbytes = None)(delayed(worker)(store_id, training_set_df.loc[training_set_df["store_id"] == store_id].copy(), target_df.loc[training_set_df["store_id"] == store_id].copy(), testing_set_df.loc[testing_set_df["store_id"] == store_id].copy()) for store_id in training_set_df["store_id"].unique().tolist())
    preds_df = pd.concat(all_preds_lst, axis = 0)

    print("*** Train + predict: Executed in:", time.time() - start_time, "seconds ***")

    submission = sample_submission_df[["id"]]
    submission = submission.merge(preds_df, on = ["id"], how = "left").fillna(0)
    submission.to_csv(PREDICTIONS_DIRECTORY_PATH_str + "submission_kaggle_01062020.csv", index = False)

    # Feature importance
    feature_importance_df = first_store_model.lgb_model.get_features_importance()
    feature_importance_df.to_excel("E:/draft_lgb_importances.xlsx")

    # Stop the timer and print the exectution time
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")
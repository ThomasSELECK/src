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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import lightgbm as lgb

from dev.files_paths import *
from m5_forecasting_accuracy.data_loading.data_loader import DataLoader
from m5_forecasting_accuracy.preprocessing.PreprocessingStep import PreprocessingStep
from m5_forecasting_accuracy.preprocessing.categorical_encoders import CategoricalFeaturesEncoder, OrdinalEncoder, GroupingEncoder, LeaveOneOutEncoder, TargetAvgEncoder
from m5_forecasting_accuracy.models.lightgbm_wrapper import LGBMRegressor
from m5_forecasting_accuracy.model_utils.efficient_pipeline import EfficientPipeline

pd.set_option("display.max_columns", 100)

def transform(data_df):
    st = time.time()
    print("Transforming data...")
    nan_features = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]

    for feature in nan_features:
        data_df[feature].fillna("unknown", inplace = True)

    cat = ["dept_id", "store_id", "cat_id", "state_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for feature in cat:
        encoder = OneHotEncoder(handle_unknown = "ignore")
        tmp = pd.DataFrame(encoder.fit_transform(data_df[[feature]]).todense(), index = data_df.index, columns = [feature + "_" + str(f).replace(" ", "_") for f in encoder.categories_[0].tolist()])
        #tmp.drop("unknown", axis = 1, inplace = True)
        tmp = tmp.astype(np.int8)

        if feature not in ["item_id", "dept_id", "store_id", "cat_id", "state_id"]:
            data_df.drop(feature, axis = 1, inplace = True)

        data_df = pd.concat([data_df, tmp], axis = 1)

    cat = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]
    for feature in cat:
        encoder = TargetAvgEncoder()
        data_df[feature] = encoder.fit_transform(data_df[feature], data_df["demand"])
    
    print("Transforming data... done in", round(time.time() - st, 3), "secs")

    return data_df

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
        
    enable_validation = True

    # define list of features
    """
    features_lst = ["date", "item_id_OrdinalEncoder", "dept_id_OrdinalEncoder", "cat_id_OrdinalEncoder", "store_id_OrdinalEncoder", "state_id_OrdinalEncoder", 
                "year", "month", "week", "day", "dayofweek", "event_name_1_OrdinalEncoder", "event_type_1_OrdinalEncoder", "event_name_2_OrdinalEncoder", "event_type_2_OrdinalEncoder", 
                "snap_CA", "snap_TX", "snap_WI", "sell_price", "lag_t28", "lag_t29", "lag_t30", "rolling_mean_t7", "rolling_std_t7", "rolling_mean_t30", "rolling_mean_t90", 
                "rolling_mean_t180", "rolling_std_t30", "price_change_t1", "price_change_t365", "rolling_price_std_t7", "rolling_price_std_t30", "rolling_skew_t30", "rolling_kurt_t30"]
    """

    features = ["date", "item_id", "dept_id", "cat_id", "store_id", "state_id", "year", "month", "week", "day", "dayofweek", "event_name_1", "event_type_1", "event_name_2", "event_type_2", 
                "snap_CA", "snap_TX", "snap_WI", "sell_price", "lag_t28", "lag_t29", "lag_t30", "rolling_mean_t7", "rolling_std_t7", "rolling_mean_t30", "rolling_mean_t90", 
                "rolling_mean_t180", "rolling_std_t30", "price_change_t1", "price_change_t365", "rolling_price_std_t7", "rolling_price_std_t30", "rolling_skew_t30", "rolling_kurt_t30"]

    
    dl = DataLoader()
    train_df, test1_df, test2_df, sample_submission_df = dl.load_data(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, enable_validation = True)

    data_df = pd.concat([train_df, test1_df], axis = 0)

    # We have the data to build our first model, let's build a baseline and predict the validation data (in our case is test1)
    data_df = transform(data_df)

    """categorical_columns_to_be_encoded_lst = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    categorical_encoders_lst = [OrdinalEncoder(missing_value_replacement = "unknown"), OrdinalEncoder(missing_value_replacement = "unknown"), OrdinalEncoder(missing_value_replacement = "unknown"),
    OrdinalEncoder(missing_value_replacement = "unknown"), OrdinalEncoder(missing_value_replacement = "unknown"), OrdinalEncoder(missing_value_replacement = "unknown"), 
    OrdinalEncoder(missing_value_replacement = "unknown"), OrdinalEncoder(missing_value_replacement = "unknown"), OrdinalEncoder(missing_value_replacement = "unknown")]
    cat_enc = CategoricalFeaturesEncoder(categorical_columns_to_be_encoded_lst, categorical_encoders_lst)
    data_df = cat_enc.fit_transform(data_df)"""

    prp = PreprocessingStep()
    data_df = prp.transform(data_df)
              
    # reduce memory for new features so we can train
    data_df = dl.reduce_mem_usage(data_df, "data_df")

    train_df = data_df.loc[data_df["date"] <= "2016-04-24"]
    target_sr = train_df["demand"]
    train_df.drop(["demand", "id", "wm_yr_wk"], axis = 1, inplace = True) #train_df[features]
    test1_df = data_df[(data_df["date"] > "2016-04-24")]

    useless_features_lst = ["event_name_1_Eid_al-Fitr", "event_name_1_Halloween", "event_name_1_EidAlAdha", "event_name_1_StPatricksDay", "event_name_1_LentStart", "event_name_1_Chanukah_End",
                            "cat_id_HOBBIES", "dept_id_HOUSEHOLD_2", "dept_id_HOBBIES_2", "event_name_1_Cinco_De_Mayo", "event_name_1_NBAFinalsStart", "event_name_1_LentWeek2",
                            "event_name_1_MartinLutherKingDay", "event_name_1_MemorialDay", "event_name_1_NBAFinalsEnd", "event_type_2_unknown", "event_type_2_Religious", "event_type_2_Cultural",
                            "event_name_2_Father's_day", "event_type_1_Religious", "event_name_1_OrthodoxEaster", "event_name_1_Pesach_End", "event_name_1_Purim_End", "event_name_1_Ramadan_starts",
                            "event_name_1_OrthodoxChristmas"]
    train_df.drop(useless_features_lst, axis = 1, inplace = True)
    test1_df.drop(useless_features_lst, axis = 1, inplace = True)

    del data_df

    gc.collect()

    """
    if not os.path.exists("E:/M5_Forecasting_Accuracy_cache/cached_data.pkl"):
        with open("E:/M5_Forecasting_Accuracy_cache/cached_data.pkl", "wb") as f:
            pickle.dump((train_df, target_sr, test1_df, test2_df, sample_submission_df), f)
    else:
        with open("E:/M5_Forecasting_Accuracy_cache/cached_data.pkl", "rb") as f:
            train_df, target_sr, test1_df, test2_df, sample_submission_df = pickle.load(f)"""

    st2 = time.time()
    print("Training model...") 

    # define random hyperparammeters
    lgb_params = {
        "boosting_type": "gbdt",
        "metric": "rmse",
        "objective": "regression",
        "n_jobs": -1,
        "seed": 236,
        "learning_rate": 0.025,
        "bagging_fraction": 0.85,
        "bagging_freq": 10, 
        "colsample_bytree": 0.75,
        "max_depth": -1,
        "reg_alpha": 0,
        "reg_lambda": 0.5,
        "min_split_gain": 0.001,
        "device": "cpu",
        "verbosity": -1
    }
    
    lgbm = LGBMRegressor(lgb_params, early_stopping_rounds = 50, maximize = False, nrounds = 8000, eval_split_type = "time", eval_start_date = "2016-03-27", eval_date_col = "date", verbose_eval = 100, enable_cv = False)
    lgbm.fit(train_df, target_sr)
        
    print("Training model... done in", round(time.time() - st2, 3), "secs")

    st = time.time()
    print("Generating predictions...")

    #features = [f for f in features if f != "date"]
    test1_df.drop(["date", "id"], axis = 1, inplace = True)
    y_pred = lgbm.predict(test1_df)
    test1_df["demand"] = y_pred
    
    predictions = test1_df[["id", "date", "demand"]]
    predictions = pd.pivot(predictions, index = "id", columns = "date", values = "demand").reset_index()
    predictions.columns = ["id"] + ["F" + str(i + 1) for i in range(28)]

    evaluation_rows = [row for row in sample_submission_df["id"] if "evaluation" in row] 
    evaluation = sample_submission_df[sample_submission_df["id"].isin(evaluation_rows)]
    validation = sample_submission_df[["id"]].merge(predictions, on = "id")
    
    final = pd.concat([validation, evaluation])
    final.to_csv(PREDICTIONS_DIRECTORY_PATH_str + "submission_16032020.csv", index = False)

    print("Generating predictions... done in", round(time.time() - st, 3), "secs")

    feature_importance_df = lgbm.get_features_importance()
    feature_importance_df.to_excel("E:/lgb_feature_importance.xlsx", index = False)

    lgbm.plot_features_importance()
    
    # Stop the timer and print the exectution time
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")

    # 15/03 - [1998]  training's rmse: 2.24776        valid_1's rmse: 2.11289 - Public LB score: 0.64345 - File submission_14032020.csv
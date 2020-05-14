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

import warnings
warnings.filterwarnings("ignore")

from dev.files_paths import *
from m5_forecasting_accuracy.data_loading.data_loader import DataLoader
from m5_forecasting_accuracy.preprocessing.PreprocessingStep3 import PreprocessingStep3
from m5_forecasting_accuracy.models.lightgbm_wrapper import LGBMRegressor
from m5_forecasting_accuracy.model_utils.CustomTimeSeriesSplitter import CustomTimeSeriesSplitter
from m5_forecasting_accuracy.preprocessing.categorical_encoders import CategoricalFeaturesEncoder, OrdinalEncoder, GroupingEncoder, LeaveOneOutEncoder, TargetAvgEncoder
from m5_forecasting_accuracy.models_evaluation.wrmsse_metric_fast import WRMSSEEvaluator
from m5_forecasting_accuracy.models_evaluation.metric_dashboard import WRMSSEDashboard

pd.set_option("display.max_columns", 100)

def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
        
    enable_validation = True
    max_lags = 57
    test_days = 1

    dl = DataLoader()
    training_set_df, target_df, testing_set_df, truth_df, orig_target_df = dl.load_data_v3(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, "2016-03-27", enable_validation = enable_validation, first_day = 350, max_lags = max_lags)

    print("Training set shape:", training_set_df.shape)
    print("Testing set shape:", testing_set_df.shape)

    prp = PreprocessingStep3(dt_col = "date", keep_last_train_days = 366) # 366 = shift + max rolling (365)
    training_set_df = prp.fit_transform(training_set_df, target_df["demand"]) # y is not used here
    testing_set_df = prp.transform(testing_set_df)

    print("Training set shape after preprocessing:", training_set_df.shape)
    print("Testing set shape after preprocessing:", testing_set_df.shape)

    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint2_v6.pkl", "wb") as f:
        pickle.dump((training_set_df, testing_set_df, target_df, truth_df, orig_target_df), f)

    lgb_params = {
        "boosting_type": "gbdt",
        "metric": "custom",
        "objective": "custom",
        "n_jobs": -1,
        "seed": 20,
        "learning_rate": 0.075,
        "subsample": 0.66,
        "bagging_freq": 2,
        "colsample_bytree": 0.77,
        "max_depth": -1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbosity": -1
    }

    # Attach "date" to X_train for cross validation.

    ## Those two lines need to be executed if data is loaded from checkpoint2_v6.pkl
    #training_set_df = training_set_df.sort_values(["id", "date"], ascending = True).reset_index(drop = True)
    #target_df = target_df.sort_values(["id", "date"], ascending = True).reset_index(drop = True)

    evaluator = WRMSSEEvaluator(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, "2016-03-27")
    id_date = testing_set_df[["id", "date"]].reset_index(drop = True)
    training_set_weights_sr = evaluator.generate_dataset_weights(training_set_df)
    eval_start_date = "2016-02-28"
    useless_features_lst = ["wm_yr_wk", "quarter", "id", "shifted_demand", "d"]
    y_train = target_df["demand"].reset_index(drop = True)
    training_set_df.drop(useless_features_lst, axis = 1, inplace = True)
    testing_set_df.drop(["date"] + useless_features_lst, axis = 1, inplace = True)
    X_train = training_set_df.reset_index(drop = True)
    X_test = testing_set_df.reset_index(drop = True)

    gc.collect()

    lgb_model = LGBMRegressor(lgb_params, early_stopping_rounds = 200, custom_eval_function = evaluator.lgb_feval, custom_objective_function = evaluator.lgb_fobj, maximize = False, nrounds = 3000, eval_split_type = "time", eval_start_date = eval_start_date, eval_date_col = "date", verbose_eval = 100, enable_cv = False)
    lgb_model.fit(X_train, y_train, sample_weights = training_set_weights_sr)

    # Feature importance
    feature_importance_df = lgb_model.get_features_importance()
    feature_importance_df.to_excel("E:/draft_lgb_importances.xlsx")

    # Making predictions
    preds = lgb_model.predict(X_test)
    preds = id_date.assign(demand = preds)
    #preds = pd.concat([orig_target_df.loc[orig_target_df["date"] == "2016-03-27"], preds], axis = 0)
    #preds = preds.sort_values(["id", "date"]).reset_index(drop = True)
    #preds["demand"] = preds.groupby("id")["demand"].cumsum()
    preds = preds.loc[preds["date"] == "2016-03-28"]
        
    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint4_v4_best_preds.pkl", "rb") as f:
        best_preds = pickle.load(f)

    best_preds.columns = ["id", "date", "best_preds"]
    preds.columns = ["id", "date", "preds"]
    preds = preds.merge(best_preds, how = "left", on = ["id", "date"])
    preds = preds.merge(orig_target_df, how = "left", on = ["id", "date"])
    preds["preds_diff"] = np.abs(preds["best_preds"] - preds["preds"])
    preds["best_preds_diff_to_tgt"] = np.abs(preds["best_preds"] - preds["demand"])
    preds["preds_diff_to_tgt"] = np.abs(preds["preds"] - preds["demand"])
    preds["best_solution"] = preds[["best_preds_diff_to_tgt", "preds_diff_to_tgt"]].min(axis = 1)
    preds["best_solution_col"] = (preds["best_preds_diff_to_tgt"] == preds["best_solution"]).astype(np.int8).map({1: "best_preds", 0: "preds"})
    preds["new_best_solution"] = preds["best_preds"]
    preds["new_best_solution"].loc[preds["best_solution_col"] == "preds"] = preds["preds"].loc[preds["best_solution_col"] == "preds"]
    best_preds_rmse_by_date_df = preds[["date", "best_preds", "demand"]].groupby("date").apply(lambda x: rmse(x["demand"], x["best_preds"])).reset_index()
    best_preds_rmse_by_date_df.columns = ["date", "best_preds_rmse"]
    preds_rmse_by_date_df = preds[["date", "preds", "demand"]].groupby("date").apply(lambda x: rmse(x["demand"], x["preds"])).reset_index()
    preds_rmse_by_date_df.columns = ["date", "preds_rmse"]
    rmse_by_date_df = best_preds_rmse_by_date_df.merge(preds_rmse_by_date_df, how = "left", on = "date")
    
    print(rmse_by_date_df)

    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint4_v4_best_preds.pkl", "rb") as f:
        best_preds = pickle.load(f)

    preds2 = preds.copy()
    preds2 = preds2[["id", "date", "preds"]]
    preds2.columns = ["id", "date", "demand"]
    best_preds = best_preds.loc[best_preds["date"] > "2016-03-28"]
    best_preds = pd.concat([preds2, best_preds], axis = 0)
    best_preds.sort_values(["date", "id"], inplace = True)
    truth_df.columns = ["id", "date", "truth"]
    best_preds = best_preds.merge(truth_df, how = "left", on = ["id", "date"])
    best_preds_piv = best_preds[["id", "date", "demand"]].pivot(index = "id", columns = "date", values = "demand").reset_index()
    truth_piv = best_preds[["id", "date", "truth"]].pivot(index = "id", columns = "date", values = "truth").reset_index()
    truth_piv.set_index("id", inplace = True)
    best_preds_piv.set_index("id", inplace = True)
    print("Validation WRMSSE:", round(evaluator.wrmsse(best_preds_piv, truth_piv, score_only = True), 6)) 
    # Validation WRMSSE: 0.558016, Validation WRMSSE: 0.560827 with training with WRMSSE metric
    # [2480]	training's WRMSSE: 0.550449	valid_1's WRMSSE: 0.591994
    
    # Validation WRMSSE: 0.560619
    # [1000]	training's WRMSSE: 0.563472	valid_1's WRMSSE: 0.597382 => training with custom loss + custom metric
    #         date  best_preds_rmse  preds_rmse
    # 0 2016-03-28         1.891767    1.809265

    # Validation WRMSSE: 0.559974
    # [1338]	training's WRMSSE: 0.5504	valid_1's WRMSSE: 0.595589 => training with regression loss + custom metric
    #         date  best_preds_rmse  preds_rmse
    # 0 2016-03-28         1.891767    1.811851

    # Validation WRMSSE: 0.559778
    # [611]	training's WRMSSE: 0.4978	valid_1's WRMSSE: 0.592817 => training with tweedie loss + custom metric
    #         date  best_preds_rmse  preds_rmse
    # 0 2016-03-28         1.891767    1.865172

    # Validation WRMSSE: 0.560329
    # [1815]	training's WRMSSE: 0.553814	valid_1's WRMSSE: 0.593808 => training with custom loss + custom metric + sqrt(weights)
    #         date  best_preds_rmse  preds_rmse
    # 0 2016-03-28         1.891767    1.800062

    d = WRMSSEDashboard(PLOTS_DIRECTORY_PATH_str + "dashboard/")
    d.create_dashboard(evaluator, best_preds)

    ####

    imp_type = "gain"
    importances = np.zeros(X_test.shape[1])
    preds = np.zeros(X_test.shape[0])

    features_names = []
    for model in models:
        preds += model.predict(X_test)
        importances += model.feature_importance(imp_type)
        features_names = model.feature_name()

    preds = preds / cv.get_n_splits()
    preds = id_date.assign(demand = preds)

    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint4_v5.pkl", "wb") as f:
        pickle.dump(preds, f)

    if enable_validation:
        with open("E:/M5_Forecasting_Accuracy_cache/checkpoint1_v4.pkl", "rb") as f:
            training_set_df, target_df, testing_set_df, truth_df, sample_submission_df = pickle.load(f)

        with open("E:/M5_Forecasting_Accuracy_cache/checkpoint4_v5.pkl", "rb") as f:
            preds = pickle.load(f)

        training_set_df["demand"] = target_df["demand"]
        testing_set_df["demand"] = truth_df["demand"]
        e = WRMSSEEvaluator(training_set_df, testing_set_df)
        print("Validation WRMSSE:", round(e.score(preds), 6))
    else:
        make_submission(preds, sample_submission_df, DAYS_PRED)

    # Feature importance
    importances = importances / cv.get_n_splits()
    feature_importance_df = pd.DataFrame({"feature": features_names, "importance": importances}).sort_values(by = "importance", ascending = False).reset_index(drop = True)
    print(feature_importance_df)
    feature_importance_df.to_excel("E:/importances.xlsx")

    # Stop the timer and print the exectution time
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")
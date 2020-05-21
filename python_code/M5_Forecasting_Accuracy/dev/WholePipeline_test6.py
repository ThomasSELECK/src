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

import warnings
warnings.filterwarnings("ignore")

from dev.files_paths import *
from m5_forecasting_accuracy.data_loading.data_loader import DataLoader

from m5_forecasting_accuracy.models_evaluation.metric_dashboard import WRMSSEDashboard
from m5_forecasting_accuracy.models.lgbm_day_model import LGBMDayModel

pd.set_option("display.max_columns", 100)

def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
        
    enable_validation = False
    max_lags = 57
    test_days = 1
    day_to_predict = 1 # This means we want to predict t + 1 knowing t

    if enable_validation:
        date_to_predict = "2016-03-28"
        train_test_date_split = "2016-03-27"
        eval_start_date = "2016-02-28"
    else:
        date_to_predict = "2016-04-25"
        train_test_date_split = "2016-04-24"
        eval_start_date = "2016-03-27"

    dl = DataLoader()
    training_set_df, target_df, testing_set_df, truth_df, orig_target_df, sample_submission_df = dl.load_data_v3(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, day_to_predict, train_test_date_split, enable_validation = enable_validation, first_day = 350, max_lags = max_lags)

    print("Training set shape:", training_set_df.shape)
    print("Testing set shape:", testing_set_df.shape)

    with open("E:/M5_Forecasting_Accuracy_cache/loaded_data_16052020.pkl", "wb") as f:
        pickle.dump((training_set_df, target_df, testing_set_df, truth_df, orig_target_df, sample_submission_df), f)

    y_train = target_df["demand"].reset_index(drop = True)

    day_model = LGBMDayModel(train_test_date_split, eval_start_date)
    day_model.fit(training_set_df, y_train)
    preds = day_model.predict(testing_set_df, date_to_predict)
    #preds = preds.merge(orig_target_df[["id", "date", "shifted_demand"]], how = "left", on = ["id", "date"])
    #preds["demand"] = preds["shifted_demand"] + preds["demand"]
    #preds.drop("shifted_demand", axis = 1, inplace = True)
    print("*** Train + predict: Executed in:", time.time() - start_time, "seconds ***")

    # Attach "date" to X_train for cross validation.

    ## Those two lines need to be executed if data is loaded from checkpoint2_v6.pkl
    #training_set_df = training_set_df.sort_values(["id", "date"], ascending = True).reset_index(drop = True)
    #target_df = target_df.sort_values(["id", "date"], ascending = True).reset_index(drop = True)

    # Feature importance
    feature_importance_df = day_model.lgb_model.get_features_importance()
    feature_importance_df.to_excel("E:/draft_lgb_importances.xlsx")

    if enable_validation:
        with open("E:/preds_test_18052020.pkl", "wb") as f:
            pickle.dump((preds, truth_df, orig_target_df), f)

        """
        with open("E:/preds_test5.pkl", "rb") as f:
            preds, truth_df = pickle.load(f)

        truth_df.columns = ["id", "date", "demand"]
        orig_target_df = truth_df.copy()
        preds = preds.loc[preds["date"] == "2016-03-28"]
        """

        # Score the predictions
        preds2 = preds.copy()
        preds2.columns = ["id", "date", "preds"]
        preds_rmse_by_date_df = preds2.merge(truth_df, how = "left", on = ["id", "date"])
        preds_rmse_by_date_df = preds_rmse_by_date_df[["date", "preds", "demand"]].groupby("date").apply(lambda x: rmse(x["demand"], x["preds"])).reset_index()
        preds_rmse_by_date_df.columns = ["date", "preds_rmse"]
        print(preds_rmse_by_date_df)

        best_preds_piv = preds[["id", "date", "demand"]].pivot(index = "id", columns = "date", values = "demand").reset_index()
        truth_piv = truth_df[["id", "date", "demand"]].pivot(index = "id", columns = "date", values = "demand").reset_index()
        truth_piv.set_index("id", inplace = True)
        best_preds_piv.set_index("id", inplace = True)
        best_preds_piv.columns = ["F" + str(i) for i in range(1, 29)]
        truth_piv.columns = ["F" + str(i) for i in range(1, 29)]
        print("Validation WRMSSE:", round(day_model.evaluator.wrmsse(best_preds_piv, truth_piv, score_only = True), 6))

        """
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
        best_preds = best_preds.loc[best_preds["date"] > date_to_predict]
        best_preds = pd.concat([preds2, best_preds], axis = 0)
        best_preds.sort_values(["date", "id"], inplace = True)
        truth_df.columns = ["id", "date", "truth"]
        best_preds = best_preds.merge(truth_df, how = "left", on = ["id", "date"])
        best_preds_piv = best_preds[["id", "date", "demand"]].pivot(index = "id", columns = "date", values = "demand").reset_index()
        truth_piv = best_preds[["id", "date", "truth"]].pivot(index = "id", columns = "date", values = "truth").reset_index()
        truth_piv.set_index("id", inplace = True)
        best_preds_piv.set_index("id", inplace = True)
        best_preds_piv.columns = ["F" + str(i) for i in range(1, 29)]
        truth_piv.columns = ["F" + str(i) for i in range(1, 29)]
        print("Validation WRMSSE:", round(day_model.evaluator.wrmsse(best_preds_piv, truth_piv, score_only = True), 6)) 
        """

        # Validation WRMSSE: 0.557925
        # [703]	training's WRMSSE: 0.536104	valid_1's WRMSSE: 0.552914 => training with custom loss + custom metric + sqrt(weights) + outliers removal
        #         date  best_preds_rmse  preds_rmse
        # 0 2016-03-28         1.891767    1.815375

        # Validation WRMSSE: 0.558455
        # [1947]  training's WRMSSE: 0.50431      valid_1's WRMSSE: 0.556969 => training with custom loss + custom metric + sqrt(weights) + outliers removal + categorical features
        #         date  best_preds_rmse  preds_rmse
        # 0 2016-03-28         1.891767    1.776379

        # Validation WRMSSE: 0.558404
        # [2144]  training's WRMSSE: 0.501793     valid_1's WRMSSE: 0.557556 => training with custom loss + custom metric + sqrt(weights) + outliers removal + categorical features
        #         date  best_preds_rmse  preds_rmse
        # 0 2016-03-28         1.891767    1.772304

        d = WRMSSEDashboard(PLOTS_DIRECTORY_PATH_str + "dashboard/")
        d.create_dashboard(day_model.evaluator, best_preds_piv.reset_index())

        # TODO: Look to FOODS_3 items as they seems to have the largest error on Kaggle LB
    else:
        with open("E:/M5_Forecasting_Accuracy_cache/kaggle_preds.pkl", "wb") as f:
            pickle.dump(preds, f)
        best_submission_df = pd.read_csv("D:/Projets_Data_Science/Competitions/Kaggle/M5_Forecasting_Accuracy/predictions/submission_kaggle_16052020_LB_0.47397.csv")
        preds.drop("date", axis = 1, inplace = True)
        preds.columns = ["id", "F1"]
        best_submission_df.drop("F1", axis = 1, inplace = True)
        best_submission_df = best_submission_df.merge(preds, how = "left", on = "id").fillna(0.0)
        best_submission_df = best_submission_df[["id"] + ["F" + str(i) for i in range(1, 29)]]
        best_submission_df.to_csv(PREDICTIONS_DIRECTORY_PATH_str + "submission_kaggle_21052020.csv", index = False)
        
        """preds = preds.pivot(index = "id", columns = "date", values = "demand").reset_index()
        preds.columns = ["id"] + ["F" + str(d + 1) for d in range(28)]

        evals = sample_submission_df[sample_submission_df["id"].str.endswith("evaluation")]
        vals = sample_submission_df[["id"]].merge(preds, how="inner", on="id")
        final = pd.concat([vals, evals])

        final.to_csv(PREDICTIONS_DIRECTORY_PATH_str + "submission_kaggle_15052020.csv", index = False)"""

    # Stop the timer and print the exectution time
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")

"""
def get_max_cross_corr(datax, datay, range_min = -90, range_max = 91):
    rs = pd.Series([datax.corr(datay.shift(lag)) for lag in range(-90, 91)], index = list(range(-90, 91)))
    max_value = rs.max()
    min_value = rs.min()
    offset_max = rs.index[np.argmax(rs)]
    offset_min = rs.index[np.argmin(rs)]

    return (max_value, min_value, offset_max, offset_min)

from joblib import Parallel, delayed

target_piv_df = target_df.pivot(index = "id", columns = "date", values = "demand")
target_piv_df = target_piv_df.transpose()
all_time_series_lst = target_piv_df.columns.tolist()

result = Parallel(n_jobs = 16, verbose = 1)(delayed(get_max_cross_corr)(target_piv_df[all_time_series_lst[0]], target_piv_df[all_time_series_lst[i]]) for i in range(1, 30490))

result_df = pd.DataFrame({"feature1 - feature2": [all_time_series_lst[0] + " - " + all_time_series_lst[i] for i in range(1, 30490)], 
                          "max_cross_corr": [r[0] for r in result], 
                          "min_cross_corr": [r[1] for r in result], 
                          "offset_max": [r[2] for r in result], 
                          "offset_min": [r[3] for r in result]})

st = time.time()
rs = pd.Series([crosscorr(target_piv_df[all_time_series_lst[0]], target_piv_df[all_time_series_lst[1]], lag) for lag in range(-90, 91)])
print("Exec:", time.time() - st, "secs")
rs.index = list(range(-90, 91))

offset = rs.index[np.argmax(rs)]
f, ax = plt.subplots(figsize = (14, 3))
rs.plot.line(ax = ax)
ax.axvline(0, color = "k", linestyle = "--", label = "Center")
ax.axvline(rs.index[np.argmax(rs)], color = "r", linestyle = "--", label = "Peak synchrony")
ax.set(title = f"Offset = {offset} frames\nS1 leads <> S2 leads", xlabel = "Offset", ylabel = "Pearson r")
plt.legend()
plt.show()

d1 = df['S1_Joy']
d2 = df['S2_Joy']
seconds = 5
fps = 30
rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]

offset = np.ceil(len(rs)/2)-np.argmax(rs)
f,ax=plt.subplots(figsize=(14,3))
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
plt.legend()
"""
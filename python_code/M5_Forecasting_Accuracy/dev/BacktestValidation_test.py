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
from m5_forecasting_accuracy.models_evaluation.wrmsse_metric import WRMSSEEvaluator
from m5_forecasting_accuracy.models_evaluation.backtest import Backtest

pd.set_option("display.max_columns", 100)

def train_lgb(bst_params, fit_params, X, y, cv, drop_when_train=None):
    models = []

    if drop_when_train is None:
        drop_when_train = []

    for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y)):
        print(f"\n---------- Fold: ({idx_fold + 1} / {cv.get_n_splits()}) ----------\n")

        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]
        train_set = lgb.Dataset(X_trn.drop(drop_when_train, axis = 1), label = y_trn)
        val_set = lgb.Dataset(X_val.drop(drop_when_train, axis = 1), label = y_val)

        """
        X_trn["d"] = (pd.to_datetime(X_trn["date"]) - pd.to_datetime("2011-01-29")).dt.days + 1
        X_val["d"] = (pd.to_datetime(X_val["date"]) - pd.to_datetime("2011-01-29")).dt.days + 1
        tmp = X_trn["id"].str.split("_", expand = True)
        X_trn["store_id"] = tmp[3].astype(str) + "_" + tmp[4]
        X_trn["dept_id"] = tmp[0].astype(str) + "_" + tmp[1]
        X_trn["state_id"] = tmp[3]
        X_trn["cat_id"] = tmp[0]
        tmp = X_val["id"].str.split("_", expand = True)
        X_val["store_id"] = tmp[3].astype(str) + "_" + tmp[4]
        X_val["dept_id"] = tmp[0].astype(str) + "_" + tmp[1]
        X_val["state_id"] = tmp[3]
        X_val["cat_id"] = tmp[0]
        evaluator = WRMSSEEvaluator(pd.concat([X_trn, y_trn], axis = 1), pd.concat([X_val, y_val], axis = 1))

        gc.collect()

        print("Training...")
        """

        model = lgb.train(
            bst_params,
            train_set,
            valid_sets = [train_set, val_set],
            valid_names = ["train", "valid"],
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

    final.to_csv(PREDICTIONS_DIRECTORY_PATH_str + "submission_kaggle_30032020.csv", index=False)

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

    categorical_columns_to_be_encoded_lst = ["dept_id", "cat_id", "store_id", "state_id"] #["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    categorical_encoders_lst = [OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder()] #, OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder(), OrdinalEncoder()]

    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint1_v4.pkl", "wb") as f:
        pickle.dump((training_set_df, target_sr, testing_set_df, sample_submission_df), f)

    """categorical_columns_to_be_encoded_lst = ["item_id", "dept_id", "dept_id", "store_id", "store_id", 
                                             "cat_id", "cat_id", "state_id", "state_id", "event_name_1", 
                                             "event_type_1", "event_name_2", "event_type_2", "weekday", "weekday"]

    categorical_encoders_lst = [TargetAvgEncoder(), LabelBinarizer(), TargetAvgEncoder(), TargetAvgEncoder(), LabelBinarizer(), 
                                LabelBinarizer(), TargetAvgEncoder(), LabelBinarizer(), TargetAvgEncoder(), LabelBinarizer(), 
                                LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), LabelBinarizer(), TargetAvgEncoder()]"""
    
    cat_enc = CategoricalFeaturesEncoder(categorical_columns_to_be_encoded_lst, categorical_encoders_lst)
    training_set_df = cat_enc.fit_transform(training_set_df, target_sr) # y is not used here; think to generate y_lag using shifts
    testing_set_df = cat_enc.transform(testing_set_df)
    
    prp = PreprocessingStep(test_days = 28, dt_col = "date", keep_last_train_days = 366) # 366 = shift + max rolling (365)
    training_set_df = prp.fit_transform(training_set_df, target_sr) # y is not used here; think to generate y_lag using shifts
    testing_set_df = prp.transform(testing_set_df)

    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint2_v4.pkl", "wb") as f:
        pickle.dump((cat_enc, prp, training_set_df, testing_set_df, target_sr), f)

    print("Training set shape after preprocessing:", training_set_df.shape)
    print("Testing set shape after preprocessing:", testing_set_df.shape)

    dt_col = "date"            
    id_date = testing_set_df[["id", "date"]].reset_index(drop = True) # keep these two columns to use later.
    DAYS_PRED = sample_submission_df.shape[1] - 1  # 28

    # Attach "date" to X_train for cross validation.
    useless_features_lst = ["wm_yr_wk", "quarter", "id", "demand", "shifted_demand"]
    y_train = target_sr.reset_index(drop = True)
    #training_set_df.drop(["wm_yr_wk", "quarter", "demand", "shifted_demand"], axis = 1, inplace = True)
    training_set_df.drop(useless_features_lst, axis = 1, inplace = True)
    testing_set_df.drop(["date"] + useless_features_lst, axis = 1, inplace = True)
    X_train = training_set_df.reset_index(drop = True)
    X_test = testing_set_df.reset_index(drop = True)

    gc.collect()
    
    bst_params = {
        "boosting_type": "gbdt",
        "metric": "rmse",
        "objective": "poisson",
        "n_jobs": -1,
        "seed": 20,
        "learning_rate": 0.025,
        "bagging_fraction": 0.66,
        "bagging_freq": 2,
        "colsample_bytree": 0.77,
        "max_depth": -1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbosity": -1
    }

    fit_params = {
        "num_boost_round": 10000,
        "early_stopping_rounds": 200,
        "verbose_eval": 100,
    }

    # Training models
    cv = CustomTimeSeriesSplitter(n_splits = 3, train_days = 365 * 2, test_days = DAYS_PRED, dt_col = "date")
    #models = train_lgb(bst_params, fit_params, X_train, y_train, cv, drop_when_train = ["date", "id"])
    models = train_lgb(bst_params, fit_params, X_train, y_train, cv, drop_when_train = ["date"])

    del X_train, y_train
    gc.collect()

    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint3_v4.pkl", "wb") as f:
        pickle.dump((models, cv), f)

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

    # 24/03:
    # [1214]  train's rmse: 2.16541   valid's rmse: 2.25236
    # [859]   train's rmse: 2.19513   valid's rmse: 2.16627
    # [1726]  train's rmse: 2.12953   valid's rmse: 2.14768
    # Public LB score: 0.59706 - File: submission_kaggle_24032020_LB_0.59706.csv

    # 25/03:
    # [1179]  train's rmse: 2.22252   valid's rmse: 2.22984
    # [2674]  train's rmse: 2.14329   valid's rmse: 2.16277
    # [1554]  train's rmse: 2.19111   valid's rmse: 2.14433
    # Public LB score: 0.57104 - File: submission_kaggle_25032020_LB_0.57104.csv

    # 25/03:
    # [1443]  train's rmse: 2.2042    valid's rmse: 2.22475
    # [4187]  train's rmse: 2.0997    valid's rmse: 2.15633
    # [4672]  train's rmse: 2.08887   valid's rmse: 2.13024
    # Public LB score: 0.55862 - File: submission_kaggle_25032020_LB_0.55862.csv


    """
    def max_consecutive_ones(a):
        a_ext = np.concatenate(( [0], a, [0] ))
        idx = np.flatnonzero(a_ext[1:] != a_ext[:-1])
        a_ext[1:][idx[1::2]] = idx[::2] - idx[1::2]
        return a_ext.cumsum()[1:-1].max()

    def max_consecutive_ones_at_end(a):
        a_ext = np.concatenate(( [0], a, [0] ))
        idx = np.flatnonzero(a_ext[1:] != a_ext[:-1])
        a_ext[1:][idx[1::2]] = idx[::2] - idx[1::2]
        a_cum = a_ext.cumsum()
        return int(a_cum[1:-1].max() == a_cum[-2])

    tmp = X[["id", "demand"]]
    tmp["demand"] = tmp["demand"].apply(lambda x: int(x == 0))
    tmp2 = tmp.groupby(["id"])["demand"].apply(max_consecutive_ones).reset_index()
    tmp3 = tmp.groupby(["id"])["demand"].apply(max_consecutive_ones_at_end).reset_index()
    tmp2 = tmp2.merge(tmp3, how = "left", on = "id")
    tmp2.columns = ["id", "max_consecutive_zeros", "max_consecutive_zeros_at_end"]
    tmp2.sort_values(["max_consecutive_zeros_at_end", "max_consecutive_zeros"], ascending = False).to_excel("E:/contiguous_zeros.xlsx", index = False)
    """

# Load data
cal = pd.read_csv(CALENDAR_PATH_str)
sell_prices = pd.read_csv(SELL_PRICES_PATH_str)
stv = pd.read_csv(SALES_TRAIN_PATH_str)

def lgbm_process(train_df, valid_df, models):
    valid = valid_df.copy()
    #train["d"] = (pd.to_datetime(train["date"]) - pd.to_datetime("2011-01-29")).dt.days + 1
    #valid["d"] = (pd.to_datetime(valid["date"]) - pd.to_datetime("2011-01-29")).dt.days + 1
          
    id_date = valid[["id", "date"]].reset_index(drop = True) # keep these two columns to use later.
    date_lst = valid["d"].unique()
    date_lst.sort()
    date_lst = ["d_" + str(d) for d in date_lst]

    # Attach "date" to X_train for cross validation.
    useless_features_lst = ["wm_yr_wk", "quarter", "id", "shifted_demand"] + ["all_id", "d", "store_id", "dept_id", "state_id", "cat_id"]
    #train.drop(["demand"] + useless_features_lst, axis = 1, inplace = True)
    valid.drop(["date"] + useless_features_lst, axis = 1, inplace = True) # "demand" is already removed by backtest
    #X_train = train.reset_index(drop = True)
    X_test = valid.reset_index(drop = True)

    # Make predictions
    preds = np.zeros(valid.shape[0])

    features_names = []
    for model in models:
        preds += model.predict(X_test)

    preds = preds / 3 # 3 = cv.get_n_splits()

    id_date = id_date.assign(demand = preds)
    
    return id_date

"""train = stv.melt(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name='d', value_name='demand')
train = train.merge(cal)
train = train.merge(sell_prices, how = "left", on = ["item_id", "store_id", "wm_yr_wk"])
train = add_snap_col(train)
train["demand"] = train["demand"].apply(lambda x: int(x))
train["d"] = train["d"].str.replace("d_", "").apply(lambda x: int(x))"""

# For LGBM only
with open("E:/M5_Forecasting_Accuracy_cache/checkpoint1_v4.pkl", "rb") as f:
    training_set_df, target_df, testing_set_df, truth_df, sample_submission_df = pickle.load(f)

with open("E:/M5_Forecasting_Accuracy_cache/checkpoint2_v4.pkl", "rb") as f:
    cat_enc, prp, training_set_df, testing_set_df, target_sr = pickle.load(f)

with open("E:/M5_Forecasting_Accuracy_cache/checkpoint3_v4.pkl", "rb") as f:
    models, cv = pickle.load(f)

gc.collect()

training_set_df["d"] = (pd.to_datetime(training_set_df["date"]) - pd.to_datetime("2011-01-29")).dt.days + 1
tmp = training_set_df["id"].str.split("_", expand = True)
training_set_df["store_id"] = tmp[3].astype(str) + "_" + tmp[4]
training_set_df["dept_id"] = tmp[0].astype(str) + "_" + tmp[1]
training_set_df["state_id"] = tmp[3]
training_set_df["cat_id"] = tmp[0]
training_set_df["demand"] = target_df["demand"]

del testing_set_df, truth_df, sample_submission_df, cat_enc, prp, target_sr, cv

gc.collect()

backtest = Backtest(training_set_df, process_lst = [lgbm_process], process_names_lst = ["lgbm_3_avg"])

# Instantiate a Backtest object with the differenct processes youd like to compare
#backtest = Backtest(train, process_lst=[process_func, process_01, process_02, process_03, process_04], process_names_lst=['agg_28', 'agg_60', 'agg_90', 'agg_120', 'agg_150'])

# Backtest all processess and time frames at once using .score_all()
## In this case will use every 28 day period going backward 2 years
scores_df = backtest.score_all(models, [0, 28, 56, 84, 112, 140, 168, 196, 224])#, 252, 280, 308, 336, 364, 392, 420, 448, 476, 504, 532, 560, 588, 616, 644, 672, 700, 728])

# Plot the performance accross time
scores_df.plot(figsize = (20,7))
plt.xlabel('Validation end day', fontsize=20)
plt.ylabel('WRMSSEE score', fontsize=20)
plt.title('Process performance over time', fontsize=26)
plt.show()

# Read scores_df to a csv file to save your backtest information easily
scores_df.to_csv('backtest_results_' + '_'.join(backtest.process_names) + '.csv')


# Load data
cal = pd.read_csv(CALENDAR_PATH_str)
sell_prices = pd.read_csv(SELL_PRICES_PATH_str)
ss = pd.read_csv(SAMPLE_SUBMISSION_PATH_str)
stv = pd.read_csv(SALES_TRAIN_PATH_str)

train_df = stv.melt(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name='d', value_name='demand')
train_df = train_df.merge(cal)
train_df = train_df.merge(sell_prices, how = "left", on = ["item_id", "store_id", "wm_yr_wk"])
train_df["demand"] = train_df["demand"].apply(lambda x: int(x))
train_df["d"] = train_df["d"].str.replace("d_", "").apply(lambda x: int(x))
train_df["all_id"] = "all"

train_fold_df = train_df.loc[train_df["d"] <= train_df["d"].max() - 28].reset_index(drop = True)
valid_fold_df = train_df.loc[train_df["d"] > train_df["d"].max() - 28].reset_index(drop = True).copy()

train_target_columns = ["d_" + str(c) for c in train_fold_df["d"].unique().tolist()]
weight_columns = train_target_columns[-28:]

group_ids = ("all_id", "state_id", "store_id", "cat_id", "dept_id", "item_id", 
                 ["state_id", "cat_id"],  ["state_id", "dept_id"], ["store_id", "cat_id"], 
                 ["store_id", "dept_id"], ["item_id", "state_id"], ["item_id", "store_id"])

id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "all_id"]
weight_df = train_fold_df[["item_id", "store_id", "d", "demand", "sell_price"]].loc[train_fold_df["d"] > train_fold_df["d"].max() - 28]
weight_df["value"] = weight_df["demand"] * weight_df["sell_price"]
weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level = 2)["value"]
tmp = train_fold_df[["item_id", "store_id"]].drop_duplicates()
weight_df = weight_df.loc[zip(tmp["item_id"], tmp["store_id"]), :].reset_index(drop = True)
weight_df.columns = ["d_" + str(c) for c in weight_df.columns.tolist()]
weight_df = pd.concat([train_fold_df[id_columns].drop_duplicates(), weight_df], axis = 1, sort = False)
weight_df["all_id"] = "all"

weights_map = {}
for i, group_id in enumerate(tqdm(group_ids, leave = False)):
    lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis = 1)
    lv_weight = lv_weight / lv_weight.sum()

    for i in range(len(lv_weight)):
        weights_map[get_name(lv_weight.index[i])] = np.array([lv_weight.iloc[i]])

weights = pd.DataFrame(weights_map).T / len(group_ids)
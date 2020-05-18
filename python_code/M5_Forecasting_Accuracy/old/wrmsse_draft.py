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
from scipy.sparse import csr_matrix

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
    training_set_df, target_df, testing_set_df, truth_df, orig_target_df = dl.load_data_v3(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, "2016-03-27", enable_validation = enable_validation, first_day = 1, max_lags = max_lags, shift_target = False)

    print("Training set shape:", training_set_df.shape)
    print("Testing set shape:", testing_set_df.shape)

    training_set_df["demand"] = target_df["demand"]
    testing_set_df["demand"] = truth_df["demand"]

    eval_start_date = "2016-02-28"
    lgbm_train_id_date = training_set_df[["id", "date"]].loc[training_set_df["date"] <= eval_start_date].reset_index(drop = True)
    lgbm_valid_id_date = training_set_df[["id", "date"]].loc[training_set_df["date"] > eval_start_date].reset_index(drop = True)

    evaluator = WRMSSEEvaluator(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, "2016-03-27")

    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")

    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint4_v4_best_preds.pkl", "rb") as f:
        best_preds = pickle.load(f)

    truth_df["date"] = pd.to_datetime(truth_df["date"])
    best_preds = best_preds.merge(truth_df, how = "left", on = ["id", "date"])
    best_preds.columns = ["id", "date", "preds", "demand"]
    preds = best_preds[["id", "date", "preds"]].pivot(index = "id", columns = "date", values = "preds").reset_index()
    y_true = best_preds[["id", "date", "demand"]].pivot(index = "id", columns = "date", values = "demand").reset_index()
    preds.columns = ["id"] + ["F" + str(d + 1) for d in range(28)]
    y_true.columns = ["id"] + ["F" + str(d + 1) for d in range(28)]
    
    start_time = time.time()
    print("Score:", round(evaluator.score(preds), 6))
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")

    preds.set_index("id", inplace = True)
    y_true.set_index("id", inplace = True)

    start_time = time.time()
    print("Score2:", round(evaluator.wrmsse(y_true, preds, score_only = True), 6))
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")

    max_day = pd.Series(training_set_df["d"].unique()).str.replace("d_", "").astype(np.int32).max()
    days_lst = ["d_" + str(d) for d in range(max_day - 28 + 1, max_day + 1)]
    data_df = training_set_df[["id", "state_id", "store_id", "cat_id", "dept_id", "item_id", "demand", "d", "sell_price"]].loc[training_set_df["d"].isin(days_lst)]
    data_df["sale_usd"] = data_df["demand"] * data_df["sell_price"]
    data_df["all_id"] = "all"

    training_set_df.drop("d", axis = 1, inplace = True)
    testing_set_df.drop("d", axis = 1, inplace = True)
    evaluator = WRMSSEEvaluator(training_set_df, testing_set_df)

    ### Build roll up matrix to easily compute aggregations. And build an index, so we always know whats where. ###
    groups_ids = [["state_id"], ["store_id"], ["cat_id"], ["dept_id"], ["state_id", "cat_id"], ["state_id", "dept_id"], ["store_id", "cat_id"], ["store_id", "dept_id"], ["item_id"], ["state_id", "item_id"], ["id"]]
    ids_data_df = data_df[["id", "state_id", "store_id", "cat_id", "dept_id", "item_id"]].drop_duplicates().reset_index(drop = True)
    
    # List of categories combinations for aggregations as defined in docs:
    dummies_list = [ids_data_df[group_id[0]] if len(group_id) == 1 else ids_data_df[group_id[0]].astype(str) + "--" + ids_data_df[group_id[1]].astype(str) for group_id in groups_ids]

    ## First element Level_0 aggregation 'all_sales':
    dummies_df_list = [pd.DataFrame(np.ones(ids_data_df.shape[0]).astype(np.int8), index = ids_data_df.index, columns = ["all"]).T]

    # List of dummy dataframes:
    for i, cats in enumerate(dummies_list):
        dummies_df_list += [pd.get_dummies(cats, drop_first = False, dtype = np.int8).T]
    
    # Concat dummy dataframes in one go: Level is constructed for free.
    roll_mat_df = pd.concat(dummies_df_list, keys = list(range(12)), names = ["level", "id"])#.astype(np.int8, copy=False)

    roll_index = roll_mat_df.index
    roll_mat_csr = csr_matrix(roll_mat_df.values)

    ### S - sequence length weights ###
    # Rollup sales:
    sales_train_val = evaluator.train_series

    # Find sales start index:
    start_no = np.argmax(sales_train_val.values > 0, axis = 1)
    
    # Replace days less than min day number with np.nan: Next code line is super slow:
    tmp1 = np.diag(1 / (start_no + 1))
    tmp2 = np.tile(np.arange(1, max_day + 1), (start_no.shape[0], 1))
    flag = np.dot(tmp1, tmp2) < 1
    sales_train_val2 = np.where(flag, np.nan, sales_train_val)

    # Denominator of RMSSE / RMSSE
    S = np.nansum(np.diff(sales_train_val2, axis = 1) ** 2, axis = 1) / (max_day - start_no - 1)

    ### W - USD sales weights ###
    W = 12 * evaluator.weights.values.flatten()
    SW = W / np.sqrt(S)

    with open("E:/M5_Forecasting_Accuracy_cache/checkpoint4_v4_best_preds.pkl", "rb") as f:
        best_preds = pickle.load(f)

    best_preds = best_preds.pivot(index = "id", columns = "date", values="demand").reset_index()
    best_preds.columns = ["id"] + ["F" + str(d + 1) for d in range(28)]
    best_preds.drop("id", axis = 1, inplace = True)
    truth_df = truth_df.pivot(index = "id", columns = "date", values="demand").reset_index()
    truth_df.columns = ["id"] + ["F" + str(d + 1) for d in range(28)]
    truth_df.drop("id", axis = 1, inplace = True)
    
    wrmsse(best_preds, truth_df, score_only = True)

    #print("Validation WRMSSE:", round(evaluator.score(best_preds), 6))
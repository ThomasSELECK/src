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
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from joblib import Parallel, delayed
from functools import reduce

import warnings
warnings.filterwarnings("ignore")

from dev.files_paths import *
from m5_forecasting_accuracy.data_loading.data_loader import DataLoader
from m5_forecasting_accuracy.preprocessing.PreprocessingStep2 import PreprocessingStep2
from m5_forecasting_accuracy.models.lightgbm_wrapper import LGBMRegressor
from m5_forecasting_accuracy.model_utils.CustomTimeSeriesSplitter import CustomTimeSeriesSplitter
from m5_forecasting_accuracy.preprocessing.categorical_encoders import CategoricalFeaturesEncoder, OrdinalEncoder, GroupingEncoder, LeaveOneOutEncoder, TargetAvgEncoder
from m5_forecasting_accuracy.models_evaluation.wrmsse_metric_fast import WRMSSEEvaluator

pd.set_option("display.max_columns", 100)

def make_autoregressive_predictions(testing_set_df, max_lags, train_cols, alpha, weight, m_lgb):
    local_testing_set_df = testing_set_df.copy()
    cols = [f"F{i}" for i in range(1, 29)]
    
    # Make predictions
    for tdelta in range(0, 28):
        day = fday + timedelta(days = tdelta)
        tst = prp.transform(local_testing_set_df.copy(), start_date = day - timedelta(days = max_lags), end_date = day)
        tst = tst.loc[tst.date == day , train_cols]
        local_testing_set_df.loc[local_testing_set_df["date"] == day, "sales"] = alpha * m_lgb.predict(tst) # magic multiplier by kyakovlev
    
    te_sub = local_testing_set_df.loc[local_testing_set_df["date"] >= fday, ["id", "sales"]].copy()
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount() + 1]
    te_sub = te_sub.set_index(["id", "F"]).unstack()["sales"][cols].reset_index()
    te_sub.fillna(0.0, inplace = True)
    te_sub.sort_values("id", inplace = True)
    te_sub.set_index("id", inplace = True)
    te_sub = weight * te_sub
    
    gc.collect()

    return te_sub

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(777)
        
    enable_validation = True
    max_lags = 57

    dl = DataLoader()
    training_set_df, testing_set_df, truth_df = dl.load_data_v2(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, "2016-03-27", enable_validation = enable_validation, first_day = 350, max_lags = max_lags)

    print("Training set shape:", training_set_df.shape)
    print("Testing set shape:", testing_set_df.shape)

    prp = PreprocessingStep2(test_days = 28, dt_col = "date")
    training_set_df = prp.fit_transform(training_set_df, training_set_df["sales"]) # y is not used here
    
    # Need to drop first rows (where sales is null)
    training_set_df = training_set_df.loc[~training_set_df["sales"].isnull()]

    cat_feats = ["item_id", "dept_id","store_id", "cat_id", "state_id", "event_name_1", "event_name_2", "event_type_1", "event_type_2"]
    useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
    train_cols = training_set_df.columns[~training_set_df.columns.isin(useless_cols)]

    # Generate train and valid datasets for LightGBM
    X_train = training_set_df[train_cols]
    y_train = training_set_df["sales"]

    ## TODO: do split baseed on date. Put last 28 days into the validation dataset.
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 2000000, random_state = 42)
    train_set = lgb.Dataset(X_train, label = y_train, categorical_feature = cat_feats, free_raw_data = False)
    val_set = lgb.Dataset(X_valid, label = y_valid, categorical_feature = cat_feats, free_raw_data = False)

    #del training_set_df, X_train, y_train, X_valid, y_valid
    gc.collect()

    params = {
        "boosting_type": "gbdt",
        "metric": "rmse",
        "objective": "poisson",
        "n_jobs": -1,
        "seed": 20,        
        "learning_rate": 0.075,
        "sub_row": 0.75,
        "bagging_freq": 1,
        "lambda_l2" : 0.1,
        "verbosity": 1,
        "num_iterations": 2300,
        "num_leaves": 128,
        "min_data_in_leaf": 60,
        "max_depth": -1
    }
    m_lgb = lgb.train(params, train_set, valid_sets = [train_set, val_set], valid_names = ["train", "valid"], verbose_eval = 100, early_stopping_rounds = 200)
    # m_lgb.save_model("model.lgb") # m_lgb = lgb.Booster(model_file="model.lgb")

    # Feature importance
    importances = m_lgb.feature_importance("gain")
    features_names = m_lgb.feature_name()
    feature_importance_df = pd.DataFrame({"feature": features_names, "importance": importances}).sort_values(by = "importance", ascending = False).reset_index(drop = True)
    feature_importance_df.to_excel("E:/draft_lgb_importances.xlsx")

    # Generate predictions in an autoregressive way
    alphas = [1.035, 1.03, 1.025]
    weights = [1 / len(alphas)] * len(alphas)
    fday = datetime(2016, 4, 25)

    if enable_validation:
        fday = datetime(2016, 4, 25) - timedelta(days = 28)

    predictions_lst = Parallel(n_jobs = len(alphas), verbose = 1, pre_dispatch = len(alphas), batch_size = 1)(delayed(make_autoregressive_predictions)(testing_set_df, max_lags, train_cols, alpha, weight, m_lgb) for alpha, weight in zip(alphas, weights))
    sub = reduce(pd.DataFrame.add, predictions_lst)

    if enable_validation:
        sub = sub.reset_index()
        testing_set_df["demand"] = truth_df["sales"]
        training_set_df["demand"] = training_set_df["sales"]
        training_set_df.drop("d", axis = 1, inplace = True)
        testing_set_df.drop("d", axis = 1, inplace = True)
        evaluator = WRMSSEEvaluator(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, "2016-03-27")

        preds = pd.melt(sub, id_vars = ["id"], value_vars = [col for col in sub.columns if col.startswith("F")], var_name = "d", value_name = "sales")
        preds["date"] = preds["d"].str.replace("F", "").astype(np.int8).apply(lambda x: fday + timedelta(days = x - 1))
        preds.drop("d", axis = 1, inplace = True)
        preds.columns = ["id", "demand", "date"]
        preds = preds[["id", "date", "demand"]]

        with open("E:/preds_test5.pkl", "wb") as f:
            pickle.dump((preds, truth_df), f)

        preds_div = preds[["id", "date", "demand"]].pivot(index = "id", columns = "date", values = "demand").reset_index()
        truth_piv = truth_df[["id", "date", "sales"]].pivot(index = "id", columns = "date", values = "sales").reset_index()
        truth_piv.set_index("id", inplace = True)
        preds_div.set_index("id", inplace = True)
        print("Validation WRMSSE:", round(evaluator.wrmsse(preds_div, truth_piv, score_only = True), 6)) 
    else:
        # Generate submission file
        sub = sub.reset_index()
        sub2 = sub.copy()
        sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
        sub = pd.concat([sub, sub2], axis = 0, sort = False)
        sub.to_csv(PREDICTIONS_DIRECTORY_PATH_str + "draft_script_submission_3.csv", index = False)

    # Stop the timer and print the exectution time
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")

    # Public LB: 0.48372.
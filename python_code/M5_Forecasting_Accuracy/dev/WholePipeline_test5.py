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

import warnings
warnings.filterwarnings("ignore")

from dev.files_paths import *
from m5_forecasting_accuracy.data_loading.data_loader import DataLoader
from m5_forecasting_accuracy.preprocessing.PreprocessingStep import PreprocessingStep
from m5_forecasting_accuracy.models.lightgbm_wrapper import LGBMRegressor
from m5_forecasting_accuracy.model_utils.CustomTimeSeriesSplitter import CustomTimeSeriesSplitter
from m5_forecasting_accuracy.preprocessing.categorical_encoders import CategoricalFeaturesEncoder, OrdinalEncoder, GroupingEncoder, LeaveOneOutEncoder, TargetAvgEncoder
from m5_forecasting_accuracy.models_evaluation.wrmsse_metric import WRMSSEEvaluator

pd.set_option("display.max_columns", 100)

CAL_DTYPES = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
              "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32'}
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32"}
h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016, 4, 25) 
FIRST_DAY = 350

def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
        
    enable_validation = False

    dl = DataLoader()
    training_set_df, testing_set_df = dl.load_data_v2(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, "2016-03-27", enable_validation = enable_validation, first_day = FIRST_DAY)

    print("Training set shape:", training_set_df.shape)
    print("Testing set shape:", testing_set_df.shape)

    create_fea(training_set_df)
    training_set_df.dropna(inplace = True)
    print("df.shape (cell 14):", training_set_df.shape)

    cat_feats = ["item_id", "dept_id","store_id", "cat_id", "state_id", "event_name_1", "event_name_2", "event_type_1", "event_type_2"]
    useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
    train_cols = training_set_df.columns[~training_set_df.columns.isin(useless_cols)]
    X_train = training_set_df[train_cols]
    y_train = training_set_df["sales"]

    np.random.seed(777)

    fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)
    train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
    train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], categorical_feature=cat_feats, free_raw_data=False)
    fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds], categorical_feature=cat_feats, free_raw_data=False)# This is a random sample, we're not gonna apply any time series train-test-split tricks here!

    del training_set_df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()

    params = {
        "seed": 20,
        "objective" : "poisson",
        "metric" :"rmse",
        "learning_rate" : 0.075,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        "metric": ["rmse"],
        "verbosity": 1,
        "num_iterations" : 2300,
        "num_leaves": 128,
        "min_data_in_leaf": 60,
    }
    m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=100)
    m_lgb.save_model("model.lgb")
    
    # Feature importance
    importances = m_lgb.feature_importance("gain")
    features_names = m_lgb.feature_name()
    feature_importance_df = pd.DataFrame({"feature": features_names, "importance": importances}).sort_values(by = "importance", ascending = False).reset_index(drop = True)
    feature_importance_df.to_excel("E:/draft_lgb_importances.xlsx")

    # Predictions
    alphas = [1.035, 1.03, 1.025]
    weights = [1/len(alphas)]*len(alphas)
    sub = 0.

    for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

        te = testing_set_df.copy() #create_dt(False)
        cols = [f"F{i}" for i in range(1, 29)]

        for tdelta in range(0, 28):
            day = fday + timedelta(days=tdelta)
            print(icount, day)
            tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
            create_fea(tst)
            tst = tst.loc[tst.date == day , train_cols]
            te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev

        te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
        te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
        te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
        te_sub.fillna(0., inplace = True)
        te_sub.sort_values("id", inplace = True)
        te_sub.reset_index(drop=True, inplace = True)
        te_sub.to_csv(f"submission_{icount}.csv",index=False)
        if icount == 0 :
            sub = te_sub
            sub[cols] *= weight
        else:
            sub[cols] += te_sub[cols]*weight
        print(icount, alpha, weight)

    sub2 = sub.copy()
    sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
    sub = pd.concat([sub, sub2], axis=0, sort=False)
    sub.to_csv(PREDICTIONS_DIRECTORY_PATH_str + "draft_script_submission_3.csv",index=False)

    # Stop the timer and print the exectution time
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")

    # Public LB: 0.48897.

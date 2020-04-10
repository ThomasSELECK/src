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

def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

with open("E:/M5_Forecasting_Accuracy_cache/checkpoint1_v4.pkl", "rb") as f:
    training_set_df, target_df, testing_set_df, truth_df, sample_submission_df = pickle.load(f)

with open("E:/M5_Forecasting_Accuracy_cache/checkpoint4_v5.pkl", "rb") as f:
    preds = pickle.load(f)

training_set_df["demand"] = target_df["demand"]
testing_set_df["demand"] = truth_df["demand"]
e = WRMSSEEvaluator(training_set_df, testing_set_df)
print("Validation WRMSSE:", round(e.score(preds), 6))

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

# Data exploration
with open("E:/M5_Forecasting_Accuracy_cache/checkpoint1_v4.pkl", "rb") as f:
    training_set_df, target_df, testing_set_df, truth_df, sample_submission_df = pickle.load(f)

with open("E:/M5_Forecasting_Accuracy_cache/checkpoint4_v4_best_preds.pkl", "rb") as f:
    best_preds = pickle.load(f)

with open("E:/M5_Forecasting_Accuracy_cache/checkpoint4_v4.pkl", "rb") as f:
    preds = pickle.load(f)

preds.columns = ["id", "date", "preds"]
best_preds.columns = ["id", "date", "best_preds"]
preds = preds.merge(best_preds, how = "left", on = ["id", "date"])

truth_df["date"] = pd.to_datetime(truth_df["date"])
preds = preds.merge(truth_df, how = "left", on = ["id", "date"])

preds["preds_diff"] = np.abs(preds["best_preds"] - preds["preds"])
preds["best_preds_diff_to_tgt"] = np.abs(preds["best_preds"] - preds["demand"])
preds["preds_diff_to_tgt"] = np.abs(preds["preds"] - preds["demand"])
preds["best_solution"] = preds[["best_preds_diff_to_tgt", "preds_diff_to_tgt"]].min(axis = 1)
preds["best_solution_col"] = (preds["best_preds_diff_to_tgt"] == preds["best_solution"]).astype(np.int8).map({1: "best_preds", 0: "preds"})
preds["new_best_solution"] = preds["best_preds"]
preds["new_best_solution"].loc[preds["best_solution_col"] == "preds"] = preds["preds"].loc[preds["best_solution_col"] == "preds"]
preds2 = preds[["id", "date", "new_best_solution"]]
preds2.columns = ["id", "date", "demand"]
#print("Validation WRMSSE:", round(e.score(preds2), 6)) # Validation WRMSSE: 0.539443

# Best RMSE by date
best_preds_rmse_by_date_df = preds[["date", "best_preds", "demand"]].groupby("date").apply(lambda x: rmse(x["demand"], x["best_preds"])).reset_index()
best_preds_rmse_by_date_df.columns = ["date", "best_preds_rmse"]
preds_rmse_by_date_df = preds[["date", "preds", "demand"]].groupby("date").apply(lambda x: rmse(x["demand"], x["preds"])).reset_index()
preds_rmse_by_date_df.columns = ["date", "preds_rmse"]
rmse_by_date_df = best_preds_rmse_by_date_df.merge(preds_rmse_by_date_df, how = "left", on = "date")
rmse_by_date_df.to_excel("E:/rmse_by_date_df.xlsx", index = False)

# Best RMSE by item_id
tmp = preds["id"].str.split("_", expand = True)
preds["item_id"] = tmp[0].astype(str) + "_" + tmp[1] + "_" + tmp[2]

best_preds_rmse_by_item_id_df = preds[["item_id", "best_preds", "demand"]].groupby("item_id").apply(lambda x: rmse(x["demand"], x["best_preds"])).reset_index()
best_preds_rmse_by_item_id_df.columns = ["item_id", "best_preds_rmse"]
preds_rmse_by_item_id_df = preds[["item_id", "preds", "demand"]].groupby("item_id").apply(lambda x: rmse(x["demand"], x["preds"])).reset_index()
preds_rmse_by_item_id_df.columns = ["item_id", "preds_rmse"]
rmse_by_item_id_df = best_preds_rmse_by_item_id_df.merge(preds_rmse_by_item_id_df, how = "left", on = "item_id")
rmse_by_item_id_df.sort_values("best_preds_rmse", ascending = False, inplace = True)
rmse_by_item_id_df.to_excel("E:/rmse_by_item_id_df.xlsx", index = False)

# Best RMSE by id
best_preds_rmse_by_id_df = preds[["id", "best_preds", "demand"]].groupby("id").apply(lambda x: rmse(x["demand"], x["best_preds"])).reset_index()
best_preds_rmse_by_id_df.columns = ["id", "best_preds_rmse"]
preds_rmse_by_id_df = preds[["id", "preds", "demand"]].groupby("id").apply(lambda x: rmse(x["demand"], x["preds"])).reset_index()
preds_rmse_by_id_df.columns = ["id", "preds_rmse"]
rmse_by_id_df = best_preds_rmse_by_id_df.merge(preds_rmse_by_id_df, how = "left", on = "id")
rmse_by_id_df.sort_values("best_preds_rmse", ascending = False, inplace = True)
rmse_by_id_df.to_excel("E:/rmse_by_id_df.xlsx", index = False)

# Faire un graph par série
series_by_id_df = pd.pivot_table(preds, index = "date", columns = "id", values = ["best_preds", "preds", "demand"])
series_by_id_df.columns = [item[0] + "_" + item[1] for item in series_by_id_df.columns]
train_series_by_id_df = pd.pivot_table(target_df, index = "date", columns = "id", values = "demand")

train_series_by_id_df[["FOODS_2_360_WI_2_validation"]].plot.line()
plt.show()

plt.rcParams["figure.figsize"] = (20.0, 9.0)
for item_id in preds["item_id"].unique():
    ax = series_by_id_df.filter(regex = "^preds_" + item_id).plot.line(linestyle = "dotted")
    series_by_id_df.filter(regex = "^best_preds_" + item_id).plot.line(linestyle = "dashed", ax = ax)
    series_by_id_df.filter(regex = "^demand_" + item_id).plot.line(linewidth = 2, ax = ax)
    mean_sr = series_by_id_df.filter(regex = "^demand_" + item_id).mean(axis = 1).rename(item_id + "_avg").plot.line(linewidth = 2, color = "black", ax = ax)
    plt.xlabel("Date", fontsize = 20)
    plt.ylabel("Demand", fontsize = 20)
    plt.title("Predictions of time series for item: " + item_id, fontsize = 26)
    plt.legend(bbox_to_anchor = (1.0, 1.0), loc = "upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIRECTORY_PATH_str + item_id + ".png", dpi = 300)
    plt.clf()
    plt.close()

training_set_df["date"] = pd.to_datetime(training_set_df["date"])
testing_set_df["date"] = pd.to_datetime(testing_set_df["date"])
df = training_set_df.loc[training_set_df["id"] == "FOODS_2_360_WI_2_validation"]
df2 = testing_set_df.loc[testing_set_df["id"] == "FOODS_2_360_WI_2_validation"]
df.set_index("date", inplace = True)
df2.set_index("date", inplace = True)

cycle, trend = sm.tsa.filters.hpfilter(df["demand"], lamb = 6.25) # Annual lambda
gdp_decomp = df[["demand"]]
gdp_decomp["cycle"] = cycle
gdp_decomp["trend"] = trend

gdp_decomp.plot.line()
plt.show()

result = STL(df["demand"]).fit()
result.plot()
plt.show()

sell_prices = pd.read_csv(SELL_PRICES_PATH_str)
stv = pd.read_csv(SALES_TRAIN_PATH_str)
stv.drop(["item_id", "dept_id", "cat_id", "store_id", "state_id"], axis = 1, inplace = True)
stv.set_index("id", inplace = True)
cor_mat = np.corrcoef(stv)
cor_mat = pd.DataFrame(cor_mat, index = stv.index, columns = stv.index.tolist())
abs_cor_mat = np.abs(cor_mat)

count = 0

for i in range(abs_cor_mat.shape[0]):
    for j in range(i + 1, abs_cor_mat.shape[0]):
        if abs_cor_mat.iloc[i, j] > 0.8:
            count += 1
# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

from math import ceil

from sklearn.preprocessing import LabelEncoder
import multiprocessing as mp
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


def create_prices_features(self, X):
    # We can do some basic aggregations
    X["price_max"] = X.groupby(["store_id", "item_id"])["sell_price"].transform("max")
    X["price_min"] = X.groupby(["store_id", "item_id"])["sell_price"].transform("min")
    X["price_std"] = X.groupby(["store_id", "item_id"])["sell_price"].transform("std")
    X["price_mean"] = X.groupby(["store_id", "item_id"])["sell_price"].transform("mean")

    # and do price normalization (min/max scaling)
    X["price_norm"] = X["sell_price"] / X["price_max"]

    # Some items are can be inflation dependent
    # and some items are very "stable"
    X["price_nunique"] = X.groupby(["store_id", "item_id"])["sell_price"].transform("nunique")
    X["item_nunique"] = X.groupby(["store_id", "sell_price"])["item_id"].transform("nunique")

    # Now we can add price "momentum" (some sort of)
    # Shifted by week 
    # by month mean
    # by year mean
    X["price_momentum"] = X["sell_price"] / X.groupby(["store_id", "item_id"])["sell_price"].transform(lambda x: x.shift(1))
    X["price_momentum_m"] = X["sell_price"] / X.groupby(["store_id", "item_id", "month"])["sell_price"].transform("mean")
    X["price_momentum_y"] = X["sell_price"] / X.groupby(["store_id", "item_id", "year"])["sell_price"].transform("mean")

    return X

def create_lag_features(self, X):
    features_args_lst = [(28, i, "mean") for i in [7, 14, 30, 60, 180]] + [(28, i, "std") for i in [7, 14, 30, 60, 180]] + [(d_shift, d_window, "mean") for d_shift in [1, 7, 14] for d_window in [7, 14, 30, 60]]
    num_cores = mp.cpu_count()
    data_df = X[["id", "shifted_demand"]].copy()

    start_time = time.time()
    lag_features_df = pd.concat(Parallel(n_jobs = num_cores)(delayed(generate_lags)(data_df, d_shift, d_window, agg_type) for d_shift, d_window, agg_type in features_args_lst), axis = 1)
    X = pd.concat([X, lag_features_df], axis = 1)
    print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

    X = DataLoader.reduce_mem_usage(X, "data_df") # Need to take same dtypes as train
    
    return X

def generate_lags(data_df, d_shift, d_window, agg_type = "mean"):
    if agg_type == "mean":
        res =  data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift - 1).rolling(d_window).mean()).astype(np.float16)
        res = res.rename("rolling_mean_" + str(d_shift) + "_" + str(d_window))
    elif agg_type == "std":
        res = data_df.groupby(["id"])["shifted_demand"].transform(lambda x: x.shift(d_shift - 1).rolling(d_window).std()).astype(np.float16)
        res = res.rename("rolling_std_" + str(d_shift) + "_" + str(d_window))

    return res

def create_target_encoding_features(self, X):
    group_ids = [
        ["state_id"],
        ["store_id"],
        ["cat_id"],
        ["dept_id"],
        ["state_id", "cat_id"],
        ["state_id", "dept_id"],
        ["store_id", "cat_id"],
        ["store_id", "dept_id"],
        ["item_id"],
        ["item_id", "state_id"],
        ["item_id", "store_id"]
    ]

    for col in group_ids:
        col_name = "_" + "_".join(col) + "_"
        X["enc" + col_name + "mean"] = X.groupby(col)["shifted_demand"].transform("mean").astype(np.float16)
        X["enc" + col_name + "std"] = X.groupby(col)["shifted_demand"].transform("std").astype(np.float16)

    return X

# Call to main
if __name__ == "__main__":
    # Vars
    TARGET = 'demand'         # Our main target
    END_TRAIN = 1913         # Last day in train set
    MAIN_INDEX = ['id','d']  # We can identify item by these columns

    # Load Data
    print('Load Main Data')

    # Here are reafing all our data without any limitations and dtype modification
    train_df = pd.read_csv('D:/Projets_Data_Science/Competitions/Kaggle/M5_Forecasting_Accuracy/data/raw/sales_train_validation.csv')
    X = pd.read_csv('D:/Projets_Data_Science/Competitions/Kaggle/M5_Forecasting_Accuracy/data/raw/sell_prices.csv')
    calendar_df = pd.read_csv('D:/Projets_Data_Science/Competitions/Kaggle/M5_Forecasting_Accuracy/data/raw/calendar.csv')

    with open("E:/M5_Forecasting_Accuracy_cache/loaded_data_16052020.pkl", "rb") as f:
        training_set_df, target_df, testing_set_df, truth_df, orig_target_df, sample_submission_df = pickle.load(f)

    training_set_df["demand"] = orig_target_df["demand"]

    # Make Grid
    print('Create Grid')

    ########################### Apply on grid_df
    #################################################################################
    # lets read grid from 
    # https://www.kaggle.com/kyakovlev/m5-simple-fe
    # to be sure that our grids are aligned by index
    grid_df = pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl')
    grid_df[TARGET][grid_df['d']>(1913-28)] = np.nan
    base_cols = list(grid_df)

    icols =  [
                ['state_id'],
                ['store_id'],
                ['cat_id'],
                ['dept_id'],
                ['state_id', 'cat_id'],
                ['state_id', 'dept_id'],
                ['store_id', 'cat_id'],
                ['store_id', 'dept_id'],
                ['item_id'],
                ['item_id', 'state_id'],
                ['item_id', 'store_id']
                ]

    for col in icols:
        print('Encoding', col)
        col_name = '_'+'_'.join(col)+'_'
        grid_df['enc'+col_name+'mean'] = grid_df.groupby(col)[TARGET].transform('mean').astype(np.float16)
        grid_df['enc'+col_name+'std'] = grid_df.groupby(col)[TARGET].transform('std').astype(np.float16)

    keep_cols = [col for col in list(grid_df) if col not in base_cols]
    grid_df = grid_df[['id','d']+keep_cols]
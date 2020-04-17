#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the M5 Forecasting Accuracy competition                  #
#                                                                             #
# This file provides everything needed to load the data.                      #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2020-03-14                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import pickle
import time
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import STL
import multiprocessing as mp
from tqdm import tqdm

from m5_forecasting_accuracy.preprocessing.categorical_encoders import TargetAvgEncoder

class DataLoader():
    """
    This class is used to load the project's data.
    """

    def __init__(self, n_cores = -1):
        """
        This is the class' constructor.

        Parameters
        ----------
        None
            
        Returns
        -------
        None
        """
        
        if n_cores == -1:
            self.n_cores = mp.cpu_count()
        else:
            self.n_cores = n_cores

    @staticmethod
    def reduce_mem_usage(data_df, dataset_name, verbose = True):
        """
        This method reduces the memory footprint of each numeric feature 
        by choosing the most appropriate data type.

        Parameters
        ----------
        data_df: Pandas DataFrame
                Data we want to reduce memory footprint of.

        dataset_name: string
                Name of the dataset that will be printed in the logs.

        verbose: bool
                Whether to display statistics on memory footprint reduction.
            
        Returns
        -------
        None
        """

        print("    Reducing memory usage for '" + dataset_name + "'...")

        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        start_mem = data_df.memory_usage().sum() / 1024 ** 2    

        for col in data_df.columns:
            col_type = data_df[col].dtypes

            if col_type in numerics:
                c_min = data_df[col].min()
                c_max = data_df[col].max()

                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        data_df[col] = data_df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        data_df[col] = data_df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        data_df[col] = data_df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        data_df[col] = data_df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        data_df[col] = data_df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        data_df[col] = data_df[col].astype(np.float32)
                    else:
                        data_df[col] = data_df[col].astype(np.float64)

        end_mem = data_df.memory_usage().sum() / 1024 ** 2

        if verbose: 
            print("        Memory usage decreased from {:5.2f} MB to {:5.2f} MB ({:.1f}% reduction)".format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

        return data_df

    def _encode_categorical(self, df, cols):
        for col in cols:
            # Leave NaN as it is.
            le = LabelEncoder()
            not_null = df[col][df[col].notnull()]
            df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

        return df
           
    @staticmethod
    def time_series_decomposer(x):
        id, df = x
        result = STL(df).fit()
        result_df = pd.concat([result.trend, result.seasonal, result.resid], axis = 1)
        result_df.columns = ["trend", "seasonality", "residual"]
        result_df.reset_index(inplace = True)
        result_df["id"] = id

        return result_df[["id", "date", "trend", "seasonality", "residual"]]

    def load_data(self, calendar_data_path_str, sell_prices_data_path_str, sales_train_validation_data_path_str, sample_submission_data_path_str, train_test_date_split, enable_validation = True):
        """
        This function is a wrapper for the loading of the data.

        Parameters
        ----------
        training_set_path_str : string
                A string containing the path of the training set file.

        train_test_date_split : string
                A date where the data will be splitted into training and testing sets.

        enable_validation : bool (default = True)
                Whether to split training data into training and validation data.
            
        Returns
        -------
        training_set_df : pd.DataFrame
                A pandas DataFrame containing the training set.

        testing_set_df : pd.DataFrame
                A pandas DataFrame containing the testing set.
        """

        st = time.time()

        # Load the data
        print("Loading the data...")

        # Load train and test data
        print("    Reading files from disk...")

        calendar_df = pd.read_csv(calendar_data_path_str)
        calendar_df = self.reduce_mem_usage(calendar_df, "calendar_df")

        sell_prices_df = pd.read_csv(sell_prices_data_path_str)
        sell_prices_df = self.reduce_mem_usage(sell_prices_df, "sell_prices_df")

        sales_train_validation_df = pd.read_csv(sales_train_validation_data_path_str)
        sales_train_validation_df = self.reduce_mem_usage(sales_train_validation_df, "sales_train_validation_df")

        sample_submission_df = pd.read_csv(sample_submission_data_path_str)
        sample_submission_df = self.reduce_mem_usage(sample_submission_df, "sample_submission_df")
                
        print("    Reading files from disk... done in", round(time.time() - st, 3), "secs")

        # Encode some categorical features
        calendar_df = self._encode_categorical(calendar_df, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"])
        
        item_id_le = LabelEncoder()
        not_null = sales_train_validation_df["item_id"][sales_train_validation_df["item_id"].notnull()]
        sales_train_validation_df["item_id"] = pd.Series(item_id_le.fit_transform(not_null), index = not_null.index)
        not_null = sell_prices_df["item_id"][sell_prices_df["item_id"].notnull()]
        sell_prices_df["item_id"] = pd.Series(item_id_le.transform(not_null), index = not_null.index)

        # Get products table
        id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        products_df = sales_train_validation_df[id_columns]

        # Melt sales data: each column "d_[0-9]+" give the sales amount for the day "[0-9]+". So, we convert the data to get one line per day instead of one column.
        sales_train_validation_df = sales_train_validation_df.melt(id_vars = id_columns, var_name = "d", value_name = "demand")

        # Seperate test dataframes
        vals = sample_submission_df[sample_submission_df["id"].str.endswith("validation")]
        evals = sample_submission_df[sample_submission_df["id"].str.endswith("evaluation")]
            
        # Change column names
        DAYS_PRED = sample_submission_df.shape[1] - 1  # 28
        vals.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]
        evals.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]
        
        # Merge with products table
        evals["id"] = evals["id"].str.replace("_evaluation", "_validation")
        vals = vals.merge(products_df, how = "left", on = "id")
        evals = evals.merge(products_df, how = "left", on = "id")
        evals["id"] = evals["id"].str.replace("_validation", "_evaluation")

        # Melt the test data. So, we convert the data to get one line per day instead of one column.
        vals = vals.melt(id_vars = id_columns, var_name = "d", value_name = "demand")
        evals = evals.melt(id_vars = id_columns, var_name = "d", value_name = "demand")

        #sales_train_validation_df["part"] = "train"
        vals["part"] = "validation"
        evals["part"] = "evaluation"

        training_set_df = sales_train_validation_df
        testing_set_df = pd.concat([vals, evals], axis = 0)
        
        # get only a sample for fst training
        """training_set_df["d2"] = training_set_df["d"].str.replace("d_", "").apply(lambda x: int(x))
        training_set_df = training_set_df.loc[training_set_df["d2"] >= 815] # 1183
        training_set_df.drop("d2", axis = 1, inplace = True)"""

        # delete evaluation for now.
        testing_set_df = testing_set_df[testing_set_df["part"] != "evaluation"]
        testing_set_df.drop("part", axis = 1, inplace = True)
        gc.collect()
        
        # Merge calendar data
        calendar_df = calendar_df.drop(["weekday", "wday", "month", "year"], axis = 1)
        training_set_df = training_set_df.merge(calendar_df, how = "left", on = "d").drop("d", axis = 1)
        testing_set_df = testing_set_df.merge(calendar_df, how = "left", on = "d").drop("d", axis = 1)

        # Merge sell prices data
        training_set_df = training_set_df.merge(sell_prices_df, how = "left", on = ["store_id", "item_id", "wm_yr_wk"])
        testing_set_df = testing_set_df.merge(sell_prices_df, how = "left", on = ["store_id", "item_id", "wm_yr_wk"])

        # Remove leading zero values for each product as this can mean the product was not sold at that period 
        """product_release_df = sell_prices_df.groupby(["store_id", "item_id"])["wm_yr_wk"].min().reset_index()
        product_release_df.columns = ["store_id", "item_id", "release"]
        training_set_df = training_set_df.merge(product_release_df, how = "left", on = ["store_id", "item_id"])
        training_set_df = training_set_df.loc[training_set_df["wm_yr_wk"] >= training_set_df["release"]]
        training_set_df.drop("release", axis = 1, inplace = True)"""
                             
        # Decompose target into trend, seasonality and residuals
        """st = time.time()
        df = training_set_df[["id", "date", "demand"]]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace = True)
        chunks_lst = list(df.groupby("id")["demand"])
        with mp.Pool(self.n_cores) as pool:
            res = pool.map(DataLoader.time_series_decomposer, chunks_lst, chunksize = 5)
        res = pd.concat(res, axis = 0)
        df.reset_index(inplace = True)
        df = df.merge(res, how = "left", on = ["id", "date"])
        df2 = df[["id", "date", "trend"]]
        df2.columns = ["id", "date", "demand"]
        df2["date"] = df2["date"].dt.strftime("%Y-%m-%d")
        training_set_df.drop("demand", axis = 1, inplace = True)
        training_set_df = training_set_df.merge(df2, how = "left", on = ["id", "date"])
        print("Time series decomposition:", time.time() - st, "secs")"""

        # Shift demand column to avoid leakage
        tmp_df = pd.concat([training_set_df[["id", "date", "demand"]], testing_set_df[["id", "date", "demand"]]], axis = 0, ignore_index = True).reset_index(drop = True)
        tmp_df["shifted_demand"] = tmp_df.groupby(["id"])["demand"].transform(lambda x: x.shift(28))
        training_set_df = training_set_df.merge(tmp_df, how = "left", on = ["id", "date", "demand"])
        testing_set_df = testing_set_df.merge(tmp_df, how = "left", on = ["id", "date", "demand"])

        # Need to drop first rows (where shifted_demand is null)
        training_set_df = training_set_df.loc[~training_set_df["shifted_demand"].isnull()]
        training_set_df["shifted_demand"] = training_set_df["shifted_demand"].astype(np.int32)

        # Reduce memory usage
        training_set_df = self.reduce_mem_usage(training_set_df, "training_set_df")
        for col in testing_set_df.columns.tolist():
            testing_set_df[col] = testing_set_df[col].astype(training_set_df[col].dtype)

        del calendar_df, sell_prices_df
        gc.collect()

        # Remove outliers
        """means = training_set_df[["id", "demand"]].groupby("id").mean()
        stds = training_set_df[["id", "demand"]].groupby("id").std()
        maxs = training_set_df[["id", "demand"]].groupby("id").max()
        means.reset_index(inplace = True)
        stds.reset_index(inplace = True)
        maxs.reset_index(inplace = True)
        means.columns = ["id", "demand_mean"]
        stds.columns = ["id", "demand_std"]
        maxs.columns = ["id", "demand_max"]
        df = means.merge(stds, how = "left", on = "id").merge(maxs, how = "left", on = "id")
        df["new_max"] = df["demand_mean"] + 4 * df["demand_std"]
        df["new_max"] = df["new_max"].apply(lambda x: np.max([2, x]))
        training_set_df = training_set_df.merge(df[["id", "new_max"]], how = "left", on = "id")
        print("Removed", training_set_df["new_max"].loc[training_set_df["demand"] > training_set_df["new_max"]].shape[0], "outliers.")
        training_set_df["demand"].loc[training_set_df["demand"] > training_set_df["new_max"]] = training_set_df["new_max"].loc[training_set_df["demand"] > training_set_df["new_max"]]
        training_set_df.drop("new_max", axis = 1, inplace = True)"""

        # Remove rows where target is zero ?

        # Some products have constant demand equal to zero:
        ## FOODS_2_394_TX_3_validation
                          
        # Generate a validation set if enable_validation is True
        if enable_validation:
            print("Generating validation set...")
            
            # Split data on 'date' feature
            testing_set_df = training_set_df.loc[pd.to_datetime(training_set_df["date"]) > train_test_date_split].reset_index(drop = True)
            training_set_df = training_set_df.loc[pd.to_datetime(training_set_df["date"]) <= train_test_date_split].reset_index(drop = True)

            # Save target
            target_df = training_set_df[["id", "date", "demand"]]
            truth_df = testing_set_df[["id", "date", "demand"]]
                        
            print("Generating validation set... done")
        else:
            # Save target
            target_df = training_set_df[["id", "date", "demand"]]
            truth_df = None

        # Remove truth from data
        training_set_df.drop("demand", axis = 1, inplace = True)
        testing_set_df.drop("demand", axis = 1, inplace = True)

        print("Loading data... done")

        return training_set_df, target_df, testing_set_df, truth_df, sample_submission_df

    def load_data_v2(self, calendar_data_path_str, sell_prices_data_path_str, sales_train_validation_data_path_str, sample_submission_data_path_str, train_test_date_split, enable_validation = True, first_day = 1200, max_lags = 57):
        """
        This function is a wrapper for the loading of the data.

        Parameters
        ----------
        training_set_path_str : string
                A string containing the path of the training set file.

        train_test_date_split : string
                A date where the data will be splitted into training and testing sets.

        enable_validation : bool (default = True)
                Whether to split training data into training and validation data.
            
        Returns
        -------
        training_set_df : pd.DataFrame
                A pandas DataFrame containing the training set.

        testing_set_df : pd.DataFrame
                A pandas DataFrame containing the testing set.
        """

        st = time.time()
        calendar_dtypes_dict = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", "event_type_2": "category", "weekday": "category", "wm_yr_wk": "int16", 
                                "wday": "int16", "month": "int16", "year": "int16", "snap_CA": "float32", "snap_TX": "float32", "snap_WI": "float32"}
        sell_prices_dtypes_dict = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16", "sell_price": "float32"}

        # Load the data
        print("Loading the data...")

        # Load train and test data
        print("    Reading files from disk...")

        # For `sales_train_validation`
        tr_last = 1913
        valid_days_lst = [f"d_{day}" for day in range(first_day, tr_last + 1)]
        id_columns_lst = ["id", "item_id", "dept_id", "store_id", "cat_id", "state_id"]
        sales_train_validation_dtype_dict = {numcol: "float32" for numcol in valid_days_lst} 
        sales_train_validation_dtype_dict.update({col: "category" for col in id_columns_lst if col != "id"})

        calendar_df = pd.read_csv(calendar_data_path_str, dtype = calendar_dtypes_dict, parse_dates = ["date"])
        sell_prices_df = pd.read_csv(sell_prices_data_path_str, dtype = sell_prices_dtypes_dict)
        sales_train_validation_df = pd.read_csv(sales_train_validation_data_path_str, usecols = id_columns_lst + valid_days_lst, dtype = sales_train_validation_dtype_dict)
                                
        print("    Reading files from disk... done in", round(time.time() - st, 3), "secs")
        
        st = time.time()
        print("    Merging datasets...")
        # Add days for testing set
        for day in range(tr_last + 1, tr_last + 28 + 1):
            sales_train_validation_df[f"d_{day}"] = np.nan

        sales_train_validation_df = pd.melt(sales_train_validation_df, id_vars = id_columns_lst, value_vars = [col for col in sales_train_validation_df.columns if col.startswith("d_")], var_name = "d", value_name = "sales")
        sales_train_validation_df = sales_train_validation_df.merge(calendar_df, how = "left", on = "d", copy = False)
        sales_train_validation_df = sales_train_validation_df.merge(sell_prices_df, how = "left", on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
        print("    Merging datasets... done in", round(time.time() - st, 3), "secs")
        
        st = time.time()
        print("    Encoding categorical features...")
        # Encode categorical features to integer
        columns_to_encode_lst = [col for col in id_columns_lst if col != "id"]
        for col, col_dtype in list(sell_prices_dtypes_dict.items()) + list(calendar_dtypes_dict.items()):
            if col_dtype == "category":
                columns_to_encode_lst.append(col)
        columns_to_encode_lst = list(set(columns_to_encode_lst)) # Remove duplicates

        for col in columns_to_encode_lst:
            sales_train_validation_df[col] = sales_train_validation_df[col].cat.codes.astype("int16")
            sales_train_validation_df[col] -= sales_train_validation_df[col].min()
        print("    Encoding categorical features... done in", round(time.time() - st, 3), "secs")

        st = time.time()
        print("    Creating final datasets...")
        # Create training set
        valid_days_lst = [f"d_{day}" for day in range(first_day, tr_last + 1)]
        training_set_df = sales_train_validation_df.loc[sales_train_validation_df["d"].isin(valid_days_lst)].reset_index(drop = True)

        # Create testing set
        valid_days_lst = [f"d_{day}" for day in range(tr_last - max_lags, tr_last + 28 + 1)]
        testing_set_df = sales_train_validation_df.loc[sales_train_validation_df["d"].isin(valid_days_lst)].reset_index(drop = True)
        
        # Generate a validation set if enable_validation is True
        if enable_validation:
            print("Generating validation set...")
            
            # Split data on 'date' feature
            testing_set_df = training_set_df.loc[training_set_df["date"] > train_test_date_split].reset_index(drop = True)
            training_set_df = training_set_df.loc[training_set_df["date"] <= train_test_date_split].reset_index(drop = True)
                                    
            truth_df = testing_set_df[["id", "date", "sales"]]
            testing_set_df["sales"] = np.nan

            print("Generating validation set... done")
        else:
            truth_df = None
        
        print("    Creating final datasets... done in", round(time.time() - st, 3), "secs")

        print("Loading data... done")

        return training_set_df, testing_set_df, truth_df
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

import time
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataLoader():
    """
    This class is used to load the project's data.
    """

    def __init__(self):
        """
        This is the class' constructor.

        Parameters
        ----------
        None
            
        Returns
        -------
        None
        """

        pass

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
       
    def _merge_calendar(data, calendar):
        calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)
        return data.merge(calendar, how="left", on="d").drop("d", axis=1)


    def _merge_sell_prices(data, sell_prices):
        return data.merge(sell_prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])
    
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
        
        print("calendar_df shape:", calendar_df.shape)
        print("sell_prices_df shape:", sell_prices_df.shape)
        print("sales_train_validation_df shape:", sales_train_validation_df.shape)
        print("sample_submission_df shape:", sample_submission_df.shape)

        # calendar shape: (1969, 14)
        # sell_prices shape: (6841121, 4)
        # sales_train_val shape: (30490, 1919)
        # submission shape: (60980, 29)
        
        print("    Reading files from disk... done in", round(time.time() - st, 3), "secs")

        calendar_df = self._encode_categorical(calendar_df, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"])
        calendar_df = self.reduce_mem_usage(calendar_df, "calendar_df")

        sales_train_validation_df = self._encode_categorical(sales_train_validation_df, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],)
        sales_train_validation_df = self.reduce_mem_usage(sales_train_validation_df, "sales_train_validation_df")

        sell_prices_df = self._encode_categorical(sell_prices_df, ["item_id", "store_id"])
        sell_prices_df = self.reduce_mem_usage(sell_prices_df, "sell_prices_df")

        nrows = 27500000
        id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

        # Get products table
        products_df = sales_train_validation_df[id_columns]

        # Melt sales data: each column "d_[0-9]+" give the sales amount for the day "[0-9]+". So, we convert the data to get one line per day instead of one column.
        sales_train_validation_df = sales_train_validation_df.melt(id_vars = id_columns, var_name = "d", value_name = "demand")
        sales_train_validation_df = self.reduce_mem_usage(sales_train_validation_df, "sales_train_validation_df")

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

        #del sales_train_validation_df, vals, evals
        #gc.collect()

        # get only a sample for fst training
        training_set_df = training_set_df.loc[nrows:]

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

        # Reduce memory usage
        training_set_df = self.reduce_mem_usage(training_set_df, "training_set_df")
        testing_set_df = self.reduce_mem_usage(testing_set_df, "testing_set_df")

        del calendar_df, sell_prices_df
        gc.collect()
                     
        target_sr = training_set_df["demand"].reset_index(drop = True)
        """training_set_df.drop("demand", axis = 1, inplace = True)
        testing_set_df.drop("demand", axis = 1, inplace = True)"""
                  
        """
        # Generate a validation set if enable_validation is True
        if enable_validation:
            print("Generating validation set...")
            test_size_ratio = 0.2

            # Split data on 'RescuerID' feature as this feature is not overlapping train and test
            unique_rescuer_ids_npa = training_set_df["RescuerID"].unique()
            train_rescuer_ids_npa, test_rescuer_ids_npa = train_test_split(unique_rescuer_ids_npa, test_size = test_size_ratio, random_state = 2019)

            testing_set_df = training_set_df.loc[training_set_df["RescuerID"].isin(test_rescuer_ids_npa)]
            training_set_df = training_set_df.loc[training_set_df["RescuerID"].isin(train_rescuer_ids_npa)]
    
            # Extract truth / target
            truth_sr = testing_set_df[target_name_str]
            testing_set_df = testing_set_df.drop(target_name_str, axis = 1)

            # Reindex DataFrames
            training_set_df = training_set_df.reset_index(drop = True)
            testing_set_df = testing_set_df.reset_index(drop = True)
            truth_sr = truth_sr.reset_index(drop = True)

            print("Generating validation set... done")
        else:
            truth_sr = None

        # Extract target for training set
        target_sr = training_set_df[target_name_str]
        training_set_df = training_set_df.drop(target_name_str, axis = 1)"""

        print("Loading data... done")

        return training_set_df, target_sr, testing_set_df, sample_submission_df
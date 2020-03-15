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

    def reduce_mem_usage(self, data_df, dataset_name, verbose = True):
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
    
    def load_data(self, calendar_data_path_str, sell_prices_data_path_str, sales_train_validation_data_path_str, sample_submission_data_path_str, enable_validation = True):
        """
        This function is a wrapper for the loading of the data.

        Parameters
        ----------
        training_set_path_str : string
                A string containing the path of the training set file.

        testing_set_path_str : string
                A string containing the path of the testing set file.

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
        print("        Calendar has {} rows and {} columns".format(calendar_df.shape[0], calendar_df.shape[1]))

        sell_prices_df = pd.read_csv(sell_prices_data_path_str)
        sell_prices_df = self.reduce_mem_usage(sell_prices_df, "sell_prices_df")
        print("        Sell prices has {} rows and {} columns".format(sell_prices_df.shape[0], sell_prices_df.shape[1]))

        sales_train_validation_df = pd.read_csv(sales_train_validation_data_path_str)
        print("        Sales train validation has {} rows and {} columns".format(sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]))

        sample_submission_df = pd.read_csv(sample_submission_data_path_str)

        print("    Reading files from disk... done in", round(time.time() - st, 3), "secs")
        
        # Get products table
        products_df = sales_train_validation_df[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].drop_duplicates()

        # Melt sales data: each column "d_[0-9]+" give the sales amount for the day "[0-9]+". So, we convert the data to get one line per day instead of one column.
        sales_train_validation_df = pd.melt(sales_train_validation_df, id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name = "day", value_name = "demand")
        print("    Melted sales train validation has {} rows and {} columns".format(sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]))
        sales_train_validation_df = self.reduce_mem_usage(sales_train_validation_df, "sales_train_validation_df")
            
        # Seperate test dataframes
        test1_rows = [row for row in sample_submission_df["id"] if "validation" in row]
        test2_rows = [row for row in sample_submission_df["id"] if "evaluation" in row]
        test1_df = sample_submission_df[sample_submission_df["id"].isin(test1_rows)]
        test2_df = sample_submission_df[sample_submission_df["id"].isin(test2_rows)]
    
        # Change column names
        test1_df.columns = ["id", "d_1914", "d_1915", "d_1916", "d_1917", "d_1918", "d_1919", "d_1920", "d_1921", "d_1922", "d_1923", "d_1924", "d_1925", "d_1926", "d_1927", "d_1928",
                            "d_1929", "d_1930", "d_1931", "d_1932", "d_1933", "d_1934", "d_1935", "d_1936", "d_1937", "d_1938", "d_1939", "d_1940", "d_1941"]
        test2_df.columns = ["id", "d_1942", "d_1943", "d_1944", "d_1945", "d_1946", "d_1947", "d_1948", "d_1949", "d_1950", "d_1951", "d_1952", "d_1953", "d_1954", "d_1955", "d_1956", 
                            "d_1957", "d_1958", "d_1959", "d_1960", "d_1961", "d_1962", "d_1963", "d_1964", "d_1965", "d_1966", "d_1967", "d_1968", "d_1969"]
        
        # Merge with products table
        test2_df["id"] = test2_df["id"].str.replace("_evaluation", "_validation")
        test1_df = test1_df.merge(products_df, how = "left", on = "id")
        test2_df = test2_df.merge(products_df, how = "left", on = "id")
        test2_df["id"] = test2_df["id"].str.replace("_validation", "_evaluation")
    
        # Melt the test data. So, we convert the data to get one line per day instead of one column.
        test1_df = pd.melt(test1_df, id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name = "day", value_name = "demand")
        test2_df = pd.melt(test2_df, id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name = "day", value_name = "demand")
        train_df = sales_train_validation_df
        
        # get only a sample for fst training
        nrows = 27500000
        train_df = train_df.loc[nrows:]
    
        # drop some calendar features
        calendar_df.drop(["weekday", "wday", "month", "year"], inplace = True, axis = 1)
        
        train_df = pd.merge(train_df, calendar_df, how = "left", left_on = ["day"], right_on = ["d"])
        train_df.drop(["d", "day"], inplace = True, axis = 1)
        test1_df = pd.merge(test1_df, calendar_df, how = "left", left_on = ["day"], right_on = ["d"])
        test1_df.drop(["d", "day"], inplace = True, axis = 1)
        test2_df = pd.merge(test2_df, calendar_df, how = "left", left_on = ["day"], right_on = ["d"])
        test2_df.drop(["d", "day"], inplace = True, axis = 1)

        # get the sell price data (this feature should be very important)
        train_df = train_df.merge(sell_prices_df, on = ["store_id", "item_id", "wm_yr_wk"], how = "left")
        print("Our final dataset to train has {} rows and {} columns".format(train_df.shape[0], train_df.shape[1]))
        test1_df = test1_df.merge(sell_prices_df, on = ["store_id", "item_id", "wm_yr_wk"], how = "left")
        test2_df = test2_df.merge(sell_prices_df, on = ["store_id", "item_id", "wm_yr_wk"], how = "left")
    
        gc.collect()

          
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

        return train_df, test1_df, test2_df, sample_submission_df
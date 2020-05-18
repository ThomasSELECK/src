#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the M5 Forecasting Accuracy competition                  #
#                                                                             #
# This file defines the metric used in the competition.                       #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2020-03-15                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union
from tqdm.auto import tqdm as tqdm
import pickle
import time
from scipy.sparse import csr_matrix

class WRMSSEEvaluator(object):
    """
    Code inspired from https://www.kaggle.com/dhananjay3/wrmsse-evaluator-with-extra-features/
    and https://www.kaggle.com/sibmike/fast-clear-wrmsse-18ms/.

    This class defines the metric used in the competition.
    """

    def __init__(self, calendar_data_path_str, sell_prices_data_path_str, sales_train_validation_data_path_str, train_test_date_split):
        """
        This is the class' constructor.

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
        None
        """

        self.sales_train_validation_df = pd.read_csv(sales_train_validation_data_path_str)
        self.sales_train_validation_df.sort_values("id", inplace = True)
        self.calendar_df = pd.read_csv(calendar_data_path_str)
        self.sell_prices_df = pd.read_csv(sell_prices_data_path_str)

        # Generate a validation set if enable_validation is True
        train_cols_lst = [i for i in self.sales_train_validation_df.columns if not i.startswith("d_")] + self.calendar_df["d"].loc[self.calendar_df["date"] <= "2016-03-27"].tolist()
        test_cols_lst = [i for i in self.sales_train_validation_df.columns if not i.startswith("d_")] + self.calendar_df["d"].loc[(self.calendar_df["date"] > "2016-03-27") & (self.calendar_df["date"] <= "2016-04-24")].tolist()
        self.train_df = self.sales_train_validation_df[train_cols_lst].reset_index(drop = True)
        self.valid_df = self.sales_train_validation_df[test_cols_lst].reset_index(drop = True)
                           
        self.train_target_columns = [i for i in self.train_df.columns if i.startswith("d_")]
        self.weight_columns = self.train_df.iloc[:, -28:].columns.tolist()
        self.train_df["all_id"] = "all"
        self.valid_df["all_id"] = "all"
        
        self.id_columns = [i for i in self.train_df.columns if not i.startswith("d_")]
        self.valid_target_columns = [i for i in self.valid_df.columns if i.startswith("d_")]

        self.group_ids = ("all_id", "state_id", "store_id", "cat_id", "dept_id", "item_id", 
                 ["state_id", "cat_id"], ["state_id", "dept_id"], ["store_id", "cat_id"], 
                 ["store_id", "dept_id"], ["item_id", "state_id"], ["item_id", "store_id"])

        self.train_series = self.trans_30490_to_42840(self.train_df, self.train_target_columns)
        self.valid_series = self.trans_30490_to_42840(self.valid_df, self.valid_target_columns)
        self.roll_mat_csr, self.roll_index = self.create_rollup_index()
        self.group_ids_items = self.generate_group_ids_items()

        self.S = self.get_s()
        self.W = self.get_w()
        self.SW = self.W / np.sqrt(self.S)

        self.weights = self.get_weight_df() # Equivalent to get_w()
        self.scale = self.get_scale() # Equivalent to get_s()

        #self.train_series = None
        #self.train_df = None
        gc.collect()

    def trans_30490_to_42840(self, df, cols):
        """
        This method transform the 30490 series to all 42840 series.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        weights: pd.DataFrame
        """

        series_map_lst = []
        for group_id in self.group_ids:
            if type(group_id) == str:
                group_id = [group_id]

            tr = df.groupby(group_id)[cols].sum()
            if len(group_id) == 2:
                tr.index = pd.Series(tr.index.values).apply(lambda x: "--".join(x))

            series_map_lst.append(tr)
            
        res = pd.concat(series_map_lst)

        return res

    def generate_group_ids_items(self):
        """
        This method create a DataFrame of time series for each item in self.group_ids.

        Parameters
        ----------
        None

        Returns
        -------
        group_ids_items_df: pd.DataFrame
                All time series associated with each item from self.group_ids.
        """

        groups_ids = [["all_id"], ["state_id"], ["store_id"], ["cat_id"], ["dept_id"], ["item_id"], ["state_id", "cat_id"], ["state_id", "dept_id"], ["store_id", "cat_id"], ["store_id", "dept_id"], ["item_id", "state_id"], ["item_id", "store_id"]]
        group_ids_items_df = pd.DataFrame({"group_id": self.roll_index.get_level_values("level"), "time_series_ids": self.roll_index.get_level_values("id")})
        group_ids_items_df["group_id"] = group_ids_items_df["group_id"].apply(lambda x: groups_ids[x])
                   
        return group_ids_items_df

    def create_rollup_index(self):
        # List of categories combinations for aggregations as defined in docs:
        groups_ids = [["state_id"], ["store_id"], ["cat_id"], ["dept_id"], ["item_id"], ["state_id", "cat_id"], ["state_id", "dept_id"], ["store_id", "cat_id"], ["store_id", "dept_id"], ["item_id", "state_id"], ["item_id", "store_id"]]
        dummies_list = [self.sales_train_validation_df[group_id[0]] if len(group_id) == 1 else self.sales_train_validation_df[group_id[0]].astype(str) + "--" + self.sales_train_validation_df[group_id[1]].astype(str) for group_id in groups_ids]

        # First element Level_0 aggregation 'all_sales':
        dummies_df_list = [pd.DataFrame(np.ones(self.sales_train_validation_df.shape[0]).astype(np.int8), index = self.sales_train_validation_df.index, columns = ["all"]).T]

        # List of dummy dataframes:
        for i, cats in enumerate(dummies_list):
            dummies_df_list += [pd.get_dummies(cats, drop_first = False, dtype = np.int8).T]
    
        # Concat dummy dataframes in one go: Level is constructed for free.
        roll_mat_df = pd.concat(dummies_df_list, keys = list(range(12)), names = ["level", "id"])

        # Save values as sparse matrix & save index for future reference:
        roll_index = roll_mat_df.index
        roll_mat_csr = csr_matrix(roll_mat_df.values)

        return roll_mat_csr, roll_index

    def get_s(self):    
        # Rollup sales:
        sales_train_val = self.roll_mat_csr * self.sales_train_validation_df[self.train_target_columns].values
    
        # Find sales start index:
        start_no = np.argmax(sales_train_val > 0, axis = 1)
        
        # Replace days less than min day number with np.nan: Next code line is super slow:
        flag = np.dot(np.diag(1 / (start_no + 1)), np.tile([int(c.replace("d_", "")) for c in self.train_target_columns], (self.roll_mat_csr.shape[0], 1))) < 1
        sales_train_val = np.where(flag, np.nan, sales_train_val)

        # Denominator of RMSSE / RMSSE
        weight1 = np.nansum(np.diff(sales_train_val, axis = 1) ** 2, axis = 1) / (np.max([int(c.replace("d_", "")) for c in self.train_target_columns]) - start_no - 1)
    
        return weight1

    def get_w(self):
        data = self.sales_train_validation_df[["id", "store_id", "item_id"] + self.weight_columns]
        data = data.melt(id_vars = ["id", "store_id", "item_id"], var_name = "d", value_name = "sales")
        data = pd.merge(data, self.calendar_df, how = "left", on = ["d"])
        data = data.merge(self.sell_prices_df, on = ["store_id", "item_id", "wm_yr_wk"], how = "left")
        data["sales_usd"] = data["sales"] * data["sell_price"]
        data = data[["id", "sales_usd"]]

        # Calculate the total sales in USD for each item id:
        total_sales_usd = data.groupby(["id"], sort = False)["sales_usd"].apply(np.sum).values
    
        # Roll up total sales by ids to higher levels:
        weight2 = self.roll_mat_csr * total_sales_usd
        
        return 12 * weight2 / np.sum(weight2)

    def get_weight_df(self) -> pd.DataFrame:
        """
        This method returns weights for each of the 42840 series in a dataFrame.

        Parameters
        ----------
        None

        Returns
        -------
        weights_df: pd.DataFrame
        """

        day_to_week = self.calendar_df.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = self.train_df[["item_id", "store_id"] + self.weight_columns].set_index(["item_id", "store_id"])
        weight_df = (weight_df.stack().reset_index().rename(columns = {"level_2": "d", 0: "value"}))
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(self.sell_prices_df, how = "left", on = ["item_id", "store_id", "wm_yr_wk"])
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level = 2)["value"]
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop = True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis = 1, sort = False)

        weights_map_lst = []
        for group_id in self.group_ids:
            if type(group_id) == str:
                group_id = [group_id]

            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis = 1)
            lv_weight = lv_weight / lv_weight.sum()
            
            if len(group_id) == 2:
                lv_weight.index = pd.Series(lv_weight.index.values).apply(lambda x: "--".join(x))

            weights_map_lst.append(lv_weight)

        weights_df = pd.concat(weights_map_lst) / len(self.group_ids)

        return weights_df

    def get_scale(self):
        """
        This method computes the scaling factor for each series ignoring starting zeros.

        Parameters
        ----------
        None

        Returns
        -------
        : np.array
        """

        scales = []
        for i in range(len(self.train_series)):
            series = self.train_series.iloc[i].values
            series = series[np.argmax(series != 0):]
            scale = ((series[1:] - series[:-1]) ** 2).mean()
            scales.append(scale)

        return np.array(scales)

    def get_rmsse(self, valid_preds) -> pd.Series:
        """
        This method returns RMSSE scores for all 42840 series.

        Parameters
        ----------
        valid_preds: 

        Returns
        -------
        rmsse: pd.Series
        """

        score = ((self.valid_series - valid_preds) ** 2).mean(axis = 1)
        rmsse = (score / self.scale).map(np.sqrt)

        return rmsse

    def wrmsse(self, preds, y_true, score_only = False):
        '''
        preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
        y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
        sequence_length - np.array of size (42840,)
        sales_weight - sales weights based on last 28 days: np.array (42840,)
        '''
    
        # Reindex series to match roll_mat_csr index
        if isinstance(preds, pd.DataFrame) and isinstance(y_true, pd.DataFrame):
            preds = preds.reindex(self.sales_train_validation_df["id"]).values
            y_true = y_true.reindex(self.sales_train_validation_df["id"]).values

        if score_only: 
            return np.sum(np.sqrt(np.mean(np.square(self.roll_mat_csr.dot(y_true - preds)), axis = 1)) * self.SW) / 12
        else: 
            score_matrix = (np.square(self.roll_mat_csr.dot(y_true - preds)) * np.square(self.W)[:, None]) / self.S[:, None]
            score = np.sum(np.sqrt(np.mean(score_matrix, axis = 1))) / 12

            return score, score_matrix

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        """
        This method computes the WRMSSE score for predictions.

        Parameters
        ----------
        valid_preds: pd.DataFrame or np.array

        Returns
        -------
        : float
        """

        valid_preds.columns = ["id"] + self.valid_target_columns
        valid_preds = self.valid_df[self.id_columns].merge(valid_preds, how = "left", on = "id")
        self.valid_preds = self.trans_30490_to_42840(valid_preds, self.valid_target_columns)
        self.rmsse = self.get_rmsse(self.valid_preds)
        self.contributors = pd.concat([self.weights, self.rmsse], axis = 1, sort = False).prod(axis = 1)
        
        return np.sum(self.contributors)

    def generate_dataset_weights(self, data_df):
        weights = np.tile(self.weights.values.reshape((42840, 1)), 30490)
        weights = pd.DataFrame(np.multiply(self.roll_mat_csr.todense(), weights), index = self.weights.index, columns = self.sales_train_validation_df["id"].tolist())
        #weights = np.square(weights).div(self.scale, axis = 0).sum()
        weights = weights.div(np.sqrt(self.scale), axis = 0).sum()
        weights_dict = weights.to_dict()
        res = data_df["id"].map(weights_dict)
        #res *= 100000 # Scaling factor to allow LightGBM to train
        return res

    def lgb_feval(self, preds, dtrain):
        y_true = dtrain.get_label()
        y_true = y_true.reshape((self.sales_train_validation_df.shape[0], -1))
        preds = preds.reshape((self.sales_train_validation_df.shape[0], -1))

        # Only keep last 28 days
        preds = preds[:, -28:]
        y_true = y_true[:, -28:]
        score = self.wrmsse(preds, y_true, score_only = True)
        
        return "WRMSSE", score, False

    def lgb_fobj(self, preds, dtrain):
        y_true = dtrain.get_label()
        weights = dtrain.get_weight()
        
        grad = -2 * weights * (y_true - preds)
        hess = 2 * weights

        return grad, hess
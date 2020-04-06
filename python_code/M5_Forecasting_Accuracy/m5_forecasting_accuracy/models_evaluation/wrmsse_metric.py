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

import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union
from tqdm.auto import tqdm as tqdm
import pickle
import time

class WRMSSEEvaluator(object):
    """
    Code inspired from https://www.kaggle.com/dhananjay3/wrmsse-evaluator-with-extra-features/.

    This class defines the metric used in the competition.
    """

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame):
        """
        This is the class' constructor.

        Parameters
        ----------
        train_df: pd.DataFrame

        valid_df: pd.DataFrame

        calendar: pd.DataFrame

        prices: pd.DataFrame

        Returns
        -------
        None
        """

        self.train_df = train_df.copy() # Do this as train_df will be modified
        self.valid_df = valid_df

        self.train_df["all_id"] = "all"
        self.valid_df["all_id"] = "all"
        self.train_df["date"] = pd.to_datetime(self.train_df["date"])
        self.valid_df["date"] = pd.to_datetime(self.valid_df["date"])

        if "d" not in self.train_df.columns.tolist():
            self.train_df["d"] = (self.train_df["date"] - pd.to_datetime("2011-01-29")).dt.days + 1
        if "d" not in self.valid_df.columns.tolist():
            self.valid_df["d"] = (self.valid_df["date"] - pd.to_datetime("2011-01-29")).dt.days + 1

        self.group_ids = ("all_id", "state_id", "store_id", "cat_id", "dept_id", "item_id", 
                 ["state_id", "cat_id"], ["state_id", "dept_id"], ["store_id", "cat_id"], 
                 ["store_id", "dept_id"], ["item_id", "state_id"], ["item_id", "store_id"])

        self.id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "all_id"]

        self.train_target_columns = ["d_" + str(c) for c in self.train_df["d"].unique().tolist()]
        self.valid_target_columns = ["d_" + str(c) for c in self.valid_df["d"].unique().tolist()]
        self.weight_columns = self.train_target_columns[-28:]

        self.train_series = self.trans_30490_to_42840(self.train_df)
        self.valid_series = self.trans_30490_to_42840(self.valid_df)
        self.weights = self.get_weight_df()
        self.scale = self.get_scale()
        self.train_id_date = self.train_df[["id", "date"]].reset_index(drop = True)
        self.valid_id_date = self.valid_df[["id", "date"]].reset_index(drop = True)
        self.train_series = None
        self.train_df = None

        gc.collect()

    def trans_30490_to_42840(self, df):
        """
        This method transform the 30490 series to all 42840 series.

        Parameters
        ----------
        df: 

        Returns
        -------
        weights: pd.DataFrame
        """

        series_map_lst = []
        for i, group_id in enumerate(self.group_ids):
            if type(group_id) == str:
                group_id = [group_id]

            tr = pd.pivot_table(df[group_id + ["d", "demand"]], index = group_id, columns = "d", values = "demand").reset_index()

            if len(group_id) == 1:
                tr.index = tr[group_id[0]].astype(str)
            elif len(group_id) == 2:
                tr.index = tr[group_id[0]].astype(str) + "--" + tr[group_id[1]]

            tr.drop(group_id, axis = 1, inplace = True)
            tr.columns = ["d_" + str(c) for c in tr.columns]   
            series_map_lst.append(tr)
            
        res = pd.concat(series_map_lst)
           
        return res
    
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

        weight_df = self.train_df[["item_id", "store_id", "d", "demand", "sell_price"]].loc[self.train_df["d"] > self.train_df["d"].max() - 28]
        weight_df["value"] = weight_df["demand"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level = 2)["value"]
        tmp = self.train_df[["item_id", "store_id"]].drop_duplicates()
        weight_df = weight_df.loc[zip(tmp["item_id"], tmp["store_id"]), :].reset_index(drop = True)
        weight_df.columns = ["d_" + str(c) for c in weight_df.columns.tolist()]
        weight_df = pd.concat([self.train_df[self.id_columns].drop_duplicates(), weight_df], axis = 1, sort = False)
        weight_df["all_id"] = "all"

        weights_map_lst = []
        for i, group_id in enumerate(self.group_ids):
            if type(group_id) == str:
                group_id = [group_id]

            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis = 1)
            lv_weight = lv_weight / lv_weight.sum()
            lv_weight = lv_weight.reset_index()

            if len(group_id) == 1:
                lv_weight.index = lv_weight[group_id[0]].astype(str)
            elif len(group_id) == 2:
                lv_weight.index = lv_weight[group_id[0]].astype(str) + "--" + lv_weight[group_id[1]]

            lv_weight.drop(group_id, axis = 1, inplace = True)
            lv_weight.columns = ["d_" + str(c) for c in lv_weight.columns]   
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
        for i in tqdm(range(len(self.train_series))):
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

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns = self.valid_target_columns)

        valid_preds = self.valid_df[self.id_columns + ["date"]].merge(valid_preds, how = "left", on = ["id", "date"])
        valid_preds["d"] = (valid_preds["date"] - pd.to_datetime("2011-01-29")).dt.days + 1
        valid_preds = self.trans_30490_to_42840(valid_preds)
        self.rmsse = self.get_rmsse(valid_preds)
        self.contributors = pd.concat([self.weights, self.rmsse], axis = 1, sort = False).prod(axis = 1)

        return np.sum(self.contributors)

    def lgb_feval(self, preds, dtrain):
        if preds.shape[0] == self.train_df.shape[0]: # We are evaluating training set
            tmp = self.train_id_date.copy()
            preds = tmp.assign(demand = preds)
            score = self.score(preds)
        else: # We are evaluating validation set
            tmp = self.valid_id_date.copy()
            preds = tmp.assign(demand = preds)
            score = self.score(preds)
        
        return "WRMSSE", score, False

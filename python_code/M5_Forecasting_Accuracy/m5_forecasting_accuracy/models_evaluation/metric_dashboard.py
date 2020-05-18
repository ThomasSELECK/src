#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the M5 Forecasting Accuracy competition                  #
#                                                                             #
# This file defines a dashboard used to monitor the competition metric.       #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2020-03-15                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import gc
import time
from datetime import datetime
from datetime import timedelta

from sklearn import preprocessing
import lightgbm as lgb

from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm

class WRMSSEDashboard(object):
    """
    Code inspired from https://www.kaggle.com/tnmasui/m5-wrmsse-evaluation-dashboard.

    This class defines the metric used in the competition.
    """

    def __init__(self, plots_directory_path_str):
        """
        This is the class' constructor.

        Parameters
        ----------
        plots_directory_path_str: string
                Path where the plots will be stored.

        Returns
        -------
        None
        """

        self.plots_directory_path_str = plots_directory_path_str
        plt.rcParams["figure.figsize"] = (20.0, 9.0)

    def _create_viz_df(self, df, date_df):
        """
        This method computes a DataFrame for plotting a time series.

        Parameters
        ----------
        df: pd.DataFrame

        date_df: pd.DataFrame

        Returns
        -------
        df: pd.DataFrame
        """

        df = df.T.reset_index()
        df = df.rename(columns = {"index": "d"})
        df = df.merge(date_df, how = "left", on = "d")
        df = df.set_index("date")
        df = df.drop(["d"], axis = 1)
    
        return df

    def create_dashboard(self, evaluator, valid_pred_mlt_df):
        """
        This method creates multiple plots of WRMSSE at different aggregation
        levels.

        Parameters
        ----------
        evaluator: WRMSSEEvaluator object
                Evaluator that will be used to generate the dashboard.

        valid_pred_mlt_df: pd.DataFrame
                DataFrame containing the predictions we want to evaluate.

        Returns
        -------
        None
        """

        WRMSSEE = evaluator.score(valid_pred_mlt_df) # Need a DataFrame with shape (30490, 29) as input (ID + 28 date cols)

        all_rmsse_df = evaluator.contributors.copy().reset_index()
        all_rmsse_df.columns = ["time_series_ids", "wrmsse"]
        all_rmsse_df = all_rmsse_df.merge(evaluator.group_ids_items, how = "left", on = "time_series_ids")
        all_rmsse_df["group_id"] = all_rmsse_df["group_id"].apply(lambda x: ";".join(x))
        all_rmsse_sr = all_rmsse_df.groupby("group_id")["wrmsse"].sum() * 12

        wrmsses = [all_rmsse_sr.mean()] + all_rmsse_sr.to_list()
        labels = ["Overall"] + [f"Level {i}" for i in range(1, 13)]

        ## WRMSSE by Level
        fig = plt.figure(figsize = (20, 9))
        ax = sns.barplot(x = labels, y = wrmsses)
        ax.set_title("WRMSSE by Level", fontsize = 20, fontweight = "bold")
        ax.set_xlabel("")
        ax.set_ylabel("WRMSSE")

        # Add values on top of bars
        for index, val in enumerate(wrmsses):
            ax.text(index * 1, val + .01, round(val, 4), color = "black", ha = "center")

        fig.tight_layout()
        fig.savefig(self.plots_directory_path_str + "WRMSSE_by_Level.png", dpi = 300)
        plt.close(fig)
    
        # configuration array for the charts
        n_rows = [1, 1, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3]
        n_cols = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        group_ids = [
            "all_id",
            "state_id",
            "store_id",
            "cat_id",
            "dept_id",
            ["state_id", "cat_id"],
            ["state_id", "dept_id"],
            ["store_id", "cat_id"],
            ["store_id", "dept_id"],
            "item_id",
            ["item_id", "state_id"],
            ["item_id", "store_id"]
        ]
    
        lv_scores_df = evaluator.rmsse.copy().reset_index()
        lv_scores_df.columns = ["time_series_ids", "rmsse"]
        lv_weights_df = evaluator.weights.copy().reset_index()
        lv_weights_df.columns = ["time_series_ids", "weight"]
        lv_df = lv_scores_df.merge(lv_weights_df, how = "left", on = "time_series_ids")
        lv_df = lv_df.merge(evaluator.group_ids_items, how = "left", on = "time_series_ids")
        lv_df["group_id"] = lv_df["group_id"].apply(lambda x: ";".join(x))
        lv_df["weight"] *= 12

        for i, group_id in enumerate(group_ids):
            i += 1

            if isinstance(group_id, list):
                group_id = ";".join(group_id)
            scores = lv_df[["time_series_ids", "rmsse"]].loc[lv_df["group_id"] == group_id].set_index("time_series_ids")
            weights = lv_df[["time_series_ids", "weight"]].loc[lv_df["group_id"] == group_id].set_index("time_series_ids")

            if i > 1 and i < 9:
                if i < 7:
                    fig, axs = plt.subplots(1, 2, figsize = (20, 9))
                else:
                    fig, axs = plt.subplots(2, 1, figsize = (20, 9))
                
                ## RMSSE plot
                scores.plot.bar(width = 0.8, ax = axs[0], color = "g")
                axs[0].set_title(f"RMSSE", size = 14)
                axs[0].set(xlabel = "", ylabel = "RMSSE")

                if i >= 4:
                    axs[0].tick_params(labelsize = 8)

                for index, val in enumerate(scores["rmsse"]):
                    axs[0].text(index * 1, val + .01, round(val, 4), color = "black", ha = "center", fontsize = 10 if i == 2 else 8)
            
                ## Weight plot
                weights.plot.bar(width = 0.8, ax = axs[1])
                axs[1].set_title(f"Weight", size = 14)
                axs[1].set(xlabel = "", ylabel = "Weight")

                if i >= 4:
                    axs[1].tick_params(labelsize = 8)

                for index, val in enumerate(weights["weight"]):
                    axs[1].text(index * 1, val + .01, round(val, 2), color = "black", ha = "center", fontsize = 10 if i == 2 else 8)
             
                fig.suptitle(f"Level {i}: {evaluator.group_ids[i - 1]}", size = 24, y = 0.995, fontweight = "bold")
                fig.tight_layout(rect = [0, 0, 1, 0.95])
                fig.savefig(f"{self.plots_directory_path_str}Level_{i}_RMSSE_weight_plot.png", dpi = 300)
                plt.close(fig)

            train_df = evaluator.train_series.copy().reset_index().rename(columns = {"index": "time_series_ids"})
            valid_df = evaluator.valid_series.copy().reset_index().rename(columns = {"index": "time_series_ids"})
            valid_preds_df = evaluator.valid_preds.copy().reset_index().rename(columns = {"index": "time_series_ids"})
            train_df = train_df.merge(evaluator.group_ids_items, how = "left", on = "time_series_ids")
            train_df["group_id"] = train_df["group_id"].apply(lambda x: ";".join(x))
            valid_df = valid_df.merge(evaluator.group_ids_items, how = "left", on = "time_series_ids")
            valid_df["group_id"] = valid_df["group_id"].apply(lambda x: ";".join(x))
            valid_preds_df = valid_preds_df.merge(evaluator.group_ids_items, how = "left", on = "time_series_ids")
            valid_preds_df["group_id"] = valid_preds_df["group_id"].apply(lambda x: ";".join(x))

            trn = train_df.filter(regex = "d_.*").loc[train_df["group_id"] == group_id].iloc[:, -28 * 3:]
            val = valid_df.filter(regex = "d_.*").loc[valid_df["group_id"] == group_id]
            pred = valid_preds_df.filter(regex = "d_.*").loc[valid_preds_df["group_id"] == group_id]

            date_df = evaluator.calendar_df[["d", "date"]]
            date_df["date"] = pd.to_datetime(date_df["date"])
            trn = self._create_viz_df(trn, date_df)
            val = self._create_viz_df(val, date_df)
            pred = self._create_viz_df(pred, date_df)

            n_cate = trn.shape[1] if i < 7 else 9

            fig, axs = plt.subplots(n_rows[i - 1], n_cols[i - 1], figsize = (20, 9))
            if i > 1:
                axs = axs.flatten()

            ## Time series plot
            for k in range(0, n_cate):
                ax = axs[k] if i > 1 else axs

                trn.iloc[:, k].plot(ax = ax, label = "train")
                val.iloc[:, k].plot(ax = ax, label = "valid")
                pred.iloc[:, k].plot(ax = ax, label = "pred")
                ax.set_title(f"Series name: {scores.index[k]}  RMSSE: {scores['rmsse'].iloc[k]:.4f}", size = 14)
                ax.set(xlabel = "", ylabel = "sales")
                ax.tick_params(labelsize = 8)
                ax.legend(loc = "upper left", prop = {"size": 10})

            fig.suptitle(f"Level {i}: {group_ids[i - 1]}", size = 24, y = 0.995, fontweight = "bold")
            fig.tight_layout(rect = [0, 0, 1, 0.95])
            fig.savefig(f"{self.plots_directory_path_str}Level_{i}_ts_plot.png", dpi = 300)
            plt.close(fig)
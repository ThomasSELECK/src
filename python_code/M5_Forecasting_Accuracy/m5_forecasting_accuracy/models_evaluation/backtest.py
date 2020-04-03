#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the M5 Forecasting Accuracy competition                  #
#                                                                             #
# This file defines a backtest framework for evaluating a model's             #
# performances.                                                               #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2020-03-28                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import gc
import numpy as np
import pandas as pd

from .wrmsse_metric import WRMSSEEvaluator

class Backtest(object):
    """
    Code inspired from https://www.kaggle.com/chrisrichardmiles/m5-backtest-workflow-made-easy

    This class validates our prediction process by running it on different time periods, 
    with the last 28 days of the time period used as the validation set.
    This object will store the final scores in the .scores attribute.
    """

    def __init__(self, data_df, process_lst: list, process_names_lst = ["process"], verbose = False): 
        """
        This is the class' constructor.

        Parameters
        ----------
        data_df: pd.DataFrame
                DataFrame containing the sales data for training.

        process_lst: list
                List of prediction processes we want to evaluate.

        process_names: list (default = ["process"])
                List of the names coresponding to prediction processes in `process_lst`.

        verbose: bool (default = False)
                Whether to print the score.

        Returns
        -------
        None
        """

        self.data_df = data_df
        self.process_lst = process_lst
        self.process_names_lst = process_names_lst
        self.verbose = verbose
        self.valid_end_days = []
        self.scores = {name: [] for name in process_names_lst}        

    def score(self, model, days_back = 0): 
        """
        This method gives the score for the predictions if the data ended `days_back` days ago.

        Parameters
        ----------
        days_back: integer (default = 0)

        Returns
        -------
        None
        """

        if days_back != 0: 
            data_df = self.data_df.loc[self.data_df["d"] <= self.data_df["d"].max() - days_back] # Remove last `days_back` columns (remove last `days_back` days from data)
        else:   
            data_df = self.data_df

        train_df = data_df.loc[data_df["d"] <= data_df["d"].max() - 28].reset_index(drop = True)
        valid_df = data_df.loc[data_df["d"] > data_df["d"].max() - 28].reset_index(drop = True)
        self.valid_end_days.append(1913 - days_back)
        
        evaluator = WRMSSEEvaluator(train_df, valid_df)
        valid_df.drop("demand", axis = 1, inplace = True)
        
        for i in range(len(self.process_lst)): # For each prediction method
            # Generate predictions for validation set
            valid_preds = self.process_lst[i](train_df, valid_df, model)

            # Score the predictions
            score = evaluator.score(valid_preds)

            # Save result for later use
            self.scores[self.process_names_lst[i]].append(score)
            
            if self.verbose == True: 
                print(f"{self.process_names_lst[i]} had a score of {score} on validation period {1913 - days_back - 28} to {1913 - days_back}")
        
    def score_all(self, model, days_back = [0, 308]):
        """
        This method 

        Parameters
        ----------
        days_back: integer (default = [0, 308])

        Returns
        -------
        None
        """

        for i in range(len(days_back)): 
            self.score(model, days_back[i])

        return pd.DataFrame(self.scores, index = self.valid_end_days)
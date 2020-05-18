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
import pickle
from datetime import timedelta
from sklearn import metrics

class Backtest(object):
    """
    Code inspired from https://www.kaggle.com/chrisrichardmiles/m5-backtest-workflow-made-easy

    This class validates our prediction process by running it on different time periods, 
    with the last 28 days of the time period used as the validation set.
    This object will store the final scores in the .scores attribute.
    """

    def __init__(self, data_df, target_df, data_cache_path_str, train_nb_days, test_nb_days, process_lst: list, process_names_lst = ["process"], nb_folds = 3, verbose = False): 
        """
        This is the class' constructor.

        Parameters
        ----------
        data_df: pd.DataFrame
                DataFrame containing the sales data for training.

        target_df: pd.DataFrame
                DataFrame containing the associated target for training.

        train_nb_days: int
                Number of days of data to use for training set.

        test_nb_days: int
                Number of days of data to use for testing set.

        process_lst: list
                List of prediction processes we want to evaluate.

        process_names_lst: list (default = ["process"])
                List of the names coresponding to prediction processes in `process_lst`.

        nb_folds: int (default = 3)
                Number of validation folds to do.

        verbose: bool (default = False)
                Whether to print the score.

        Returns
        -------
        None
        """

        self.data_cache_path_str = data_cache_path_str
        self.train_nb_days = train_nb_days
        self.test_nb_days = test_nb_days
        self.process_lst = process_lst
        self.process_names_lst = process_names_lst
        self.nb_folds = nb_folds
        self.verbose = verbose
        self.valid_end_days = []
        self.scores = {name: [] for name in process_names_lst}

        # Save data to cache to save memory
        with open(self.data_cache_path_str + "data_cache.pkl", "wb") as f:
            pickle.dump((data_df, target_df), f)

        del data_df, target_df
        gc.collect()

    def _rmse(self, y_true, y_pred):
        return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    def _generate_validation_fold(self):
        """
        This method gives the score for the predictions if the data ended `days_back` days ago.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        for offset in range(self.nb_folds):
            # Load all the data from cache (do this to save memory)
            with open(self.data_cache_path_str + "data_cache.pkl", "rb") as f:
                data_df, target_df = pickle.load(f)

            # Generate train and test sets
            data_dates_lst = data_df["date"].drop_duplicates().sort_values().tolist()
            train_start_day = len(data_dates_lst) - ((self.nb_folds - offset) * self.test_nb_days + self.train_nb_days)
            train_end_day = train_start_day + self.train_nb_days
            test_start_day = train_end_day
            test_end_day = test_start_day + self.test_nb_days

            train_dates_lst = data_dates_lst[train_start_day:train_end_day]
            test_dates_lst = data_dates_lst[test_start_day:test_end_day]

            # Generate train and test labels
            training_set_df = data_df.loc[data_df["date"].isin(train_dates_lst)].reset_index(drop = True)
            testing_set_df = data_df.loc[data_df["date"].isin(test_dates_lst)].reset_index(drop = True)
            new_target_df = target_df.loc[data_df["date"].isin(train_dates_lst)].reset_index(drop = True)
            truth_df = target_df.loc[data_df["date"].isin(test_dates_lst)].reset_index(drop = True)

            # Reduce memory usage
            del data_df, target_df
            gc.collect()

            # Return result
            yield (training_set_df, testing_set_df, new_target_df, truth_df)

    def run(self): 
        """
        This method actually runs the backtest.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # For each fold
        for fold_idx, (fold_training_set_df, fold_testing_set_df, fold_target_df, fold_truth_df) in enumerate(self._generate_validation_fold()):
            train_test_date_split = fold_training_set_df["date"].max()
            eval_start_date = train_test_date_split - timedelta(days = self.test_nb_days)
            date_to_predict = train_test_date_split + timedelta(days = 1)
            print("Warning: date_to_predict offset should be computed dynamically. Currently fixed to 1.")

            # For each prediction method
            for process, process_name in zip(self.process_lst, self.process_names_lst):
                print("Running validation for process:", process_name, "on fold:", fold_idx, "...")

                # Train the model
                with open(self.data_cache_path_str + "data_bkp.pkl", "wb") as f:
                    pickle.dump((fold_training_set_df, fold_testing_set_df, fold_target_df, fold_truth_df), f)

                y_train = fold_target_df["demand"].reset_index(drop = True)
                model = process(train_test_date_split, eval_start_date)
                model.fit(fold_training_set_df, y_train)

                # Generate predictions for validation set
                preds = model.predict(fold_testing_set_df, date_to_predict)

                # Score the predictions
                preds2 = preds.copy()
                preds2.columns = ["id", "date", "preds"]
                preds_rmse_by_date_df = preds2.merge(fold_truth_df, how = "left", on = ["id", "date"])
                preds_rmse_by_date_df = preds_rmse_by_date_df[["date", "preds", "demand"]].groupby("date").apply(lambda x: self._rmse(x["demand"], x["preds"])).reset_index()
                preds_rmse_by_date_df.columns = ["date", "preds_rmse"]

                best_preds_piv = preds[["id", "date", "demand"]].pivot(index = "id", columns = "date", values = "demand").reset_index()
                truth_piv = fold_truth_df[["id", "date", "demand"]].pivot(index = "id", columns = "date", values = "demand").reset_index()
                truth_piv.set_index("id", inplace = True)
                best_preds_piv.set_index("id", inplace = True)
                best_preds_piv.columns = ["F" + str(i) for i in range(1, 29)]
                truth_piv.columns = ["F" + str(i) for i in range(1, 29)]
                validation_WRMSSE = round(model.evaluator.wrmsse(best_preds_piv, truth_piv, score_only = True), 6)

                # Save result for later use
                self.scores[process_name].append((fold_idx, preds_rmse_by_date_df, validation_WRMSSE))
            
                if self.verbose == True: 
                    print(process_name, "had a score of", validation_WRMSSE, "on validation period", fold_testing_set_df["date"].min(), "to", fold_testing_set_df["date"].max())

        metrics_lst = []
        for process_name, content in self.scores.items():
            for fold_idx, preds_rmse_by_date_df, validation_WRMSSE in content:
                preds_rmse_by_date_df["process_name"] = process_name
                preds_rmse_by_date_df["fold_idx"] = fold_idx
                preds_rmse_by_date_df["WRMSSE"] = validation_WRMSSE
                metrics_lst.append(preds_rmse_by_date_df)

        metrics_df = pd.concat(metrics_lst, axis = 0)
        metrics_df.set_index("date", inplace = True)

        return metrics_df
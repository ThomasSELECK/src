#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the M5 Forecasting Accuracy competition                  #
#                                                                             #
# This file contains everything needed to train a model and make predictions  #
# for one day in the future.                                                  #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2020-05-15                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import gc
import pickle
from m5_forecasting_accuracy.preprocessing.PreprocessingStep3 import PreprocessingStep3
from m5_forecasting_accuracy.models_evaluation.wrmsse_metric_fast import WRMSSEEvaluator
from m5_forecasting_accuracy.models.lightgbm_wrapper import LGBMRegressor
from dev.files_paths import *

class LGBMDayModel(object):
    """
    This class contains everything needed to train a model and make predictions for one day in the future.
    """

    def __init__(self, train_test_date_split, eval_start_date, dept_id):
        """
        This is the class' constructor.

        Parameters
        ----------
        train_test_date_split : string
                A date where the data will be splitted into training and testing sets.

        eval_start_date : string
                Date to use to split train data into training and validation sets for LightGBM.

        Returns
        -------
        None
        """

        self.prp = PreprocessingStep3(dt_col = "date", keep_last_train_days = 366) # 366 = shift + max rolling (365)

        self.lgb_params = {
            "boosting_type": "gbdt",
            "metric": "custom",
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "n_jobs": -1,
            "seed": 20,
            "learning_rate": 0.03,
            "subsample": 0.66,
            "bagging_freq": 1,
            "colsample_bytree": 0.77,
            "max_depth": -1,
            "num_leaves": 2 ** 11 - 1,
            "min_data_in_leaf": 2 ** 12 - 1,
            "max_bin": 100,
            "boost_from_average": False,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbosity": -1
        }

        self.categorical_features_lst = ["state_id", "store_id", "item_id", "event_type_1", "event_name_1", "weekday", "dept_id"]

        self.useless_features_lst = ["wm_yr_wk", "quarter", "id", "shifted_demand", "d"]
        self.evaluator = WRMSSEEvaluator(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, train_test_date_split, dept_id)
        #self.lgb_model = LGBMRegressor(self.lgb_params, early_stopping_rounds = 200, custom_eval_function = self.evaluator.lgb_feval, custom_objective_function = self.evaluator.lgb_fobj, maximize = False, nrounds = 3000, eval_split_type = "time", eval_start_date = eval_start_date, eval_date_col = "date", verbose_eval = 100, enable_cv = False, categorical_feature = self.categorical_features_lst)
        self.lgb_model = LGBMRegressor(self.lgb_params, early_stopping_rounds = 200, custom_eval_function = self.evaluator.lgb_feval, maximize = False, nrounds = 3000, eval_split_type = "time", eval_start_date = eval_start_date, eval_date_col = "date", verbose_eval = 100, enable_cv = False, categorical_feature = self.categorical_features_lst)
    
    def fit(self, X_train, y_train):
        """
        This method fits the model for the current day.

        Parameters
        ----------
        X_train : pd.DataFrame
                Data we want to use to train the model.

        y_train : pd.DataFrame
                Target associated with data.

        Returns
        -------
        None
        """
        
        y_train = y_train["demand"].reset_index(drop = True)
        X_train = self.prp.fit_transform(X_train, y_train) # y is not used here
        training_set_weights_sr = self.evaluator.generate_dataset_weights(X_train)
        X_train.drop(self.useless_features_lst, axis = 1, inplace = True)
        X_train = X_train.reset_index(drop = True)
                            
        gc.collect()

        """with open("E:/M5_Forecasting_Accuracy_cache/processed_data.pkl", "wb") as f:
            pickle.dump((X_train, y_train, training_set_weights_sr), f)"""

        self.lgb_model.fit(X_train, y_train, sample_weights = training_set_weights_sr)

        gc.collect()
    
        return self

    def predict(self, X_test, current_date):
        """
        This method makes predictions for the current day.

        Parameters
        ----------
        X_test : pd.DataFrame
                Data for which we want to make predictions.

        current_date : string
                Date associated with the day we're predicting.

        Returns
        -------
        predictions_df : pd.DataFrame
                Predictions made for the current day.
        """

        id_date = X_test[["id", "date"]].reset_index(drop = True)
        X_test = self.prp.transform(X_test)
        X_test.drop(["date"] + self.useless_features_lst, axis = 1, inplace = True)
        X_test = X_test.reset_index(drop = True)

        preds = self.lgb_model.predict(X_test)
        predictions_df = id_date.assign(demand = preds)
        #predictions_df = predictions_df.loc[predictions_df["date"] == current_date]

        gc.collect()

        return predictions_df
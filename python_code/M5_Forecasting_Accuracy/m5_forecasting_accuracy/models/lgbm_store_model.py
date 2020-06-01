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
import lightgbm as lgb
import time
import pandas as pd
import numpy as np
import random
from m5_forecasting_accuracy.preprocessing.PreprocessingStep4 import PreprocessingStep4
from m5_forecasting_accuracy.models_evaluation.wrmsse_metric_fast import WRMSSEEvaluator
from m5_forecasting_accuracy.models.lightgbm_wrapper import LGBMRegressor
from dev.files_paths import *

START_TRAIN = 0                # We can skip some rows (Nans/faster training)
END_TRAIN = 1913               # End day of our train set
P_HORIZON = 28                 # Prediction horizon
SEED = 42
AUX_MODELS = "E:/tmp/kernels/aux_model/"
VER = 1

# Seed to make all processes deterministic
def seed_everything(seed = 0):
    random.seed(seed)
    np.random.seed(seed)

class LGBMStoreModel(object):
    """
    This class contains everything needed to train a model and make predictions recursively for one store in the future.
    """

    def __init__(self, train_test_date_split, eval_start_date, store_id):
        """
        This is the class' constructor.

        Parameters
        ----------
        train_test_date_split : string
                A date where the data will be splitted into training and testing sets.

        eval_start_date : string
                Date to use to split train data into training and validation sets for LightGBM.

        store_id : string
                ID of the store we want to make predictions for.

        Returns
        -------
        None
        """

        self.store_id = store_id

        self.prp = PreprocessingStep4(dt_col = "date", keep_last_train_days = 209) # 366 = shift + max rolling (365)

        self.lgb_params = {
                    "boosting_type": "gbdt",
                    "objective": "tweedie",
                    "tweedie_variance_power": 1.1,
                    "metric": "rmse",
                    "subsample": 0.5,
                    "subsample_freq": 1,
                    "learning_rate": 0.03,
                    "num_leaves": 2**11-1,
                    "min_data_in_leaf": 2**12-1,
                    "feature_fraction": 0.5,
                    "max_bin": 100,
                    "n_estimators": 1400,
                    "boost_from_average": False,
                    "verbose": -1,
                } 
        self.lgb_params["seed"] = SEED

        self.categorical_features_lst = ["state_id", "store_id", "item_id", "event_type_1", "event_name_1", "weekday", "dept_id"]

        self.useless_features_lst = ["wm_yr_wk", "quarter", "id", "shifted_demand", "d"]
        #self.evaluator = WRMSSEEvaluator(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, train_test_date_split, dept_id)
        #self.lgb_model = LGBMRegressor(self.lgb_params, early_stopping_rounds = 200, custom_eval_function = self.evaluator.lgb_feval, custom_objective_function = self.evaluator.lgb_fobj, maximize = False, nrounds = 3000, eval_split_type = "time", eval_start_date = eval_start_date, eval_date_col = "date", verbose_eval = 100, enable_cv = False, categorical_feature = self.categorical_features_lst)
        #self.lgb_model = LGBMRegressor(self.lgb_params, early_stopping_rounds = 200, custom_eval_function = self.evaluator.lgb_feval, maximize = False, nrounds = 3000, eval_split_type = "time", eval_start_date = eval_start_date, eval_date_col = "date", verbose_eval = 100, enable_cv = False, categorical_feature = self.categorical_features_lst)
    
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
        
        with open("E:/fit.pkl", "wb") as f:
            pickle.dump((X_train, y_train), f)

        # Get grid for current store
        y_train = y_train.loc[X_train["store_id"] == self.store_id].reset_index(drop = True)
        X_train = X_train.loc[X_train["store_id"] == self.store_id].reset_index(drop = True)
        y_train = y_train["demand"].reset_index(drop = True)
        
        X_train = self.prp.fit_transform(X_train, y_train) # y is not used here
        #training_set_weights_sr = self.evaluator.generate_dataset_weights(X_train)
        #X_train.drop(self.useless_features_lst, axis = 1, inplace = True)
        #X_train = X_train.reset_index(drop = True)
        gc.collect()

        #self.lgb_model.fit(X_train, y_train, sample_weights = training_set_weights_sr)
        
        # Create LGBM datasets
        X_store_train = X_train.loc[X_train["d"] <= END_TRAIN]
        X_store_train.drop(["id", "d", "store_id"], axis = 1, inplace = True)
        X_store_valid = X_train.loc[(X_train["d"] <= END_TRAIN) & (X_train["d"] > (END_TRAIN - P_HORIZON))]
        X_store_valid.drop(["id", "d", "store_id"], axis = 1, inplace = True)
        features_columns = X_train.columns.tolist()
        train_data = lgb.Dataset(X_store_train, label = y_train.loc[X_train["d"] <= END_TRAIN])
        valid_data = lgb.Dataset(X_store_valid, label = y_train.loc[(X_train["d"] <= END_TRAIN) & (X_train["d"] > (END_TRAIN - P_HORIZON))])
    
        # Saving part of the dataset for later predictions
        # Removing features that we need to calculate recursively 
        X_train = X_train.loc[X_train["d"] > (END_TRAIN - 100)].reset_index(drop = True)
        keep_cols = [col for col in list(X_train) if "_tmp_" not in col]
        X_train = X_train[keep_cols]
        X_train.to_pickle("E:/tmp/kernels/aux_model/" + "test_" + self.store_id + ".pkl")
    
        # Launch seeder again to make lgb training 100% deterministic
        # with each "code line" np.random "evolves" 
        # so we need (may want) to "reset" it
        seed_everything(SEED)
        self.estimator = lgb.train(self.lgb_params, train_data, valid_sets = [valid_data], verbose_eval = 100)

        # Remove temporary files and objects  to free some hdd space and ram memory
        del train_data, valid_data, X_train, y_train
        gc.collect()
    
        return self

    def predict(self, X_test):
        """
        This method makes predictions for the current day.

        Parameters
        ----------
        X_test : pd.DataFrame
                Data for which we want to make predictions.

        Returns
        -------
        predictions_df : pd.DataFrame
                Predictions made for the current day.
        """

        # Predict
        # Create Dummy DataFrame to store predictions
        all_preds = pd.DataFrame()

        # Join back the Test dataset with a small part of the training data to make recursive features
        X_test["shifted_demand"] = np.nan

        # Timer to measure predictions time 
        main_time = time.time()

        # Loop over each prediction day as rolling lags are the most timeconsuming we will calculate it for whole day
        for predict_day in range(1, 29):    
            print("Predict | Day:", predict_day)
            start_time = time.time()

            # Make temporary grid to calculate rolling lags
            grid_df = X_test.copy()
            grid_df = self.prp.transform(grid_df)
        
            # Read all our models and make predictions for each day/store pairs
            model_path = "lgb_model_" + self.store_id + "_v" + str(VER) + ".bin" 
            model_path = AUX_MODELS + model_path
                
            grid_df2 = grid_df.loc[(grid_df["d"] == (END_TRAIN + predict_day)) & (grid_df["store_id"] == self.store_id)].drop(["id", "d", "store_id"], axis = 1)
            ids_df = grid_df[["id", "d"]].loc[(grid_df["d"] == (END_TRAIN + predict_day)) & (grid_df["store_id"] == self.store_id)]
            preds = self.estimator.predict(grid_df2)
            ids_df = ids_df.assign(tmp_demand = preds)
            ids_df["d"] = ids_df["d"].apply(lambda x: "d_" + str(x))
            X_test = X_test.merge(ids_df, how = "left", on = ["id", "d"])
            X_test["shifted_demand"].loc[~X_test["tmp_demand"].isnull()] = X_test["tmp_demand"].loc[~X_test["tmp_demand"].isnull()]
            X_test.drop("tmp_demand", axis = 1, inplace = True)
    
            # Make good column naming and add to all_preds DataFrame        
            temp_df = X_test[["id", "shifted_demand"]].loc[X_test["d"] == "d_" + str(END_TRAIN + predict_day)]
            temp_df.columns = ["id", "F" + str(predict_day)]
            if "id" in list(all_preds):
                all_preds = all_preds.merge(temp_df, on = ["id"], how = "left")
            else:
                all_preds = temp_df.copy()
        
            print("#"*10, " %0.2f min round |" % ((time.time() - start_time) / 60), " %0.2f min total |" % ((time.time() - main_time) / 60), " %0.2f day sales |" % (temp_df["F"+str(predict_day)].sum()))
            del temp_df
    
        all_preds = all_preds.reset_index(drop=True)

        gc.collect()

        return all_preds
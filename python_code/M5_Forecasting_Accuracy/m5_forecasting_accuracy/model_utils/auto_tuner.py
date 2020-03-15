#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains a class defining an auto-tuning system.                  #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-05-20                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import types
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold, StratifiedKFold
from multiprocessing import Pool
import multiprocessing as mp
import seaborn as sns
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
import hyperopt
from ml_metrics import quadratic_weighted_kappa
import pickle

from wrappers.lightgbm_wrapper import LGBMClassifier
from wrappers.blended_lightgbm_wrapper import BlendedLGBMClassifier
from wrappers.mlp_wrapper import MLPClassifier
from wrappers.blended_xgboost_wrapper import BlendedXGBClassifier

class AutoTuner(object):
    """
    The purpose of this class is to auto-tune a given machine learning model to increase its performances.
    """

    def __init__(self):
        """
        Class' constructor

        Parameters
        ----------
        None
                
        Returns
        -------
        None
        """

        pass

    def _tune_model_helper(self, n_tries, model_class, hyperparameters_dict, X_train, y_train, X_test, y_test):
        """
        This method is a helper for model tuning.

        Parameters
        ----------
        n_tries : positive integer
                Number of different hyperparameter sets to try.

        model_class : Python class
                Class of the model that will be used as predictive model.

        hyperparameters_dict : Python dictionary
                Dictionary containing hyperparameters to tune and corresponding range.
                
        X_train : Pandas DataFrame
                This is the data we will use to train the models.

        y_train : Pandas Series
                This is the associated target to X_train.

        X_test : Pandas DataFrame
                This is the data we will use to evaluate the models.

        y_test : Pandas Series
                This is the associated target to X_test.
                
        Returns
        -------
        None
        """

        def objective(params):                  
            xgb_params = {"objective": "multiclass",
                          "booster": "gbtree",
                          "metric": "qwk",
                          "num_class": 5,
                          "max_depth": int(params["max_depth"]),
                          "eta": params["eta"],
                          "subsample": params["subsample"],
                          "colsample_bytree": params["colsample_bytree"],
                          "gamma": params["gamma"],
                          "min_child_weight": params["min_child_weight"],
                          "verbosity": 0,
                          "silent": 1,
                          "seed": 17,
                          "nthread": 30}

            print("Params:", xgb_params)

            xgb = BlendedXGBClassifier(xgb_params, early_stopping_rounds = 150, eval_size = 0.2, eval_split_type = "random", verbose_eval = 100, nrounds = 10000)

            # Train the model
            xgb.fit(X_train, y_train)
    
            # Make predictions
            predictions_npa = xgb.predict(X_test)
    
            # Evaluate the model
            qwk = quadratic_weighted_kappa(y_test, predictions_npa)
            print(xgb_params)
            print("QWK = ", qwk)

            return -qwk # Return negative value as we want to maximize it

        # Stores all information about each trial and about the best trial
        trials = hyperopt.Trials()

        best = fmin(fn = objective, trials = trials, space = hyperparameters_dict, algo = tpe.suggest, max_evals = n_tries)

        return best, trials

    def _tune_model_helper3(self, n_tries, model_class, hyperparameters_dict, X_train, y_train, X_test, y_test):
        """
        This method is a helper for model tuning.

        Parameters
        ----------
        n_tries : positive integer
                Number of different hyperparameter sets to try.

        model_class : Python class
                Class of the model that will be used as predictive model.

        hyperparameters_dict : Python dictionary
                Dictionary containing hyperparameters to tune and corresponding range.
                
        X_train : Pandas DataFrame
                This is the data we will use to train the models.

        y_train : Pandas Series
                This is the associated target to X_train.

        X_test : Pandas DataFrame
                This is the data we will use to evaluate the models.

        y_test : Pandas Series
                This is the associated target to X_test.
                
        Returns
        -------
        None
        """

        def objective(params):                  
            mlp_params = {"layers_architecture": [int(l) for l in params["layers"]["n_units_layer"]],
                          "enable_batch_normalization": [True] * params["layers"]["n_layers"],
                          "activations": [params["activation"]] * params["layers"]["n_layers"],
                          "dropout_probas": [params["dropout"]] * params["layers"]["n_layers"],
                          "optimizer": "adam",
                          "learning_rate": params["learning_rate"],
                          "l2_regularization": params["l2_regularization"],
                          "metric": "rmse",
                          "epochs": 500,
                          "batch_size": 2048
                          }

            mlp = MLPClassifier(mlp_params, weights_saving_directory_path_str = "D:/Projets_Data_Science/Competitions/Kaggle/PetFinder.my_Adoption_Prediction/data/weights/")

            # Train the model
            mlp.fit(X_train, y_train)
    
            # Make predictions
            predictions_npa = mlp.predict(X_test)
    
            # Evaluate the model
            qwk = quadratic_weighted_kappa(y_test, predictions_npa)
            print(mlp_params)
            print("QWK = ", qwk)

            return -qwk # Return negative value as we want to maximize it

        # Stores all information about each trial and about the best trial
        trials = hyperopt.Trials()

        best = fmin(fn = objective, trials = trials, space = hyperparameters_dict, algo = tpe.suggest, max_evals = n_tries)

        return best, trials

    def _tune_model_helper2(self, n_tries, model_class, hyperparameters_dict, X_train, y_train, X_test, y_test):
        """
        This method is a helper for model tuning.

        Parameters
        ----------
        n_tries : positive integer
                Number of different hyperparameter sets to try.

        model_class : Python class
                Class of the model that will be used as predictive model.

        hyperparameters_dict : Python dictionary
                Dictionary containing hyperparameters to tune and corresponding range.
                
        X_train : Pandas DataFrame
                This is the data we will use to train the models.

        y_train : Pandas Series
                This is the associated target to X_train.

        X_test : Pandas DataFrame
                This is the data we will use to evaluate the models.

        y_test : Pandas Series
                This is the associated target to X_test.
                
        Returns
        -------
        None
        """

        def objective(params):
            lgb_params = {"application": "multiclass",
                  "boosting": "gbdt",
                  "metric": "qwk",
                  "num_class": 5,
                  "num_leaves": int(params["num_leaves"]),
                  "max_depth": -1,
                  "learning_rate": "{:.4f}".format(params["learning_rate"]),
                  "bagging_fraction": "{:.4f}".format(params["bagging_fraction"]),
                  "feature_fraction": "{:.4f}".format(params["feature_fraction"]),
                  "min_split_gain": "{:.4f}".format(params["min_split_gain"]),
                  "min_child_samples": int(params["min_child_samples"]),
                  "min_child_weight": "{:.4f}".format(params["min_child_weight"]),
                  "verbosity": -1,
                  "seed": 17,
                  "nthread": 16,
                  "device": "cpu"}

            lgbm = BlendedLGBMClassifier(lgb_params, early_stopping_rounds = 150, eval_size = 0.2, eval_split_type = "random", verbose_eval = 100, nrounds = 10000)

            # Train the model
            lgbm.fit(X_train, y_train)
    
            # Make predictions
            predictions_npa = lgbm.predict(X_test)
    
            # Evaluate the model
            qwk = quadratic_weighted_kappa(y_test, predictions_npa)
            print("QWK = ", qwk)

            return -qwk # Return negative value as we want to maximize it

        # Stores all information about each trial and about the best trial
        trials = hyperopt.Trials()

        best = fmin(fn = objective, trials = trials, space = hyperparameters_dict, algo = tpe.suggest, max_evals = n_tries)

        return best, trials

    def tune_model(self, n_tries, hyperparameters_dict, preprocessing_pipeline, model_class, X_train, y_train, X_test, y_test):
        """
        This method tunes a ML model.

        Parameters
        ----------
        n_tries : positive integer
                Number of different hyperparameter sets to try.

        hyperparameters_dict : Python dictionary
                Dictionary containing hyperparameters to tune and corresponding range.

        preprocessing_pipeline : Scikit Learn Pipeline or object inheriting Scikit Learn BaseEstimator class.
                This is the preprocessing that will be done on the data before tuning the model. The preprocessing 
                is done only once, for maximum efficiency.

        model_class : Python class
                Class of the model that will be used as predictive model.

        X_train : Pandas DataFrame
                This is the data we will use to train the models.

        y_train : Pandas Series
                This is the associated target to X_train.

        X_test : Pandas DataFrame
                This is the data we will use to evaluate the models.

        y_test : Pandas Series
                This is the associated target to X_test.
                
        Returns
        -------
        None
        """
                    
        # Preprocess the data
        X_train = preprocessing_pipeline.fit_transform(X_train, y_train)
        X_test = preprocessing_pipeline.transform(X_test)

        # Generate hyperparameters sets
        #self.hyperparameters_sets_lst = self._generate_random_hyperparameters_sets(hyperparameters_dict)

        # Actually do the tuning
        best, trials = self._tune_model_helper(n_tries, model_class, hyperparameters_dict, X_train, y_train, X_test, y_test)
        
        # Generate a DataFrame with results
        results_df = pd.DataFrame({"model": ["MLP"] * len(trials.tids), "iteration": trials.tids, "QWK": [-x["loss"] for x in trials.results]})
        results_df = pd.concat([results_df, pd.DataFrame(trials.vals, index = results_df.index)], axis = 1) # For LightGBM

        """
        results_df["activation"] = ["relu" if a == 0 else "tanh" for a in trials.vals["activation"]]
        results_df["dropout"] = trials.vals["dropout"]
        results_df["l2_regularization"] = trials.vals["l2_regularization"]
        results_df["learning_rate"] = trials.vals["learning_rate"]
        results_df["nb_layers"] = [i + 3 for i in trials.vals["layers"]]
        for i in range(1, 8):
            results_df["layer" + str(i)] = np.nan

        results_df.loc[results_df["nb_layers"] == 3, ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7"]] = pd.DataFrame({"layer1": trials.vals["n_units_layer_31"], "layer2": trials.vals["n_units_layer_32"], "layer3": trials.vals["n_units_layer_33"], "layer4": np.nan, "layer5": np.nan, "layer6": np.nan, "layer7": np.nan}).values
        results_df.loc[results_df["nb_layers"] == 4, ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7"]] = pd.DataFrame({"layer1": trials.vals["n_units_layer_41"], "layer2": trials.vals["n_units_layer_42"], "layer3": trials.vals["n_units_layer_43"], "layer4": trials.vals["n_units_layer_44"], "layer5": np.nan, "layer6": np.nan, "layer7": np.nan})
        results_df.loc[results_df["nb_layers"] == 5, ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7"]] = pd.DataFrame({"layer1": trials.vals["n_units_layer_51"], "layer2": trials.vals["n_units_layer_52"], "layer3": trials.vals["n_units_layer_53"], "layer4": trials.vals["n_units_layer_54"], "layer5": trials.vals["n_units_layer_55"], "layer6": np.nan, "layer7": np.nan})
        results_df.loc[results_df["nb_layers"] == 6, ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7"]] = pd.DataFrame({"layer1": trials.vals["n_units_layer_61"], "layer2": trials.vals["n_units_layer_62"], "layer3": trials.vals["n_units_layer_63"], "layer4": trials.vals["n_units_layer_64"], "layer5": trials.vals["n_units_layer_65"], "layer6": trials.vals["n_units_layer_66"], "layer7": np.nan})
        results_df.loc[results_df["nb_layers"] == 7, ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7"]] = pd.DataFrame({"layer1": trials.vals["n_units_layer_71"], "layer2": trials.vals["n_units_layer_72"], "layer3": trials.vals["n_units_layer_73"], "layer4": trials.vals["n_units_layer_74"], "layer5": trials.vals["n_units_layer_75"], "layer6": trials.vals["n_units_layer_76"], "layer7": trials.vals["n_units_layer_77"]}).values
        """
        
        results_df.sort_values("QWK", ascending = False, inplace = True)
        
        return results_df
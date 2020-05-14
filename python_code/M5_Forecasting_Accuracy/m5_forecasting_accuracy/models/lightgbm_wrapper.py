#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file is an advanced wrapper for the package LightGBM that is           #
# compatible with scikit-learn API.                                           #
#                                                                             #
# Credits for LightGBM: https://github.com/Microsoft/LightGBM                 #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-12-30                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score, cohen_kappa_score

from sklearn.model_selection import StratifiedKFold
from ml_metrics import quadratic_weighted_kappa
from collections import Counter

__all__ = ["LGBMClassifier",
           "LGBMRegressor",
          ]

class AbstractLGBMWrapper(ABC, BaseEstimator):
    """
    The purpose of this class is to provide a wrapper for a LightGBM model, with cross-validation for finding the best number of rounds.
    """
    
    def __init__(self, params, early_stopping_rounds, custom_eval_function = None, custom_objective_function = None, maximize = True, nrounds = 10000, random_state = 0, eval_split_type = "random", eval_size = 0.1, eval_start_date = None, eval_date_col = None, verbose_eval = 1, enable_cv = True, categorical_feature = None):
        """
        Class' constructor

        Parameters
        ----------
        params : dictionary
                This contains the parameters of the LightGBM model.

        early_stopping_rounds : integer
                This indicates the number of rounds to keep before stopping training when the score doesn't increase. If negative, disable this functionality.

        verbose_eval : positive integer
                This indicates the frequency of scores printing. E.g. 50 means one score printed every 50 rounds.

        custom_eval_function : function
                This is a function LightGBM will use as metric.

        custom_objective_function : function
                This is a function LightGBM will use as loss.

        maximize : boolean
                Indicates if the function customEvalFunction must be maximized or minimized. Not used when customEvalFunction is None.

        nrounds : integer
                Number of rounds for LightGBM training.

        random_state : zero or positive integer
                Seed used by LightGBM to ensure reproductibility.

        eval_split_type : string, either "random" or "time"
                This indicates the type of split for evaluation: random for iid samples and time for time series. Not used when enable_cv is True.

        eval_size : float between 0 and 1 or integer > 1.
                This indicates the size of the test set. Not used when enable_cv is True.

        eval_start_date : string or None (default = None)
                Only used when not None and `eval_split_type` = "time".
                Date used to determine the beginning of validation set.

        eval_date_col : string or None (default = None)
                Only used when `eval_start_date` is not None and `eval_split_type` = "time".
                Specify the column to use for splitting data with a date. If None, then
                the DataFrame index is used instead.

        verbose_eval : bool or int, optional (default = 1)
                If True, the eval metric on the valid set is printed at each boosting stage. If int, the eval metric on the valid set is 
                printed at every verbose_eval boosting stage. The last boosting stage or the boosting stage found by using early_stopping_rounds 
                is also printed.

        enable_cv : bool (default = True)
                If True, the best number of rounds is found using Cross Validation.

        categorical_feature : list (default = None)
                List of features that must be considered as categorical features by LightGBM.
                
        Returns
        -------
        None
        """
        
        # Class' attributes
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.custom_eval_function = custom_eval_function
        self.custom_objective_function = custom_objective_function
        self.maximize = maximize
        self.nrounds = nrounds
        self.random_state = random_state
        self.eval_split_type = eval_split_type
        self.eval_size = eval_size
        self.eval_start_date = eval_start_date
        self.eval_date_col = eval_date_col
        self.verbose_eval = verbose_eval
        self.enable_cv = enable_cv
        self.categorical_feature = categorical_feature

        self.lgb_model = None
        self.model_name = "LightGBM"
        self.is_qwk = False # Flag for Quadratic Weighted Kappa

    @staticmethod
    def cohen_kappa(preds, dtrain):
        labels = dtrain.get_label()
        return "Cohen's Kappa", cohen_kappa_score(labels, preds), True # f(preds: array, dtrain: Dataset) -> name: string, value: array, is_higher_better: bool

    @staticmethod
    def R2(preds, dtrain):
        labels = dtrain.get_label()
        return "R2", r2_score(labels, preds) * 100, True # f(preds: array, dtrain: Dataset) -> name: string, value: array, is_higher_better: bool
        
    def _fit(self, X, y, stratified_split = False, return_eval = False, sample_weights = None):
        """
        This method trains the LightGBM model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the training data.

        y : Pandas Series
                This is the target related to the training data.

        stratified_split : bool (default = False)
                This flag indicates whether to make a stratified split or not.

        return_eval : bool (default = False)
                This flag indicates whether to return eval data or not.

        sample_weights : Pandas Series
                Weights that will be given to training samples.
                
        Returns
        -------
        None
        """
        
        print("LightGBM training...")                       
        if self.enable_cv:
            if sample_weights is not None:
                if self.categorical_feature is not None:
                    dtrain = lgb.Dataset(X, label = y, categorical_feature = self.categorical_feature, weight = sample_weights)
                else:
                    dtrain = lgb.Dataset(X, label = y, weight = sample_weights)
            else:
                if self.categorical_feature is not None:
                    dtrain = lgb.Dataset(X, label = y, categorical_feature = self.categorical_feature)
                else:
                    dtrain = lgb.Dataset(X, label = y)
            watchlist = [dtrain]

            print("    Cross-validating LightGBM with seed: " + str(self.random_state) + "...")                
            if self.early_stopping_rounds < 0:
                cv_output = lgb.cv(self.params, dtrain, num_boost_round = self.nrounds, feval = self.custom_eval_function, fobj = self.custom_objective_function, verbose_eval = self.verbose_eval, show_stdv = True, stratified = stratified_split)
            else:
                cv_output = lgb.cv(self.params, dtrain, num_boost_round = self.nrounds, feval = self.custom_eval_function, fobj = self.custom_objective_function, early_stopping_rounds = self.early_stopping_rounds, verbose_eval = self.verbose_eval, show_stdv = True, stratified = stratified_split)

            cv_output_df = pd.DataFrame(cv_output)
            self.nrounds = cv_output_df[cv_output_df.filter(regex = ".*-mean").columns.tolist()[0]].index[-1] + 1

            print("    Final training of LightGBM with seed: " + str(self.random_state) + " and num rounds = " + str(self.nrounds) + "...")
            if self.early_stopping_rounds < 0:
                self.lgb_model = lgb.train(self.params, dtrain, self.nrounds, watchlist, feval = self.custom_eval_function, fobj = self.custom_objective_function, verbose_eval = self.verbose_eval)
            else:
                self.lgb_model = lgb.train(self.params, dtrain, self.nrounds, watchlist, feval = self.custom_eval_function, fobj = self.custom_objective_function, early_stopping_rounds = self.early_stopping_rounds, verbose_eval = self.verbose_eval)
        else:
            if self.eval_split_type == "random":
                if stratified_split:
                    X_train, X_eval, y_train, y_eval, weights_train, weights_eval = train_test_split(X, y, sample_weights, test_size = self.eval_size, random_state = self.random_state, stratify = y)
                else:
                    X_train, X_eval, y_train, y_eval, weights_train, weights_eval = train_test_split(X, y, sample_weights, test_size = self.eval_size, random_state = self.random_state)
            else:
                if self.eval_start_date is not None:
                    if self.eval_date_col is not None:
                        X_train = X.loc[X[self.eval_date_col] <= self.eval_start_date]
                        y_train = y.loc[X[self.eval_date_col] <= self.eval_start_date]
                        if sample_weights is not None:
                            weights_train = sample_weights.loc[X[self.eval_date_col] <= self.eval_start_date]
                        X_eval = X.loc[X[self.eval_date_col] > self.eval_start_date]
                        y_eval = y.loc[X[self.eval_date_col] > self.eval_start_date]
                        if sample_weights is not None:
                            weights_eval = sample_weights.loc[X[self.eval_date_col] > self.eval_start_date]
                        X_train.drop(self.eval_date_col, axis = 1, inplace = True)
                        X_eval.drop(self.eval_date_col, axis = 1, inplace = True)
                    else:
                        X_train = X.loc[X.index <= self.eval_start_date]
                        y_train = y.loc[X.index <= self.eval_start_date]
                        if sample_weights is not None:
                            weights_train = sample_weights.loc[X.index <= self.eval_start_date]
                        X_eval = X.loc[X.index > self.eval_start_date]
                        y_eval = y.loc[X.index > self.eval_start_date]
                        if sample_weights is not None:
                            weights_eval = sample_weights.loc[X.index > self.eval_start_date]
                else:
                    if self.eval_size > 1:
                        threshold = self.eval_size
                    else:
                        threshold = int(X.shape[0] * self.eval_size)

                    X_train = X.iloc[0:threshold]
                    y_train = y[0:threshold]
                    if sample_weights is not None:
                        weights_train = sample_weights[0:threshold]
                    X_eval = X.iloc[threshold:X.shape[0]]
                    y_eval = y[threshold:y.shape[0]]
                    if sample_weights is not None:
                        weights_eval = sample_weights[threshold:y.shape[0]]
            
            gc.collect()

            if sample_weights is not None:
                if self.categorical_feature is not None:
                    dtrain = lgb.Dataset(X_train, label = y_train, categorical_feature = self.categorical_feature, weight = weights_train)
                    dvalid = lgb.Dataset(X_eval, label = y_eval, categorical_feature = self.categorical_feature, weight = weights_eval)
                else:
                    dtrain = lgb.Dataset(X_train, label = y_train, weight = weights_train)
                    dvalid = lgb.Dataset(X_eval, label = y_eval, weight = weights_eval)
            else:
                if self.categorical_feature is not None:
                    dtrain = lgb.Dataset(X_train, label = y_train, categorical_feature = self.categorical_feature)
                    dvalid = lgb.Dataset(X_eval, label = y_eval, categorical_feature = self.categorical_feature)
                else:
                    dtrain = lgb.Dataset(X_train, label = y_train)
                    dvalid = lgb.Dataset(X_eval, label = y_eval)

            watchlist = [dtrain, dvalid]
            gc.collect()

            if self.early_stopping_rounds < 0:
                self.lgb_model = lgb.train(self.params, dtrain, self.nrounds, watchlist, feval = self.custom_eval_function, fobj = self.custom_objective_function, verbose_eval = self.verbose_eval)
            else:
                self.lgb_model = lgb.train(self.params, dtrain, self.nrounds, watchlist, feval = self.custom_eval_function, fobj = self.custom_objective_function, early_stopping_rounds = self.early_stopping_rounds, verbose_eval = self.verbose_eval)

            self.nrounds = self.lgb_model.best_iteration

        if return_eval:
            return X_eval, y_eval
        else:
            return None

    @abstractmethod
    def fit(self, X, y):
        """
        This method trains the LightGBM model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the training data.

        y : Pandas Series
                This is the target related to the training data.
                
        Returns
        -------
        None
        """

        raise NotImplementedError("Not yet implemented!")

    @abstractmethod
    def predict(self, X):
        """
        This method makes predictions using the previously trained model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing predictions for each sample of the testing set.
        """

        """# Sanity checks
        if self.lgb_model is None:
            raise ValueError("You MUST train the LightGBM model using fit() before attempting to do predictions!")

        print("Predicting outcome for testing set...")
        predictions_npa = self.lgb_model.predict(X)

        return predictions_npa"""

        raise NotImplementedError("Not yet implemented!")
    
    def plot_features_importance(self, importance_type = "gain", max_num_features = None, ignore_zero = True):
        """
        This method plots model's features importance.

        Parameters
        ----------
        importance_type : string, optional (default = "gain")
                How the importance is calculated. If "split", result contains numbers of times the feature is used in a model. 
                If "gain", result contains total gains of splits which use the feature.
                
        max_num_features : int or None, optional (default = None)
                Max number of top features displayed on plot. If None or < 1, all features will be displayed.
                
        ignore_zero : bool, optional (default = True)
                Whether to ignore features with zero importance.
                
        Returns
        -------
        None
        """

        lgb.plot_importance(self.lgb_model, importance_type = importance_type, max_num_features = max_num_features, ignore_zero = ignore_zero)
        plt.show()

    def get_features_importance(self, importance_type = "gain"):
        """
        This method gets model's features importance.

        Parameters
        ----------
        importance_type : string, optional (default = "gain")
                How the importance is calculated. If "split", result contains numbers of times the feature is used in a model. 
                If "gain", result contains total gains of splits which use the feature.
                                
        Returns
        -------
        feature_importance_df : Pandas Data Frame
                Feature importance of each feature of the dataset.
        """

        importance = self.lgb_model.feature_importance(importance_type = "gain")
        features_names = self.lgb_model.feature_name()

        feature_importance_df = pd.DataFrame({"feature": features_names, "importance": importance}).sort_values(by = "importance", ascending = False).reset_index(drop = True)

        return feature_importance_df

class LGBMClassifier(AbstractLGBMWrapper, ClassifierMixin):
    """
    The purpose of this class is to provide a wrapper for a LightGBM classifier.
    """
    
    def __init__(self, params, early_stopping_rounds, custom_eval_function = None, custom_objective_function = None, maximize = True, nrounds = 10000, random_state = 0, eval_split_type = "random", eval_size = 0.1, eval_start_date = None, eval_date_col = None, verbose_eval = 1, enable_cv = True):
        """
        Class' constructor

        Parameters
        ----------
        params : dictionary
                This contains the parameters of the LightGBM model.
                
        early_stopping_rounds : integer
                This indicates the number of rounds to keep before stopping training when the score doesn't increase. If negative, disable this functionality.

        verbose_eval : positive integer
                This indicates the frequency of scores printing. E.g. 50 means one score printed every 50 rounds.

        custom_eval_function : function
                This is a function LightGBM will use as metric.

        custom_objective_function : function
                This is a function LightGBM will use as loss.

        maximize : boolean
                Indicates if the function customEvalFunction must be maximized or minimized. Not used when customEvalFunction is None.

        nrounds : integer
                Number of rounds for LightGBM training. If negative, the model will look for the best value using cross-validation.

        random_state : zero or positive integer
                Seed used by LightGBM to ensure reproductibility.

        eval_split_type : string, either "random" or "time"
                This indicates the type of split for evaluation: random for iid samples and time for time series. Not used when enable_cv is True.

        eval_size : float between 0 and 1 or integer > 1.
                This indicates the size of the test set. Not used when enable_cv is True.

        eval_start_date : string or None (default = None)
                Only used when not None and `eval_split_type` = "time".

        eval_date_col : string or None (default = None)
                Only used when `eval_start_date` is not None and `eval_split_type` = "time".
                Specify the column to use for splitting data with a date. If None, then
                the DataFrame index is used instead.

        verbose_eval : bool or int, optional (default = 1)
                If True, the eval metric on the valid set is printed at each boosting stage. If int, the eval metric on the valid set is 
                printed at every verbose_eval boosting stage. The last boosting stage or the boosting stage found by using early_stopping_rounds 
                is also printed.

        enable_cv : bool (default = True)
                If True, the best number of rounds is found using Cross Validation.
                
        Returns
        -------
        None
        """
        
        # Call to superclass
        super().__init__(params, early_stopping_rounds, custom_eval_function, custom_objective_function, maximize, nrounds, random_state, eval_split_type, eval_size, eval_start_date, eval_date_col, verbose_eval, enable_cv)

        # Create LabelEncoder to avoid issues with messed up labels
        self._le = LabelEncoder()
        self._classes = None

    def fit(self, X, y):
        """
        This method trains the LightGBM model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the training data.

        y : Pandas Series
                This is the target related to the training data.
                
        Returns
        -------
        None
        """

        # Fit the label encoder and transform the target
        _y = self._le.fit_transform(y)
        self._classes = self._le.classes_

        # Call to superclass
        super()._fit(X, _y, stratified_split = True)

        return self

    def predict(self, X):
        """
        This method makes predictions using the previously trained model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing predictions for each sample of the testing set.
        """

        # Sanity checks
        if self.lgb_model is None:
            raise ValueError("You MUST train the LightGBM model using fit() before attempting to do predictions!")

        print("Predicting outcome for testing set...")
        predictions_npa = self.lgb_model.predict(X, num_iteration = self.lgb_model.best_iteration)
        
        if self.is_qwk: # If we use Quadratic Weighted Kappa as metric
            predictions_npa = self._optR.predict(predictions_npa).astype(np.int32)
        else:
            if len(predictions_npa.shape) == 2 and predictions_npa.shape[1] > 1:
                predictions_npa = np.argmax(predictions_npa, axis = 1)
            else:
                predictions_npa = (predictions_npa > 0.5).astype(np.int8)

            predictions_npa = self._le.inverse_transform(predictions_npa)

        return predictions_npa

    def predict_proba(self, X):
        """
        This method makes predictions using the previously trained model and return probabilities.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing probabilities of each class for each sample of the testing set.
        """

        # Sanity checks
        if self.lgb_model is None:
            raise ValueError("You MUST train the LightGBM model using fit() before attempting to do predictions!")

        print("Predicting outcome for testing set...")
        predictions_npa = self.lgb_model.predict(X, num_iteration = self.lgb_model.best_iteration)

        return predictions_npa

    @property
    def classes_(self):
        """
        This method gets the class label array.

        Parameters
        ----------
        None
                
        Returns
        -------
        self._classes : numpy array
                Numpy array containing the label for each class.
        """

        if self.lgb_model is None:
            raise ValueError("You MUST train the LightGBM model using fit() before attempting to do predictions!")

        return self._classes

class LGBMRegressor(AbstractLGBMWrapper, RegressorMixin):
    """
    The purpose of this class is to provide a wrapper for a LightGBM regressor.
    """
    
    def __init__(self, params, early_stopping_rounds, custom_eval_function = None, custom_objective_function = None, maximize = True, nrounds = 10000, random_state = 0, eval_split_type = "random", eval_size = 0.1, eval_start_date = None, eval_date_col = None, verbose_eval = 1, enable_cv = True):
        """
        Class' constructor

        Parameters
        ----------
        params : dictionary
                This contains the parameters of the LightGBM model.
                
        early_stopping_rounds : integer
                This indicates the number of rounds to keep before stopping training when the score doesn't increase. If negative, disable this functionality.

        verbose_eval : positive integer
                This indicates the frequency of scores printing. E.g. 50 means one score printed every 50 rounds.

        custom_eval_function : function
                This is a function LightGBM will use as metric.

        custom_objective_function : function
                This is a function LightGBM will use as loss.

        maximize : boolean
                Indicates if the function customEvalFunction must be maximized or minimized. Not used when customEvalFunction is None.

        nrounds : integer
                Number of rounds for LightGBM training. If negative, the model will look for the best value using cross-validation.

        random_state : zero or positive integer
                Seed used by LightGBM to ensure reproductibility.

        eval_split_type : string, either "random" or "time"
                This indicates the type of split for evaluation: random for iid samples and time for time series. Not used when enable_cv is True.

        eval_size : float between 0 and 1 or integer > 1.
                This indicates the size of the test set. Not used when enable_cv is True.

        eval_start_date : string or None (default = None)
                Only used when not None and `eval_split_type` = "time".

        eval_date_col : string or None (default = None)
                Only used when `eval_start_date` is not None and `eval_split_type` = "time".
                Specify the column to use for splitting data with a date. If None, then
                the DataFrame index is used instead.

        verbose_eval : bool or int, optional (default = 1)
                If True, the eval metric on the valid set is printed at each boosting stage. If int, the eval metric on the valid set is 
                printed at every verbose_eval boosting stage. The last boosting stage or the boosting stage found by using early_stopping_rounds 
                is also printed.

        enable_cv : bool (default = True)
                If True, the best number of rounds is found using Cross Validation.
                
        Returns
        -------
        None
        """
        
        # Call to superclass
        super().__init__(params, early_stopping_rounds, custom_eval_function, custom_objective_function, maximize, nrounds, random_state, eval_split_type, eval_size, eval_start_date, eval_date_col, verbose_eval, enable_cv)
        
    def fit(self, X, y, sample_weights = None):
        """
        This method trains the LightGBM model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the training data.

        y : Pandas Series
                This is the target related to the training data.

        sample_weights : Pandas Series
                Weights that will be given to training samples.
                
        Returns
        -------
        None
        """

        # Call to superclass
        super()._fit(X, y, sample_weights = sample_weights)

        return self

    def predict(self, X):
        """
        This method makes predictions using the previously trained model.

        Parameters
        ----------
        X : Pandas DataFrame
                This is the testing data we want to make predictions on.
                
        Returns
        -------
        predictions_npa : numpy array
                Numpy array containing predictions for each sample of the testing set.
        """

        # Sanity checks
        if self.lgb_model is None:
            raise ValueError("You MUST train the LightGBM model using fit() before attempting to do predictions!")

        print("Predicting outcome for testing set...")
        predictions_npa = self.lgb_model.predict(X, num_iteration = self.lgb_model.best_iteration)

        return predictions_npa
#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains a class that defines an efficient scikit-learn like      #
# pipeline that caches already done processing to disk to accelerate running  #
# time for further executions.                                                #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-05-11                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import time
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from joblib import Memory
from inspect import signature
from sklearn.utils.metaestimators import if_delegate_has_method
from scipy.sparse import csr_matrix
import pickle
import shutil

__all__ = ["EfficientPipeline"
          ]
    
def _fit_transform_helper(transformer, X, y, name, step_index, to_pandas, additional_data, **fit_params):
    if hasattr(transformer, "fit_transform"):
        if "additional_data" in list(dict(signature(transformer.transform).parameters).keys()):
            X = transformer.fit_transform(X, y, additional_data, **fit_params)
        else:
            X = transformer.fit_transform(X, y, **fit_params)
    else:
        if "y" in list(dict(signature(transformer.transform).parameters).keys()) and "additional_data" in list(dict(signature(transformer.transform).parameters).keys()):
            X = transformer.fit(X, y, additional_data, **fit_params).transform(X, y, additional_data)
        elif "y" in list(dict(signature(transformer.transform).parameters).keys()): # If the transformer needs 'y' for training set
            X = transformer.fit(X, y, **fit_params).transform(X, y)
        else:
            X = transformer.fit(X, y, **fit_params).transform(X)
            
    # If we want to keep Pandas format
    if to_pandas and not isinstance(X, pd.DataFrame) and not isinstance(X, pd.SparseDataFrame):
        if isinstance(X, np.ndarray): # If X is a regular numpy array, convert it to standard DataFrame
            X = pd.DataFrame(X, index = step_index, columns = [name + "_col_" + str(i) for i in range(X.shape[1])])
        elif isinstance(X, csr_matrix): # If X is a sparse matrix, convert it to a SparseDataFrame
            X = pd.SparseDataFrame(X, index = step_index, columns = [name + "_col_" + str(i) for i in range(X.shape[1])])

    return X, transformer

class EfficientPipeline(Pipeline):
    """
    The purpose of this class is to provide an efficient scikit-learn like pipeline 
    that caches already done processing to disk to accelerate running time for further 
    executions. This class is based on the original Scikit-Learn Pipeline:
    http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    """

    def __init__(self, steps, cache_dir = None, to_pandas = True):
        """
        This is the class' constructor.

        Parameters
        ----------
        steps : list
                List of (name, transform) tuples that are chained, in the order in which they 
                are chained, with the last object an estimator.

        cache_dir : string (default = None)
                This is the path of the directory where cached transformers will be stored. If
                set to None, no caching is done.

        to_pandas : bool (default = True)
                This flag indicates whether or not the output of each transformer must be a Pandas
                DataFrame.
                                                
        Returns
        -------
        None
        """

        # Class' attributes
        self.steps = steps
        self.cache_dir = cache_dir
        self.to_pandas = to_pandas

        # Create cache
        self._memory = Memory(cachedir = self.cache_dir, verbose = 0)

        # Create history of steps
        self._steps_history_fit = []
        self._steps_history_fit_transform = []
        self._steps_history_transform = []

    def _fit(self, X, y = None, additional_data = None, **fit_params):
        """
        This method is a helper for fit method.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (default = None)
                This is the target associated with the X data.

        Returns
        -------
        Xt : pd.DataFrame
                This is a data frame containing the data transformed by transformers.

        final_estimator_params : dict
                Params needd for final estimator.
        """

        # Check that every step is a transformer
        self._validate_steps()

        # Get params for each transformer
        fit_params_steps = dict((name, {}) for name, step in self.steps if step is not None)

        for pname, pval in fit_params.items():
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval
            
        # Enable caching
        fit_transform_helper_cached = self._memory.cache(_fit_transform_helper)

        # Create steps history
        if self.cache_dir is not None:
            for name, step in self.steps:
                if hasattr(self.named_steps[name], "fit"):
                    self._steps_history_fit.append((name, self.named_steps[name].fit.__func__.__code__.co_code))
                else:
                    self._steps_history_fit.append((name, None))

                if hasattr(self.named_steps[name], "fit_transform"):
                    self._steps_history_fit_transform.append((name, self.named_steps[name].fit_transform.__func__.__code__.co_code))
                else:
                    self._steps_history_fit_transform.append((name, None))

                if hasattr(self.named_steps[name], "transform"):
                    self._steps_history_transform.append((name, self.named_steps[name].transform.__func__.__code__.co_code))
                else:
                    self._steps_history_transform.append((name, None))

            # If a steps history already exists, load it
            if os.path.exists(self.cache_dir + "steps_history_fit.pkl") and os.path.exists(self.cache_dir + "steps_history_fit_transform.pkl") and os.path.exists(self.cache_dir + "steps_history_transform.pkl"):
                with open(self.cache_dir + "steps_history_fit.pkl", "rb") as f:
                    last_steps_history_fit = pickle.load(f)

                with open(self.cache_dir + "steps_history_fit_transform.pkl", "rb") as f:
                    last_steps_history_fit_transform = pickle.load(f)

                with open(self.cache_dir + "steps_history_transform.pkl", "rb") as f:
                    last_steps_history_transform = pickle.load(f)

                valid_fit_steps_lst = []
                for (current_name, current_bytecode), (last_name, last_bytecode) in zip(self._steps_history_fit, last_steps_history_fit):
                    if current_name != last_name or current_bytecode != last_bytecode:
                        break
                    else:
                        valid_fit_steps_lst.append(last_name)

                valid_fit_transform_steps_lst = []
                for (current_name, current_bytecode), (last_name, last_bytecode) in zip(self._steps_history_fit_transform, last_steps_history_fit_transform):
                    if current_name != last_name or current_bytecode != last_bytecode:
                        break
                    else:
                        valid_fit_transform_steps_lst.append(last_name)

                valid_transform_steps_lst = []
                for (current_name, current_bytecode), (last_name, last_bytecode) in zip(self._steps_history_transform, last_steps_history_transform):
                    if current_name != last_name or current_bytecode != last_bytecode:
                        break
                    else:
                        valid_transform_steps_lst.append(last_name)

                self._valid_steps_lst = []
                for valid_fit_step, valid_fit_transform_step, valid_transform_step in zip(valid_fit_steps_lst, valid_fit_transform_steps_lst, valid_transform_steps_lst):
                    if valid_fit_step == valid_fit_transform_step and valid_fit_step == valid_transform_step:
                        self._valid_steps_lst.append(valid_fit_step)
            else:
                self._valid_steps_lst = []

            # Remove invalid steps
            if os.path.exists(self.cache_dir + "cached_steps_paths_dict.pkl"):
                with open(self.cache_dir + "cached_steps_paths_dict.pkl", "rb") as f:
                    cached_steps_paths_dict = pickle.load(f)

                invalid_steps_lst = list(set(cached_steps_paths_dict.keys()) - set(self._valid_steps_lst))
                for invalid_step in invalid_steps_lst:
                    shutil.rmtree(cached_steps_paths_dict[invalid_step])

        # Fit transformers and apply their transformations
        Xt = X
        cached_steps_paths_dict = {}
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if transformer is None:
                pass
            else:
                if self.to_pandas:
                    step_index = Xt.index
                else:
                    step_index = None

                if hasattr(self._memory, "cachedir") and self._memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)

                # Fit or load from cache the current transfomer
                if self.cache_dir is not None:
                    cached_steps_paths_dict[name] = fit_transform_helper_cached.get_output_dir(cloned_transformer, Xt, y, name, step_index, self.to_pandas, additional_data, **fit_params_steps[name])[0]

                Xt, fitted_transformer = fit_transform_helper_cached(cloned_transformer, Xt, y, name, step_index, self.to_pandas, additional_data, **fit_params_steps[name])
                
                # Replace the transformer of the step with the fitted transformer. This is necessary when loading the transformer from the cache.
                self.steps[step_idx] = (name, fitted_transformer)

        # Save steps history to disk
        if self.cache_dir is not None:
            with open(self.cache_dir + "steps_history_fit.pkl", "wb") as f:
                pickle.dump(self._steps_history_fit, f)

            with open(self.cache_dir + "steps_history_fit_transform.pkl", "wb") as f:
                pickle.dump(self._steps_history_fit_transform, f)

            with open(self.cache_dir + "steps_history_transform.pkl", "wb") as f:
                pickle.dump(self._steps_history_transform, f)

            with open(self.cache_dir + "cached_steps_paths_dict.pkl", "wb") as f:
                pickle.dump(cached_steps_paths_dict, f)

        if self._final_estimator is None:
            return Xt, {}
        
        return Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y = None, additional_data = None, **fit_params):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (default = None)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        Xt, fit_params_steps = self._fit(X, y, additional_data, **fit_params)
        
        # Fit the final estimator
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, y, **fit_params_steps)
        
        return self

    @if_delegate_has_method(delegate = "_final_estimator")
    def predict(self, X, additional_data = None):
        """
        Apply transforms to the data, and predict with the final estimator.

        Parameters
        ----------
        X : pd.DataFrame
                Data to predict on. Must fulfill input requirements of first step of the pipeline.
                
        Returns
        -------
        y_pred : numpy array
                Series containing the transformed data.
        """

        Xt = X

        for name, transform in self.steps[:-1]:
            if transform is not None:
                if self.to_pandas:
                    step_index = Xt.index

                if "additional_data" in list(dict(signature(transformer.transform).parameters).keys()):
                    Xt = transform.transform(Xt, additional_data)
                else:
                    Xt = transform.transform(Xt)

                # If we want to keep Pandas format
                if self.to_pandas and not isinstance(Xt, pd.DataFrame) and not isinstance(Xt, pd.SparseDataFrame):
                    if isinstance(Xt, np.ndarray): # If Xt is a regular numpy array, convert it to standard DataFrame
                        Xt = pd.DataFrame(Xt, index = step_index, columns = [name + "_col_" + str(i) for i in range(Xt.shape[1])])
                    elif isinstance(Xt, csr_matrix): # If Xt is a sparse matrix, convert it to a SparseDataFrame
                        Xt = pd.SparseDataFrame(Xt, index = step_index, columns = [name + "_col_" + str(i) for i in range(Xt.shape[1])])

        return self.steps[-1][-1].predict(Xt)

    def fit_transform(self, X, y, additional_data = None, **fit_params):
        """
        Fit the model and transform with the final estimator.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (default = None)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        last_step = self._final_estimator
        Xt, fit_params = self._fit(X, y, additional_data, **fit_params)

        if hasattr(last_step, "fit_transform"):
            return last_step.fit_transform(Xt, y, **fit_params)
        elif last_step is None:
            return Xt
        else:
            if "y" in list(dict(signature(last_step.transform).parameters).keys()) and "additional_data" in list(dict(signature(last_step.transform).parameters).keys()):
                return last_step.fit(X, y, additional_data, **fit_params).transform(Xt, y, additional_data)
            elif "y" in list(dict(signature(last_step.transform).parameters).keys()): # If the transformer needs 'y' for training set
                return last_step.fit(X, y, **fit_params).transform(Xt, y)
            else:
                return last_step.fit(X, y, **fit_params).transform(Xt)

    @if_delegate_has_method(delegate = "_final_estimator")
    def fit_predict(self, X, y, additional_data = None, **fit_params):
        """
        Applies fit_predict of last step in pipeline after transforms.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (default = None)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        Xt, fit_params = self._fit(X, y, additional_data, **fit_params)

        return self.steps[-1][-1].fit_predict(Xt, y, **fit_params)

    @if_delegate_has_method(delegate = "_final_estimator")
    def predict_proba(self, X, additional_data = None):
        """
        Apply transforms, and predict_proba of the final estimator.

        Parameters
        ----------
        X : pd.DataFrame
                Data to predict on. Must fulfill input requirements of first step of the pipeline.
                
        Returns
        -------
        y_pred : numpy array
                Series containing the transformed data.
        """

        Xt = X

        for name, transform in self.steps[:-1]:
            if transform is not None:
                if self.to_pandas:
                    step_index = Xt.index

                if "additional_data" in list(dict(signature(transform.transform).parameters).keys()):
                    Xt = transform.transform(Xt, additional_data)
                else:
                    Xt = transform.transform(Xt)

                # If we want to keep Pandas format
                if self.to_pandas and not isinstance(Xt, pd.DataFrame) and not isinstance(Xt, pd.SparseDataFrame):
                    if isinstance(Xt, np.ndarray): # If Xt is a regular numpy array, convert it to standard DataFrame
                        Xt = pd.DataFrame(Xt, index = step_index, columns = [name + "_col_" + str(i) for i in range(Xt.shape[1])])
                    elif isinstance(Xt, csr_matrix): # If Xt is a sparse matrix, convert it to a SparseDataFrame
                        Xt = pd.SparseDataFrame(Xt, index = step_index, columns = [name + "_col_" + str(i) for i in range(Xt.shape[1])])

        return self.steps[-1][-1].predict_proba(Xt)

    @if_delegate_has_method(delegate = "_final_estimator")
    def predict_log_proba(self, X, additional_data = None):
        """
        Apply transforms, and predict_log_proba of the final estimator.

        Parameters
        ----------
        X : pd.DataFrame
                Data to predict on. Must fulfill input requirements of first step of the pipeline.
                
        Returns
        -------
        y_pred : numpy array
                Series containing the transformed data.
        """

        Xt = X

        for name, transform in self.steps[:-1]:
            if transform is not None:
                if self.to_pandas:
                    step_index = Xt.index

                if "additional_data" in list(dict(signature(transformer.transform).parameters).keys()):
                    Xt = transform.transform(Xt, additional_data)
                else:
                    Xt = transform.transform(Xt)

                # If we want to keep Pandas format
                if self.to_pandas and not isinstance(Xt, pd.DataFrame) and not isinstance(Xt, pd.SparseDataFrame):
                    if isinstance(Xt, np.ndarray): # If Xt is a regular numpy array, convert it to standard DataFrame
                        Xt = pd.DataFrame(Xt, index = step_index, columns = [name + "_col_" + str(i) for i in range(Xt.shape[1])])
                    elif isinstance(Xt, csr_matrix): # If Xt is a sparse matrix, convert it to a SparseDataFrame
                        Xt = pd.SparseDataFrame(Xt, index = step_index, columns = [name + "_col_" + str(i) for i in range(Xt.shape[1])])

        return self.steps[-1][-1].predict_log_proba(Xt)

    def transform(self, X, y = None, additional_data = None):
        """
        Apply transforms, and transform with the final estimator

        Parameters
        ----------
        X : pd.DataFrame
                Data to predict on. Must fulfill input requirements of first step of the pipeline.

        y : pd.Series (default = None)
                This is the target associated with the X data.

        additional_data : object
                Object containing additional data that must be used during transform
                
        Returns
        -------
        Xt : pd.DataFrame
                Transformed data.
        """

        Xt = X

        for name, transform in self.steps:
            if transform is not None:
                if self.to_pandas:
                    step_index = Xt.index

                if "y" in list(dict(signature(transform.transform).parameters).keys()) and "additional_data" in list(dict(signature(transform.transform).parameters).keys()):
                    Xt = transform.transform(Xt, y, additional_data)
                elif "y" in list(dict(signature(transform.transform).parameters).keys()):
                    Xt = transform.transform(Xt, y)
                else:
                    Xt = transform.transform(Xt)
                    
                # If we want to keep Pandas format
                if self.to_pandas and not isinstance(Xt, pd.DataFrame) and not isinstance(Xt, pd.SparseDataFrame):
                    if isinstance(Xt, np.ndarray): # If Xt is a regular numpy array, convert it to standard DataFrame
                        Xt = pd.DataFrame(Xt, index = step_index, columns = [name + "_col_" + str(i) for i in range(Xt.shape[1])])
                    elif isinstance(Xt, csr_matrix): # If Xt is a sparse matrix, convert it to a SparseDataFrame
                        Xt = pd.SparseDataFrame(Xt, index = step_index, columns = [name + "_col_" + str(i) for i in range(Xt.shape[1])])

        return Xt
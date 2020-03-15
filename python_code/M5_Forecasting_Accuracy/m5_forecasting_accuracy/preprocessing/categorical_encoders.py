#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains some classes that performs categorical encoding that are #
# compatible with scikit-learn API.                                           #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-04-07                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import time
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["OrdinalEncoder",
           "GroupingEncoder",
           "TargetAvgEncoder",
           "LeaveOneOutEncoder",
           "CategoricalFeaturesEncoder"
          ]

class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to provide a wrapper for the Scikit-Learn LabelEncoder class, that can manage
    cases where testing set contains new labels that doesn't exist in training set.
    """

    def __init__(self, mapping_dict = None, missing_value_replacement = "NA"):
        """
        This is the class' constructor.

        Parameters
        ----------
        mapping_dict : dictionary
                Use this to provide a custom mapping between current feature levels
                and the integers you want to associate with them.

        missing_value_replacement : string
                Value used to replace missing values.
                                
        Returns
        -------
        None
        """

        # Class' attributes
        self.mapping_dict = mapping_dict
        self.missing_value_replacement = missing_value_replacement

        self._label_encoder = LabelEncoder()
        self._unique_labels_lst = []

    def fit(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (default = None)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        if self.mapping_dict is None:
            # Fit the LabelEncoder
            self._label_encoder.fit(X.fillna(self.missing_value_replacement))

            # Save the unique labels available in the training set
            self._unique_labels_lst = list(self._label_encoder.classes_)
        else:
            # Save the unique labels available in the mapping dict
            self._unique_labels_lst = list(self.mapping_dict.keys())

        return self

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a series containing the data that will be transformed.
                
        Returns
        -------
        X_copy : pd.Series
                This is a series containing the data that will be transformed.
        """
        
        # Get new labels (labels not seen in fit)
        column_labels_lst = X.unique()
        unseen_labels_lst = list(set(column_labels_lst) - set(self._unique_labels_lst))

        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)
        X_copy.fillna(self.missing_value_replacement, inplace = True)

        # Transform the feature
        if self.mapping_dict is None:
            X_copy.loc[~X_copy.isin(unseen_labels_lst)] = self._label_encoder.transform(X_copy.loc[~X_copy.isin(unseen_labels_lst)])
        else:
            X_copy.loc[~X_copy.isin(unseen_labels_lst)] = X_copy.loc[~X_copy.isin(unseen_labels_lst)].map(self.mapping_dict)

        X_copy.loc[X_copy.isin(unseen_labels_lst)] = np.nan
        
        # If there is no NaNs in the series, cast it to np.int32
        if X_copy.isnull().sum() == 0:
            X_copy = X_copy.astype(np.int32)
            
        return X_copy

class GroupingEncoder(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to provide a categorical feature encoder especially designed to
    handle features with high cardinality by grouping scarcest levels into a "OTHER" label. Then,
    it performs one-hot encoding on remaining levels.
    """

    def __init__(self, encoder, threshold, grouping_name = "OTHER"):
        """
        This is the class' constructor.

        Parameters
        ----------
        encoder : scikit-learn transformer
                This is the encoder that will be used to encode the feature after
                grouping its least frequent levels.

        threshold : either integer >= 2 or float between 0 and 1
                - If this is a float between 0 and 1, then only the scarcest levels that cumulated sum represents
                  1 - threshold will be grouped in a class named 'grouping_name'.
                - If this is an integer, then only the 'threshold' most represented levels will be kept. Others will be grouped
                  in a level named 'grouping_name'.

        grouping_name : string (default = "OTHER")
                Name given to the levels grouped for the feature.
                                                
        Returns
        -------
        None
        """

        # Class' attributes
        self.encoder = encoder
        self.threshold = threshold
        self.grouping_name = grouping_name

        self._kept_levels = []
        self.classes_ = None

    def fit(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (default = None)
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        # Get the names of the levels that will be grouped together
        levels_count_sr = X.value_counts()

        if type(self.threshold) == int and self.threshold >= 2:
            self._kept_levels = levels_count_sr.head(self.threshold).index.tolist()
        elif type(self.threshold) == float and self.threshold > 0 and self.threshold < 1:
            levels_count_df = levels_count_sr.reset_index()
            levels_count_df.columns = ["level", "count"]
            levels_count_df["cumsum"] = levels_count_df["count"].cumsum()
            levels_count_df = levels_count_df.loc[levels_count_df["cumsum"] < self.threshold * levels_count_df["count"].sum()]
            self._kept_levels = levels_count_df["level"].tolist()

        # Fit the encoder
        tmp_sr = X.copy(deep = True)
        tmp_sr.loc[~X.isin(self._kept_levels)] = self.grouping_name

        if isinstance(self.encoder, LeaveOneOutEncoder) or isinstance(self.encoder, TargetAvgEncoder):
            self.encoder.fit(tmp_sr, y)
        else:
            self.encoder.fit(tmp_sr)
        
        self.classes_ = self.encoder.classes_

        return self

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a series containing the data that will be transformed.
                
        Returns
        -------
         : numpy array
                Numpy array containing the transformed data.
        """
        
        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)

        # Group labels that need to be grouped
        X_copy.loc[~X_copy.isin(self._kept_levels)] = self.grouping_name

        # Apply the transformer
        return self.encoder.transform(X_copy)
    
class TargetAvgEncoder(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to provide a transformer that encodes categorical variable by
    replacing each level by its target mean.

    Beware: Only use this transformer with low cardinality features. Otherwise, high overfiting 
    can be introduced in the model. In this case, the LeaveOneOut encoder is preferred.
    """

    def __init__(self):
        """
        This is the class' constructor.

        Parameters
        ----------
        None
                                                
        Returns
        -------
        None
        """

        # Class' attributes
        self.classes_ = None

    def fit(self, X, y):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series
                This is the target associated with the X data.

        Returns
        -------
        None
        """

        # Compute the average target value for each feature level
        self._means_dict = y.groupby(X).mean().to_dict()

        # Compute the global mean of the feature
        self._target_mean = y.mean()

        return self

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a series containing the data that will be transformed.
                
        Returns
        -------
        X_copy : pd.Series
                Series containing the transformed data.
        """
        
        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)

        # Map mean values to feature levels
        X_copy = X_copy.map(self._means_dict)

        # Replace new levels by whole target mean
        X_copy.fillna(self._target_mean, inplace = True)

        return X_copy

class LeaveOneOutEncoder(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to provide a leave-one-out coding for categorical features.

    References
    ----------

    .. [1] Strategies to encode categorical variables with many categories. From
    https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/15748#143154
    """

    def __init__(self, add_gaussian_noise = True, sigma = 0.05):
        """
        This is the class' constructor.

        Parameters
        ----------
        add_gaussian_noise : bool (default = True)
                Add Gaussian noise or not to the data encoded in fit_transform() method. The purpose of this
                is to reduce overfitting.
        
        sigma : float (default = 0.05)
                Standard deviation of the above mentionned Gaussian distribution.
                                                
        Returns
        -------
        None
        """

        # Class' attributes
        self.add_gaussian_noise = add_gaussian_noise
        self.sigma = sigma

        self.classes_ = None
        
    def fit(self, X, y):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series
                This is the target associated with the X data.

        Returns
        -------
        self: SparseTextEncoder object
                Return current object.
        """

        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)

        # Compute the global target mean
        self._target_mean = y.mean()

        # Compute the average target value for each feature level
        self._feature_statistics_df = y.groupby(X_copy).agg(["sum", "count"])
        self._feature_statistics_df["mean"] = self._feature_statistics_df["sum"] / self._feature_statistics_df["count"]
        self._feature_statistics_df.columns = [X_copy.name + "_sum", X_copy.name + "_count", X_copy.name + "_mean"]
        self._feature_statistics_df = self._feature_statistics_df.reset_index()

        # Get all levels that only appear once
        self._single_levels_npa = self._feature_statistics_df.loc[self._feature_statistics_df[X_copy.name + "_count"] == 1].index.values

        return self

    def transform(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                This is a series containing the data that will be transformed.
                
        Returns
        -------
        X_copy : pd.Series
                Data transformed by the transformer.
        """

        # Copy the object to avoid any modification
        X_copy = X.copy(deep = True)

        # For all levels that appear only once, replace them by the average target value
        X_copy.loc[X_copy.isin(self._single_levels_npa)] = self._target_mean

        # Encode the remaining levels
        X_copy_df = pd.DataFrame(X_copy).merge(self._feature_statistics_df, how = "left", on = X_copy.name)
        X_copy_df.index = X.index

        if y is not None:
            X_copy.loc[~X_copy.isin(self._single_levels_npa)] = (X_copy_df[X_copy.name + "_sum"].loc[~X_copy.isin(self._single_levels_npa)] - y.loc[~X_copy.isin(self._single_levels_npa)]) / (X_copy_df[X_copy.name + "_count"].loc[~X_copy.isin(self._single_levels_npa)] - 1)
        
            if self.add_gaussian_noise:
                X_copy = X_copy * np.random.normal(1., self.sigma, X_copy.shape[0])
        else:
            # Encode the remaining levels
            X_copy.loc[~X_copy.isin(self._single_levels_npa)] = X_copy_df[X_copy.name + "_mean"].loc[~X_copy.isin(self._single_levels_npa)]
                
        return X_copy

    def fit_transform(self, X, y = None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.

        y : pd.Series (optional)
                This is the target associated with the X data. Only use this for training data.
                
        Returns
        -------
        X: pd.DataFrame
                Transformed data.
        """

        if y is None:
            return self.fit(X).transform(X)
        else:
            return self.fit(X, y).transform(X, y)

class CategoricalFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
    The purpose of this class is to provide a transformer that encodes categorical features into numerical ones.
    """
    
    def __init__(self, columns_names_lst, encoders_lst, missing_value_replacement = "NA", drop_initial_features = True):
        """
        Class' constructor

        Parameters
        ----------
        columns_names_lst : list
                Names of the columns we want to transform.

        encoders_lst : list
                Encoders chosen for each column of the columns_names_lst list.

        missing_value_replacement : string
                Value used to replace missing values.

        drop_initial_features : bool
                Flag indicating whether to drop or not initial features used for encoding.
                
        Returns
        -------
        None
        """
        
        if len(columns_names_lst) != len(encoders_lst):
            raise ValueError("Number of items in 'columns_names_lst' doesn't match number of items in 'encoders_lst'!")

        self.columns_names_lst = columns_names_lst
        self.encoders_lst = encoders_lst
        self.missing_value_replacement = missing_value_replacement
        self.drop_initial_features = drop_initial_features
        
    def fit(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        self: CategoricalFeaturesEncoder object
                Return current object.
        """

        for idx in range(len(self.columns_names_lst)):
            # Fit each encoder
            if isinstance(self.encoders_lst[idx], LeaveOneOutEncoder) or isinstance(self.encoders_lst[idx], TargetAvgEncoder) or isinstance(self.encoders_lst[idx], GroupingEncoder):
                self.encoders_lst[idx].fit(X[self.columns_names_lst[idx]].fillna(self.missing_value_replacement), y)
            else:
                self.encoders_lst[idx].fit(X[self.columns_names_lst[idx]].fillna(self.missing_value_replacement))
            
        return self

    def transform(self, X, y = None):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.

        y : pd.Series (optional)
                This is the target associated with the X data. Only use this for training data.
                
        Returns
        -------
        X: pd.DataFrame
                Transformed data.
        """

        start_time = time.time()
        
        for idx in range(len(self.columns_names_lst)):
            # Impute missing values
            X[self.columns_names_lst[idx]].fillna(self.missing_value_replacement, inplace = True)

            # Transform data using each encoder
            if isinstance(self.encoders_lst[idx], LabelBinarizer) or isinstance(self.encoders_lst[idx], GroupingEncoder): # We need a different handling of the preprocessing for one-hot coding based encoders, as they add new columns
                # Get the result of feature transformation and convert it to Pandas DataFrame
                tmp_npa = self.encoders_lst[idx].transform(X[self.columns_names_lst[idx]])

                if len(tmp_npa.shape) == 2 and tmp_npa.shape[1] > 1:
                    columns_lst = [self.columns_names_lst[idx] + "_" + level for level in self.encoders_lst[idx].classes_]

                    # Add the DataFrame to the existing data
                    X = pd.concat([X, pd.DataFrame(tmp_npa, index = X.index, columns = columns_lst)], axis = 1)
                else:
                    X[self.columns_names_lst[idx] + "_" + str(self.encoders_lst[idx]).split("(")[0]] = tmp_npa
                                    
            elif isinstance(self.encoders_lst[idx], LeaveOneOutEncoder):
                X[self.columns_names_lst[idx] + "_" + str(self.encoders_lst[idx]).split("(")[0]] = self.encoders_lst[idx].transform(X[self.columns_names_lst[idx]], y)
            else:
                X[self.columns_names_lst[idx] + "_" + str(self.encoders_lst[idx]).split("(")[0]] = self.encoders_lst[idx].transform(X[self.columns_names_lst[idx]])

        # Drop the initial features
        if self.drop_initial_features:
            X.drop(list(set(self.columns_names_lst)), axis = 1, inplace = True)

        print("CategoricalFeaturesEncoder transformed", len(self.columns_names_lst), "categorical features in", round(time.time() - start_time, 3), "seconds.")

        return X

    def fit_transform(self, X, y = None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.

        y : pd.Series (optional)
                This is the target associated with the X data. Only use this for training data.
                
        Returns
        -------
        X: pd.DataFrame
                Transformed data.
        """

        if y is None:
            return self.fit(X).transform(X)
        else:
            return self.fit(X, y).transform(X, y)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains the code needed for creating a transformer that encodes  #
# text using bag of words, TF-IDF or LSA representation. It is compatible     #
# with the Scikit-Learn framework and uses sparse matrices for reducing memory#
# consumption.                                                                #
#                                                                             #
# Credits for WordBatch: https://github.com/anttttti/Wordbatch                #
#                                                                             #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-04-22                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelBinarizer
from wordbatch import WordBatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix, coo_matrix, hstack
import multiprocessing as mp

__all__ = ["SparseTextEncoder",
           "LSAVectorizer"
          ]

class SparseTextEncoder(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that implements a text encoder using bag of words or TF-IDF representation.
    """

    def __init__(self, columns_names_lst, encoders_lst, nnz_threshold = 2, output_format = "csr"):
        """
        This is the class' constructor.

        Parameters
        ----------
        columns_names_lst : list
                Names of the columns we want to transform.

        encoders_lst : list
                Encoders chosen for each column of the columns_names_lst list.

        nnz_threshold: positive integer (default = 2)
                Minimum number of non-zero values we want to have in generated features. If, for a given generated feature, 
                the number of non-zero values is less than this threshold, then the feature is dropped.
                
        output_format: string (default = "csr")
                Output format of this transformer. This can be either "csr" or "pandas". In the first case, the data is 
                returned in a Scipy CSR matrix. In the latter, the data is returned in a Pandas SparseDataFrame.
                The Pandas format keeps columns names, but the conversion to this format takes some time.
                                
        Returns
        -------
        None
        """

        self.columns_names_lst = columns_names_lst
        self.encoders_lst = encoders_lst
        self.nnz_threshold = nnz_threshold
        self.output_format = output_format

        self._encoders_masks_lst = [None for i in encoders_lst] # List of masks that allows to remove columns without enough non-zero values.

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
        self: SparseTextEncoder object
                Return current object.
        """

        raise NotImplementedError("Method not implemented! Please call fit_transform() instead.")

        return self

    def fit_transform(self, X, y = None):
        """
        This method is called to fit the transformer on the training data and then transform the associated data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data used to fit the transformer.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        encoded_features_sdf: Pandas SparseDataFrame OR sparse_merge: Scipy CSR matrix
                Transformed data.
        """

        start_time = time.time()

        standard_columns_lst = list(set(X.columns.tolist()) - set(self.columns_names_lst)) # Standard columns are columns we don't want to transform
        arrays_lst = [csr_matrix(X[standard_columns_lst].values)]
        columns_names_lst = [c for c in standard_columns_lst]
        
        for idx, column in enumerate(self.columns_names_lst):
            X[column].fillna("NaN", inplace = True)

            if type(self.encoders_lst[idx]) == WordBatch:
                self.encoders_lst[idx].dictionary_freeze = True # Freeze dictionary to avoid adding new words when calling transform() method.

            # Encode the feature
            encoded_features_csr = self.encoders_lst[idx].fit_transform(X[column]) 
            
            # Compute mask to remove generated features that don't have enough non-zero values
            if type(self.encoders_lst[idx]) != LSAVectorizer:
                self._encoders_masks_lst[idx] = np.array(np.clip(encoded_features_csr.getnnz(axis = 0) - self.nnz_threshold, 0, 1), dtype = bool)
            
                # Actually remove generated features that don't have enough non-zero values
                encoded_features_csr = encoded_features_csr[:, self._encoders_masks_lst[idx]]
            else:
                encoded_features_csr = coo_matrix(encoded_features_csr)
            
            # Generate the features name
            if type(self.encoders_lst[idx]) == CountVectorizer or type(self.encoders_lst[idx]) == TfidfVectorizer:
                encoded_columns_names_lst = [column + "_" + w for w in self.encoders_lst[idx].get_feature_names()]
            elif type(self.encoders_lst[idx]) == LabelBinarizer:
                encoded_columns_names_lst = [column + "_LabelBinarizer_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]
            elif type(self.encoders_lst[idx]) == WordBatch:
                encoded_columns_names_lst = [column + "_WordBatch_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]
            elif type(self.encoders_lst[idx]) == LSAVectorizer:
                encoded_columns_names_lst = [column + "_LSA_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]

            # If the number of columns names is greater than the number of columns, drop useless column names
            if len(encoded_columns_names_lst) > encoded_features_csr.shape[1]:
                encoded_columns_names_lst = np.array(encoded_columns_names_lst)[self._encoders_masks_lst[idx]].tolist()
                
            arrays_lst.append(encoded_features_csr)
            columns_names_lst.extend(encoded_columns_names_lst)

        sparse_merge = hstack(arrays_lst).tocsr()
        
        if self.output_format == "pandas":
            encoded_features_sdf = pd.SparseDataFrame(sparse_merge, default_fill_value = 0, columns = columns_names_lst, index = X.index)
            print("SparseTextEncoder transformed", len(self.columns_names_lst), "text features into", encoded_features_sdf.shape[1], "new features in", round(time.time() - start_time, 3), "seconds.")
            return encoded_features_sdf
        else:
            print("SparseTextEncoder transformed", len(self.columns_names_lst), "text features into", sparse_merge.shape[1], "new features in", round(time.time() - start_time, 3), "seconds.")
            return sparse_merge

    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
                This is a data frame containing the data that will be transformed.
                
        Returns
        -------
        encoded_features_sdf: Pandas SparseDataFrame OR sparse_merge: Scipy CSR matrix
                Transformed data.
        """

        start_time = time.time()

        standard_columns_lst = list(set(X.columns.tolist()) - set(self.columns_names_lst))
        arrays_lst = [csr_matrix(X[standard_columns_lst].values)]
        columns_names_lst = [c for c in standard_columns_lst]
        
        for idx, column in enumerate(self.columns_names_lst):
            X[column].fillna("NaN", inplace = True)

            # Encode the feature
            encoded_features_csr = self.encoders_lst[idx].transform(X[column])

            if type(self.encoders_lst[idx]) != LSAVectorizer:
                # Remove generated features that don't have enough non-zero values
                encoded_features_csr = encoded_features_csr[:, self._encoders_masks_lst[idx]]
            else:
                encoded_features_csr = coo_matrix(encoded_features_csr)

            # Generate the features name
            if type(self.encoders_lst[idx]) == CountVectorizer or type(self.encoders_lst[idx]) == TfidfVectorizer:
                encoded_columns_names_lst = [column + "_" + w for w in self.encoders_lst[idx].get_feature_names()]
            elif type(self.encoders_lst[idx]) == LabelBinarizer:
                encoded_columns_names_lst = [column + "_LabelBinarizer_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]
            elif type(self.encoders_lst[idx]) == WordBatch:
                encoded_columns_names_lst = [column + "_WordBatch_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]
            elif type(self.encoders_lst[idx]) == LSAVectorizer:
                encoded_columns_names_lst = [column + "_LSA_" + str(w + 1) for w in range(encoded_features_csr.shape[1])]

            # If the number of columns names is greater than the number of columns, drop useless column names
            if len(encoded_columns_names_lst) > encoded_features_csr.shape[1]:
                encoded_columns_names_lst = np.array(encoded_columns_names_lst)[self._encoders_masks_lst[idx]].tolist()

            arrays_lst.append(encoded_features_csr)
            columns_names_lst.extend(encoded_columns_names_lst)
            
        sparse_merge = hstack(arrays_lst).tocsr()
        
        if self.output_format == "pandas":
            encoded_features_sdf = pd.SparseDataFrame(sparse_merge, default_fill_value = 0, columns = columns_names_lst, index = X.index)
            print("SparseTextEncoder transformed", len(self.columns_names_lst), "text features into", encoded_features_sdf.shape[1], "new features in", round(time.time() - start_time, 3), "seconds.")
            return encoded_features_sdf
        else:
            print("SparseTextEncoder transformed", len(self.columns_names_lst), "text features into", sparse_merge.shape[1], "new features in", round(time.time() - start_time, 3), "seconds.")
            return sparse_merge

class LSAVectorizer(BaseEstimator, TransformerMixin):
    """
    This class defines a Scikit-Learn transformer that implements a Latent Semantic Analysis.
    """

    def __init__(self, lsa_components, tfidf_parameters = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 10}):
        """
        This is the class' constructor.

        Parameters
        ----------
        lsa_components : positive integer
                Number of components we want to keep.

        tfidf_parameters : dict (default = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 10})
                Dict containing parameters of TfidfVectorizer used in this class.
                Each dictionary key must corresponds to one scikit-learn TfidfVectorizer parameter,
                as defined here: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

        Returns
        -------
        None
        """
        
        self.lsa_components = lsa_components
        self.tfidf_parameters = tfidf_parameters

        self._tfv = TfidfVectorizer(**self.tfidf_parameters)
        self._svd = TruncatedSVD(self.lsa_components)
        
    def fit(self, X, y = None):
        """
        Fit the transformer on provided text.

        Parameters
        ----------
        X : pd.Series
                Series containing the text to transform.

        y : pd.Series (optional)
                This is the target associated with the X data.

        Returns
        -------
        self: LSAVectorizer object
                Return current object.
        """

        # Remove missing values
        X.fillna("NA", inplace = True)

        # Fit the TfidfVectorizer
        tfidf_output_csr = self._tfv.fit_transform(X, y)

        # Fit the SVD
        self._svd.fit(tfidf_output_csr)

        print("LSA explained variance:", round(np.sum(self._svd.explained_variance_ratio_) * 100, 3), "%")

        return self
    
    def transform(self, X):
        """
        This method is called to fit the transformer on the training data.

        Parameters
        ----------
        X : pd.Series
                Data that will be transformed.
                
        Returns
        -------
        lsa_output_npa : numpy array
                Transformed data.
        """

        # Remove missing values
        X.fillna("NA", inplace = True)

        # Transform the data using TfidfVectorizer
        tfidf_output_csr = self._tfv.transform(X)

        # Reduce dimensionality using SVD
        lsa_output_npa = self._svd.transform(tfidf_output_csr)
                
        return lsa_output_npa

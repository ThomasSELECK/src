#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the PetFinder.my Adoption Prediction competition         #
#                                                                             #
# This file is used to tune a specific ML model.                              #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-03-15                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import os
import time
import numpy as np
import pandas as pd
import pickle
import gc
import seaborn as sns
import matplotlib.pyplot as plt

import json

import scipy as sp
import pandas as pd
import numpy as np

from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold

from collections import Counter

import lightgbm as lgb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import cohen_kappa_score
from ml_metrics import quadratic_weighted_kappa
from hyperopt import hp, tpe

from preprocessing.PreprocessingStep import PreprocessingStep
from preprocessing.categorical_encoders import CategoricalFeaturesEncoder, OrdinalEncoder, GroupingEncoder, LeaveOneOutEncoder, TargetAvgEncoder
from preprocessing.text_encoders import SparseTextEncoder, LSAVectorizer
from wrappers.lightgbm_wrapper import LGBMClassifier
from wrappers.blended_lightgbm_wrapper import BlendedLGBMClassifier
from pipeline.efficient_pipeline import EfficientPipeline
from preprocessing.OptimizedRounder import OptimizedRounder
from preprocessing.SentimentFilesPreprocessingStep import SentimentFilesPreprocessingStep
from preprocessing.MetadataFilesPreprocessingStep import MetadataFilesPreprocessingStep
from preprocessing.ImagesMetadataExtractionStep import ImagesMetadataExtractionStep
from preprocessing.ImagesFeaturesExtractionStep import ImagesFeaturesExtractionStep, GenerateMxNetRecordIOFile
from feature_engineering.features_selectors import DuplicatedFeaturesRemover
from wrappers.mlp_wrapper import MLPClassifier
from preprocessing.missing_values_imputation import MissingValuesImputer

from model_evaluation.auto_tuner import AutoTuner

from dev.files_paths import *
from dev.load_data import load_data

pd.set_option("display.max_columns", 100)

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
    
    st = time.time()

    enable_validation = True

    print("Loading data...")
    training_data_df, testing_data_df, target_sr, truth_sr = load_data(TRAIN_DATA_str, TEST_DATA_str, BREEDS_DATA_str, STATES_DATA_str, COLORS_DATA_str, CAT_AND_DOG_BREEDS_DATA_str, enable_validation, "AdoptionSpeed")

    # Extracting IDs
    train_id = training_data_df["PetID"]
    test_id = testing_data_df["PetID"]

    print("training_data_df.shape:", training_data_df.shape)
    print("testing_data_df.shape:", testing_data_df.shape)
    
    print("Data loaded in:", time.time() - st, "secs")
    
    # Extract data from sentiment files
    sfps = SentimentFilesPreprocessingStep(n_cores = -1)
    train_sentiment_desc_df, train_sentiments_df = sfps.extract_data_from_sentiment_files(train_id.tolist(), TRAIN_SENTIMENT_DATA_DIR_str)

    if enable_validation:
        test_sentiment_desc_df, test_sentiments_df = sfps.extract_data_from_sentiment_files(test_id.tolist(), TRAIN_SENTIMENT_DATA_DIR_str)
    else:
        test_sentiment_desc_df, test_sentiments_df = sfps.extract_data_from_sentiment_files(test_id.tolist(), TEST_SENTIMENT_DATA_DIR_str)

    # Extract data from metadata files
    mfps = MetadataFilesPreprocessingStep(n_cores = -1)
    train_metadata_desc_df, train_metadata_df = mfps.extract_data_from_metadata_files(train_id.tolist(), TRAIN_METADATA_DIR_str)

    if enable_validation:
        test_metadata_desc_df, test_metadata_df = mfps.extract_data_from_metadata_files(test_id.tolist(), TRAIN_METADATA_DIR_str)
    else:
        test_metadata_desc_df, test_metadata_df = mfps.extract_data_from_metadata_files(test_id.tolist(), TEST_METADATA_DIR_str)
    
    # Merging sentiment and metadata to main DataFrames
    training_data_df = training_data_df.merge(train_sentiments_df, how = "left", on = "PetID")
    training_data_df = training_data_df.merge(train_metadata_df, how = "left", on = "PetID")
    training_data_df = training_data_df.merge(train_metadata_desc_df, how = "left", on = "PetID")
    training_data_df = training_data_df.merge(train_sentiment_desc_df, how = "left", on = "PetID")

    testing_data_df = testing_data_df.merge(test_sentiments_df, how = "left", on = "PetID")
    testing_data_df = testing_data_df.merge(test_metadata_df, how = "left", on = "PetID")
    testing_data_df = testing_data_df.merge(test_metadata_desc_df, how = "left", on = "PetID")
    testing_data_df = testing_data_df.merge(test_sentiment_desc_df, how = "left", on = "PetID")

    # Extract features from images
    imes = ImagesMetadataExtractionStep(n_cores = -1)
    train_images_metadata_df = imes.extract_metadata_from_images(train_id.tolist(), TRAIN_IMAGES_DIR_str)

    if enable_validation:
        test_images_metadata_df = imes.extract_metadata_from_images(test_id.tolist(), TRAIN_IMAGES_DIR_str)
    else:
        test_images_metadata_df = imes.extract_metadata_from_images(test_id.tolist(), TEST_IMAGES_DIR_str)

    training_data_df = training_data_df.merge(train_images_metadata_df, how = "left", on = "PetID")
    testing_data_df = testing_data_df.merge(test_images_metadata_df, how = "left", on = "PetID")

    # Create a RecordIO file to load images faster
    gmrf = GenerateMxNetRecordIOFile(n_cores = 5)
    gmrf.generate_record_io_file(train_id.tolist(), TRAIN_RECORDIO_PREFIX_str, TRAIN_IMAGES_DIR_str)

    if enable_validation:
        gmrf.generate_record_io_file(test_id.tolist(), TEST_RECORDIO_PREFIX_str, TRAIN_IMAGES_DIR_str)
    else:
        gmrf.generate_record_io_file(test_id.tolist(), TEST_RECORDIO_PREFIX_str, TEST_IMAGES_DIR_str)

    # Extract features from images (inside the RecordIO)
    ifes = ImagesFeaturesExtractionStep(NN_WEIGHTS_str)
    train_images_features_df = ifes.extract_features_from_images(train_id.tolist(), TRAIN_RECORDIO_PREFIX_str)

    if enable_validation:
        test_images_features_df = ifes.extract_features_from_images(test_id.tolist(), TEST_RECORDIO_PREFIX_str)
    else:
        test_images_features_df = ifes.extract_features_from_images(test_id.tolist(), TEST_RECORDIO_PREFIX_str)
    
    del gmrf, ifes
    gc.collect()

    pca = PCA(64)
    train_images_features_df = pd.DataFrame(pca.fit_transform(train_images_features_df), index = train_images_features_df.index, columns = ["images_features_PCA_" + str(i) for i in range(64)])
    test_images_features_df = pd.DataFrame(pca.transform(test_images_features_df), index = test_images_features_df.index, columns = ["images_features_PCA_" + str(i) for i in range(64)])
    train_images_features_df = train_images_features_df.reset_index()
    test_images_features_df = test_images_features_df.reset_index()
    training_data_df = training_data_df.merge(train_images_features_df, how = "left", on = "PetID")
    testing_data_df = testing_data_df.merge(test_images_features_df, how = "left", on = "PetID")

    gc.collect()
            
    categorical_columns_to_be_encoded_lst = ["State", 
                                             "Breed1", "Breed2",
                                             "Color1", "Color2", "Color3"]#, "Description_language"]
    categorical_encoders_lst = [LabelBinarizer(), 
                                GroupingEncoder(encoder = TargetAvgEncoder(), threshold = 103), GroupingEncoder(encoder = TargetAvgEncoder(), threshold = 66),
                                LabelBinarizer(), LabelBinarizer(), LabelBinarizer()#, TargetAvgEncoder()
                                ]

    # Columns that will be encoded using SparseTextEncoder class
    lsa_parameters = {"min_df": 3,  "max_features": 10000, "strip_accents": "unicode", "analyzer": "word", "token_pattern": r"\w{1,}", "ngram_range": (1, 3), "use_idf": 1, "smooth_idf": 1, 
                      "sublinear_tf": 1, "stop_words": "english"}
    text_columns_to_be_encoded_lst = ["Description", "metadata_annots_top_desc", 
                                      "sentiment_file_entities_name_lst", "sentiment_file_entities_type_lst", "Colors"]
    text_encoders_lst = [LSAVectorizer(lsa_components = 140, tfidf_parameters = lsa_parameters), LSAVectorizer(lsa_components = 40, tfidf_parameters = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 10, "token_pattern": r"[a-zA-Z_]+"}), 
                         LSAVectorizer(lsa_components = 20), TfidfVectorizer(), CountVectorizer()]
    
    # Put EfficientPipeline instead of Pipeline
    preprocessing_pipeline = Pipeline([
                              ("PreprocessingStep", PreprocessingStep()),
                              ("CategoricalFeaturesEncoder", CategoricalFeaturesEncoder(categorical_columns_to_be_encoded_lst, categorical_encoders_lst)),
                              ("SparseTextEncoder", SparseTextEncoder(text_columns_to_be_encoded_lst, text_encoders_lst, output_format = "pandas")),
                              ("DuplicatedFeaturesRemover", DuplicatedFeaturesRemover())
                             ])

    # For XGBoost
    hyperparameters_dict = {
            "max_depth": hp.quniform("max_depth", 5, 10, 2),
            "eta": hp.uniform("eta", 8e-3, 2e-2),
            "subsample": hp.uniform("subsample", 0.7, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.4, 0.9),
            "gamma": hp.uniform("gamma", 0.01, 0.1),
            "min_child_weight": hp.uniform("min_child_weight", 0.01, 0.1)
        }

    """
    # For LightGBM
    hyperparameters_dict = {
            "num_leaves": hp.quniform("num_leaves", 8, 128, 2),
            "learning_rate": hp.uniform("learning_rate", 1e-3, 1e-2),
            "bagging_fraction": hp.uniform("bagging_fraction", 0.7, 1.0),
            "feature_fraction": hp.uniform("feature_fraction", 0.1, 0.5),
            "min_split_gain": hp.uniform("min_split_gain", 0.01, 0.1),
            "min_child_samples": hp.quniform("min_child_samples", 90, 200, 2),
            "min_child_weight": hp.uniform("min_child_weight", 0.01, 0.1)
        }
    """

    """
    # For MLP
    hyperparameters_dict = {
        "layers": hp.choice("layers", [{
            "n_layers": 3,
            "n_units_layer": [
                hp.quniform("n_units_layer_31", 24, 512, q = 1),
                hp.quniform("n_units_layer_32", 24, 512, q = 1),
                hp.quniform("n_units_layer_33", 24, 512, q = 1)
            ],
        }, {
            "n_layers": 4,
            "n_units_layer": [
                hp.quniform("n_units_layer_41", 24, 512, q = 1),
                hp.quniform("n_units_layer_42", 24, 512, q = 1),
                hp.quniform("n_units_layer_43", 24, 512, q = 1),
                hp.quniform("n_units_layer_44", 24, 512, q = 1)
            ],
        }, {
            "n_layers": 5,
            "n_units_layer": [
                hp.quniform("n_units_layer_51", 24, 512, q = 1),
                hp.quniform("n_units_layer_52", 24, 512, q = 1),
                hp.quniform("n_units_layer_53", 24, 512, q = 1),
                hp.quniform("n_units_layer_54", 24, 512, q = 1),
                hp.quniform("n_units_layer_55", 24, 512, q = 1)
            ],
        }, {
            "n_layers": 6,
            "n_units_layer": [
                hp.quniform("n_units_layer_61", 24, 512, q = 1),
                hp.quniform("n_units_layer_62", 24, 512, q = 1),
                hp.quniform("n_units_layer_63", 24, 512, q = 1),
                hp.quniform("n_units_layer_64", 24, 512, q = 1),
                hp.quniform("n_units_layer_65", 24, 512, q = 1),
                hp.quniform("n_units_layer_66", 24, 512, q = 1)
            ],
        }, {
            "n_layers": 7,
            "n_units_layer": [
                hp.quniform("n_units_layer_71", 24, 512, q = 1),
                hp.quniform("n_units_layer_72", 24, 512, q = 1),
                hp.quniform("n_units_layer_73", 24, 512, q = 1),
                hp.quniform("n_units_layer_74", 24, 512, q = 1),
                hp.quniform("n_units_layer_75", 24, 512, q = 1),
                hp.quniform("n_units_layer_76", 24, 512, q = 1),
                hp.quniform("n_units_layer_77", 24, 512, q = 1)
            ],
        }]),
        "activation": hp.choice("activation", [
            "relu",
            "tanh"
        ]),
    
        "dropout": hp.uniform("dropout", 0, 0.5),
        "learning_rate": hp.uniform("learning_rate", 1e-4, 1e-2),
        "l2_regularization": hp.uniform("l2_regularization", 1e-5, 1e-4),
    }
    """

    at = AutoTuner()
    # 100 tries: 22,356 s => LightGBM
    results_df = at.tune_model(1, hyperparameters_dict, preprocessing_pipeline, BlendedLGBMClassifier, training_data_df, target_sr, testing_data_df, truth_sr)

    results_df.to_excel(TUNING_RESULTS_DIR_str + "xgb_tuning_results.xlsx", index = False)
    
    # Stop the timer and print the exectution time
    print("*** Test finished : Executed in:", time.time() - start_time, "seconds ***")
#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the M5 Forecasting Accuracy competition                  #
#                                                                             #
# This file contains all files paths of the datasets.                         #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2020-03-14                                                            #
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
import shutil
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
import lightgbm as lgb
from datetime import datetime
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

from dev.files_paths import *
from m5_forecasting_accuracy.data_loading.data_loader import DataLoader

from m5_forecasting_accuracy.models_evaluation.metric_dashboard import WRMSSEDashboard
from m5_forecasting_accuracy.models.lgbm_day_model import LGBMDayModel
from m5_forecasting_accuracy.models_evaluation.backtest import Backtest

pd.set_option("display.max_columns", 100)

# Call to main
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    
    # Set the seed of numpy's PRNG
    np.random.seed(2019)
        
    max_lags = 57
    test_days = 1
    day_to_predict = 1 # This means we want to predict t + 1 knowing t

    date_to_predict = "2016-04-25"
    train_test_date_split = "2016-04-24"
    eval_start_date = "2016-03-27"
    data_cache_path_str = "E:/M5_Forecasting_Accuracy_cache/"

    dl = DataLoader()
    training_set_df, target_df, testing_set_df, truth_df, orig_target_df, sample_submission_df = dl.load_data_v3(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, day_to_predict, train_test_date_split, enable_validation = False, first_day = 1, max_lags = max_lags)

    print("Training set shape:", training_set_df.shape)
    print("Testing set shape:", testing_set_df.shape)

    backtest = Backtest(training_set_df, target_df, data_cache_path_str, 1535, 28, process_lst = [LGBMDayModel], process_names_lst = ["LGBMDayModel"], nb_folds = 10)
    metrics_df = backtest.run()

    current_date = "18052020"
    metrics_df.to_excel("D:/Projets_Data_Science/Competitions/Kaggle/M5_Forecasting_Accuracy/output/backtest_metrics_" + current_date + ".xlsx")
    
    fig = plt.figure(figsize = (20, 9))
    ax = metrics_df["preds_rmse"].plot.line()
    ax.set_title("RMSE by Date - " + current_date, fontsize = 20, fontweight = "bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    fig.savefig(PLOTS_DIRECTORY_PATH_str + "RMSE_by_date_" + current_date + "_.png", dpi = 300)
    plt.close(fig)

    fig = plt.figure(figsize = (20, 9))
    ax = metrics_df["WRMSSE"].plot.line()
    ax.set_title("WRMSSE by Date (by periods of 28 days) - " + current_date, fontsize = 20, fontweight = "bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("WRMSSE")
    fig.tight_layout()
    fig.savefig(PLOTS_DIRECTORY_PATH_str + "WRMSSE_by_date_" + current_date + "_.png", dpi = 300)
    plt.close(fig)

    """
    # Plot the performance accross time
    scores_df.plot(figsize = (20,7))
    plt.xlabel('Validation end day', fontsize=20)
    plt.ylabel('WRMSSEE score', fontsize=20)
    plt.title('Process performance over time', fontsize=26)
    plt.show()

    # Read scores_df to a csv file to save your backtest information easily
    scores_df.to_csv('backtest_results_' + '_'.join(backtest.process_names) + '.csv')
    """

    # Stop the timer and print the exectution time
    print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")
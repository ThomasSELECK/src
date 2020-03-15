#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# EMLEK - Efficient Machine LEarning toolKit                                  #
#                                                                             #
# This file contains a class defining a cross-validation system.              #
# Developped using Python 3.6.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2018-05-19                                                            #
# Version: 1.0.0                                                              #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import types
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold, StratifiedKFold, GroupKFold
from multiprocessing import Pool
import multiprocessing as mp
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15.0, 10.0]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
import seaborn as sns

class CrossValidator(object):
    """
    The purpose of this class is to cross-validate machine learning models to ensure performance is optimal and the model doesn't suffer from overfitting.
    """

    def __init__(self, evaluation_metrics_lst, monte_carlo_splits = 1, split_type = "random", stratified = True, n_folds = 5, train_window_size = None, test_window_size = None, cv_directory_str = "", n_jobs = -1):
        """
        Class' constructor

        Parameters
        ----------
        evaluation_metrics_lst : list of tuples containing (in this order): a function with the following prototype: metric(y_true, y_preds), a string indicating th metric's name and 
                a string (either "probabilities" or "predictions") indicating if we need to predict the probabilities of a sample to belong to each class or the actual prediction itself.
                These are the metrics we use to evaluate our model.

        monte_carlo_splits : positive integer
                This indicates how many times we do the cross validation with different seeds for the data split. The total number of splits is monteCarloSplits * nFolds

        split_type : string, either "random", "time" or "feature"
                This indicates if we do a classical cross-validation based on a random split or if we do a rolling window framework.
                When "time" is selected, the argument 'monteCarloSplits' is ignored, its value is fixed to 1.

        stratified : boolean
                This indicates if the split is stratified or not. Not used when splitType = "time".

        n_folds : positive integer (default = 5)
                This indicates the number of folds we want to create. Not used when splitType = "time".

        train_window_size : positive integer
                Only used if 'split_type' = "time". Indicates the length of the training window.

        test_window_size : positive integer
                Only used if 'split_type' = "time". Indicates the length of the testing window.

        cv_directory_str : string
                This is the path of the folder where the cross-validation output will be saved.

        n_jobs : integer (default = -1)
                This indicates the number of CPU cores to use to do the processing. If -1, all cores are used.
                
        Returns
        -------
        None
        """

        # Sanity checks
        if type(split_type) != str:
            raise TypeError("The argument 'split_type' MUST be a string, either 'random', 'time' or 'feature'")

        if split_type != "random" and split_type != "time" and split_type != "feature":
            raise ValueError("The argument 'split_type' MUST be a string, either 'random', 'time' or 'feature'")

        if type(stratified) != bool:
            raise TypeError("The argument 'stratified' MUST be a boolean")

        if split_type == "random" and type(n_folds) != int:
            raise TypeError("The argument 'n_folds' MUST be a positive integer")

        if split_type == "random" and n_folds < 0:
            raise ValueError("The argument 'n_folds' MUST be a positive integer")

        if split_type == "time" and (train_window_size is None or test_window_size is None):
            raise ValueError("When the argument 'split_type' is set to 'time', both arguments 'train_window_size' and 'test_window_size' MUST be set to a positive integer value")

        # Class' attributes
        self.evaluation_metrics_lst = evaluation_metrics_lst
        self.monte_carlo_splits = monte_carlo_splits
        self.split_type = split_type
        self.stratified = stratified
        self.n_folds = n_folds
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.cv_directory_str = cv_directory_str

        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs
        
        # Attributes only used at execution
        self._metrics_names_lst = [metric[1] for metric in evaluation_metrics_lst]
        self._metrics_predictions_type_lst = [metric[2] for metric in evaluation_metrics_lst]
        self._statistics_df = None # DataFrame that will contains the cross-validation results
        self._statistics_dict = {} # Temporary dictionary that will contain the cross-validation results

        for metric_name in self._metrics_names_lst:
            self._statistics_dict[metric_name] = []

        # Save predictions made during cross-validation
        self._predictions_df = None
        self._predicted_probabilities_df = None

        manager = mp.Manager()
        self.ns = manager.Namespace()

    def cv_worker(self, data):
        # Unpack the data
        model_name_str = data[0]
        mc_seed = data[1]
        fold_id = data[2]
        model = data[3]
        train_index = data[4]
        test_index = data[5]

        foldTrain_df = self.ns.dataset.iloc[train_index]
        foldTest_df = self.ns.dataset.iloc[test_index]
        foldTarget_sr = self.ns.target.iloc[train_index]
        foldTruth_sr = self.ns.target.iloc[test_index]
        
        # Store the statistics from the cross-validation
        statistics_dict = {}

        # Execute the pipeline for training
        model.fit(foldTrain_df, foldTarget_sr)

        # Execute the pipeline for testing
        if "predictions" in self._metrics_predictions_type_lst: # $WARNING$: using the string "predictions" is not a good practice
            predictions_npa = model.predict(foldTest_df)
            predictions_df = pd.DataFrame({"modelName": model_name_str, "mcSeed": mc_seed, "foldId": fold_id, "predictions": predictions_npa}, index = foldTruth_sr.index)
        else:
            predictions_df = None
            
        if "probabilities" in self._metrics_predictions_type_lst: # $WARNING$: using the string "probabilities" is not a good practice
            predictedProbabilities_npa = model.predict_proba(foldTest_df)
            predictedProbabilities_df = pd.concat([pd.Series([model_name_str for _ in range(foldTest_df.shape[0])], index = foldTruth_sr.index, name = "modelName"), 
                                                   pd.Series(mc_seed * np.ones(foldTest_df.shape[0]), index = foldTruth_sr.index, name = "mcSeed"), 
                                                   pd.Series(fold_id * np.ones(foldTest_df.shape[0]), index = foldTruth_sr.index, name = "foldId"), 
                                                   pd.DataFrame(predictedProbabilities_npa, index = foldTruth_sr.index, columns = ["class_" + str(i) for i in range(predictedProbabilities_npa.shape[1])])], axis = 1)
        else:
            predictedProbabilities_df = None
                   
        # Evaluate model performances
        for metric, metricName, metricPredictionsType in self.evaluation_metrics_lst:
            if metricPredictionsType == "predictions": # $WARNING$: using the string "predictions" is not a good practice
                statistics_dict[metricName] = metric(foldTruth_sr, predictions_npa)

            elif metricPredictionsType == "probabilities": # $WARNING$: using the string "probabilities" is not a good practice
                statistics_dict[metricName] = metric(foldTruth_sr, predictedProbabilities_npa, labels = foldTarget_sr.unique().tolist())

            statistics_dict["modelName"] = model_name_str
            statistics_dict["mcSeed"] = mc_seed
            statistics_dict["foldId"] = fold_id

            # Try to get the number of iterations the model used (for XGBoost or LightGBM) or the number of epochs for a Keras model
            try:
                statistics_dict["modelIterations"] = model._final_estimator.nrounds
            except:
                statistics_dict["modelIterations"] = "None"

        # Try to save feature importance for LightGBM
        try:
            model._final_estimator.get_features_importance().to_excel(self.cv_directory_str + "feature_importance" + model_name_str + "_" + str(mc_seed) + "_" + str(fold_id) + ".xlsx")
        except:
            pass
                                
        return statistics_dict, predictions_df, predictedProbabilities_df

    def _generate_random_data_splits(self, models_pipelines_dict, dataset_df, target_sr):
        """
        This method splits the data into random folds.

        Parameters
        ----------
        models_pipelines_dict : dictionary
                Each dictionary key is a string indicating the name of the model we are testing. This string will be used in the results file to reference the corresponding model.
                The value associated to each key is a Scikit-Learn Pipeline object. Make sure Scikit-Learn Pipeline's arguments are correctly set to what you want to test.

        dataset_df : DataFrame
                This is a DataFrame containing the features we want to use for cross-validation.

        target_sr : Series
                This is a Series containing the target variable we want to predict.
             
        Returns
        -------
        chunkData_lst : Python list
                List containing the folds for cross-validation.
        """

        # Save the data index
        data_index_sr = target_sr.index
        
        # Split the training data into folds
        chunk_data_lst = []
        for model_name, model in models_pipelines_dict.items():
            for mc_seed_idx, mc_seed in enumerate(np.random.randint(0, self.monte_carlo_splits * 1000, self.monte_carlo_splits)):
                if self.stratified:
                    k_fold_splitter = StratifiedKFold(n_splits = self.n_folds, shuffle = True, random_state = mc_seed).split(dataset_df, target_sr)
                else:
                    k_fold_splitter = KFold(n_splits = self.n_folds, shuffle = True, random_state = mc_seed).split(dataset_df)

                for idx, (train_index, test_index) in enumerate(k_fold_splitter):
                    fold_id = mc_seed_idx * self.n_folds + idx                                
                    chunk_data_lst.append((model_name, mc_seed, fold_id, model, train_index, test_index))

        return chunk_data_lst

    def _generate_data_splits_on_feature(self, models_pipelines_dict, dataset_df, target_sr, feature_name):
        """
        This method splits the data into random folds.

        Parameters
        ----------
        models_pipelines_dict : dictionary
                Each dictionary key is a string indicating the name of the model we are testing. This string will be used in the results file to reference the corresponding model.
                The value associated to each key is a Scikit-Learn Pipeline object. Make sure Scikit-Learn Pipeline's arguments are correctly set to what you want to test.

        dataset_df : DataFrame
                This is a DataFrame containing the features we want to use for cross-validation.

        target_sr : Series
                This is a Series containing the target variable we want to predict.

        feature_name : string
                Name of the feature we want to use to make the split.
             
        Returns
        -------
        chunkData_lst : Python list
                List containing the folds for cross-validation.
        """

        # Save the data index
        data_index_sr = target_sr.index
        
        # Split the training data into folds
        chunk_data_lst = []
        for model_name, model in models_pipelines_dict.items():
            k_fold_splitter = GroupKFold(n_splits = self.n_folds).split(dataset_df, target_sr, dataset_df[feature_name].values)
            for fold_id, (train_index, test_index) in enumerate(k_fold_splitter):                             
                chunk_data_lst.append((model_name, 0, fold_id, model, train_index, test_index))

        return chunk_data_lst

    def _generate_time_based_data_splits(self, models_pipelines_dict, dataset_df, target_sr):
        """
        This method splits the data into time delimited folds (rolling window).

        Parameters
        ----------
        modelsPipelines_dict : dictionary
                Each dictionary key is a string indicating the name of the model we are testing. This string will be used in the results file to reference the corresponding model.
                The value associated to each key is a Scikit-Learn Pipeline object. Make sure Scikit-Learn Pipeline's arguments are correctly set to what you want to test.

        dataset_df : DataFrame
                This is a DataFrame containing the features we want to use for cross-validation.

        target_sr : Series
                This is a Series containing the target variable we want to predict.
             
        Returns
        -------
        chunkData_lst : Python list
                List containing the folds for cross-validation.
        """
        
        # Split the training data into folds
        chunk_data_lst = []
        for model_name, model in models_pipelines_dict.items():
            for fold_id in range((dataset_df.shape[0] - self.train_window_size) // self.test_window_size):
                train_index = list(range((fold_id * self.test_window_size), (fold_id * self.test_window_size + self.train_window_size)))
                test_index = list(range((fold_id * self.test_window_size + self.train_window_size), ((fold_id * self.test_window_size + self.train_window_size) + self.test_window_size)))
                                         
                chunk_data_lst.append((model_name, 0, fold_id, model, train_index, test_index))

        print("Generated ", len(chunk_data_lst), "data folds")

        return chunk_data_lst

    def cross_validate(self, modelsPipelines_dict, dataset_df, target_sr, feature_name = None):
        """
        This method does the actual cross validation.

        Parameters
        ----------
        modelsPipelines_dict : dictionary
                Each dictionary key is a string indicating the name of the model we are testing. This string will be used in the results file to reference the corresponding model.
                The value associated to each key is a Scikit-Learn Pipeline object. Make sure Scikit-Learn Pipeline's arguments are correctly set to what you want to test.

        dataset_df : DataFrame
                This is a DataFrame containing the features we want to use for cross-validation.

        target_sr : Series
                This is a Series containing the target variable we want to predict.

        feature_name : string (default = None)
                Categorical feature to use to split the data. Not used if cvType is not "feature".
                
        Returns
        -------
        None
        """

        # Save the data index
        data_index_sr = target_sr.index

        if self.split_type == "random":
            chunk_data_lst = self._generate_random_data_splits(modelsPipelines_dict, dataset_df, target_sr)
        elif self.split_type == "time":
            chunk_data_lst = self._generate_time_based_data_splits(modelsPipelines_dict, dataset_df, target_sr)
        elif self.split_type == "feature":
            chunk_data_lst = self._generate_data_splits_on_feature(modelsPipelines_dict, dataset_df, target_sr, feature_name)
            
        self.ns.dataset = dataset_df
        self.ns.target = target_sr

        pool = mp.Pool(self.n_jobs)
        results_lst = pool.map(self.cv_worker, chunk_data_lst)
        pool.close()
        
        # Concatenate results into a single data structure
        for idx, (statistics_dict, predictions_df, predictedProbabilities_df) in enumerate(results_lst):
            if "predictions" in self._metrics_predictions_type_lst: # $WARNING$: using the string "predictions" is not a good practice
                # Store predictions
                if predictions_df is not None:
                    if self._predictions_df is None:
                        self._predictions_df = predictions_df
                    else:
                        self._predictions_df = pd.concat([self._predictions_df, predictions_df], axis = 0)
                
            if "probabilities" in self._metrics_predictions_type_lst: # $WARNING$: using the string "probabilities" is not a good practice
                # Store predictions
                if predictedProbabilities_df is not None:
                    if self._predicted_probabilities_df is None:
                        self._predicted_probabilities_df = predictedProbabilities_df
                    else:
                        self._predicted_probabilities_df = pd.concat([self._predicted_probabilities_df, predictedProbabilities_df], axis = 0)

            # Store statistics
            for key in statistics_dict:
                if key in self._statistics_dict:
                    self._statistics_dict[key].append(statistics_dict[key])
                else:
                    self._statistics_dict[key] = [statistics_dict[key]]

        # Put cross-validation statistics into a DataFrame
        self._statistics_df = pd.DataFrame(self._statistics_dict)

        # Get models hyperparameters
        modelsParams_dict = {}
        for modelName, model in modelsPipelines_dict.items():
            pipelineArguments_dict = model._final_estimator.get_params()
            modelParams_dict = {}
            for key, value in pipelineArguments_dict.items():
                if type(pipelineArguments_dict[key]) == int or type(pipelineArguments_dict[key]) == float or type(pipelineArguments_dict[key]) == str or type(pipelineArguments_dict[key]) == bool or type(pipelineArguments_dict[key]) == dict:
                    if len(key.split("__")) == 2:
                        modelParams_dict[key.split("__")[1]] = str(value)
                    else:
                        modelParams_dict[key] = str(value)

            modelParams_str = ""
            for key in sorted(modelParams_dict.keys()):
                modelParams_str += key + " = " + modelParams_dict[key] + "; "

            modelsParams_dict[modelName] = modelParams_str[:-1] # Remove last semicolon

        self._statistics_df["modelParams"] = self._statistics_df["modelName"].map(modelsParams_dict)

        # Reorder rows
        self._statistics_df.sort_values(["modelName", "mcSeed", "foldId"], inplace = True)

        if self._predictions_df is not None:
            if self.split_type == "random" or self.split_type == "feature":
                self._predictions_df = self._predictions_df.groupby("mcSeed").apply(lambda x: x.reindex(data_index_sr))
                self._predictions_df.index = self._predictions_df.index.get_level_values(1)
            elif self.split_type == "time":
                self._predictions_df.sort_values(["modelName", "foldId"], inplace = True)

        if self._predicted_probabilities_df is not None:
            if self.split_type == "random" or self.split_type == "feature":
                self._predicted_probabilities_df = self._predicted_probabilities_df.groupby("mcSeed").apply(lambda x: x.reindex(data_index_sr))
                self._predicted_probabilities_df.index = self._predicted_probabilities_df.index.get_level_values(1)
            elif self.split_type == "time":
                self._predicted_probabilities_df.sort_values(["modelName", "foldId"], inplace = True)
            
        # Reorder columns for statistics
        columnsOrder_lst = ["modelName", "mcSeed", "foldId"] + self._metrics_names_lst + ["modelIterations", "modelParams"]
        self._statistics_df = self._statistics_df[columnsOrder_lst]

        # Compute aggregated statistics
        self._aggregatedStatistics_df = self._statistics_df.groupby("modelName")[self._metrics_names_lst].agg([np.mean, np.std]).reset_index()
        self._aggregatedStatistics_df.columns = [" ".join(colName).strip() for colName in self._aggregatedStatistics_df.columns.values]
        if self.split_type == "random" or self.split_type == "feature":
            self._aggregatedStatistics_df["number of CV folds"] = self.monte_carlo_splits * self.n_folds
        else:
            self._aggregatedStatistics_df["number of CV folds"] = (dataset_df.shape[0] - self.train_window_size) // self.test_window_size
        self._aggregatedStatistics_df = self._aggregatedStatistics_df[["modelName", "number of CV folds"] + sum([[metric + " mean", metric + " std"] for metric in self._metrics_names_lst], [])]

        for modelName in modelsPipelines_dict.keys():
            for metric, metricName, metricPredictionsType in self.evaluation_metrics_lst:
                print(modelName, ":")
                print("    CV avg", metricName, ":", np.asscalar(self._aggregatedStatistics_df.loc[self._aggregatedStatistics_df["modelName"] == modelName, metricName + " mean"].values), "CV std", metricName, ":", np.asscalar(self._aggregatedStatistics_df.loc[self._aggregatedStatistics_df["modelName"] == modelName, metricName + " std"].values))
                for mcSplitIdx in range(self.monte_carlo_splits):
                    if self.split_type == "random" or self.split_type == "feature":
                        if metricPredictionsType == "predictions": # $WARNING$: using the string "predictions" is not a good practice
                            tmp_sr = self._predictions_df.loc[self._predictions_df["modelName"] == modelName, "predictions"].iloc[(mcSplitIdx * target_sr.shape[0]):((mcSplitIdx + 1) * target_sr.shape[0])]
                            print("    For MC split", mcSplitIdx, ": Whole dataset", metricName, ":", metric(target_sr, tmp_sr))
                        elif metricPredictionsType == "probabilities": # $WARNING$: using the string "probabilities" is not a good practice
                            tmp_sr = self._predicted_probabilities_df.filter(regex = "class_").loc[self._predicted_probabilities_df["modelName"] == modelName].iloc[(mcSplitIdx * target_sr.shape[0]):((mcSplitIdx + 1) * target_sr.shape[0])]
                            print("    For MC split", mcSplitIdx, ": Whole dataset", metricName, ":", metric(target_sr, tmp_sr))

                    elif self.split_type == "time":
                        if metricPredictionsType == "predictions": # $WARNING$: using the string "predictions" is not a good practice
                            print("    For MC split", mcSplitIdx, ": Whole dataset", metricName, ":", metric(target_sr.iloc[self.train_window_size:], self._predictions_df.loc[self._predictions_df["modelName"] == modelName, "predictions"]))
                        elif metricPredictionsType == "probabilities": # $WARNING$: using the string "probabilities" is not a good practice
                            print("    For MC split", mcSplitIdx, ": Whole dataset", metricName, ":", metric(target_sr.iloc[self.train_window_size:], self._predicted_probabilities_df.filter(regex = "class_.*").loc[self._predictions_df["modelName"] == modelName]))

        # Save statistics and predictions
        with pd.ExcelWriter(self.cv_directory_str + "cv_statistics.xlsx") as writer:
            self._aggregatedStatistics_df.to_excel(writer, sheet_name = "Aggregated CV Statistics", index = False)
            self._statistics_df.to_excel(writer, sheet_name = "Cross-Validation Statistics", index = False)

        if self._predictions_df is not None:
            # Reorder columns for predictions
            self._predictions_df = self._predictions_df[["modelName", "mcSeed", "foldId", "predictions"]]
            self._predictions_df.to_csv(self.cv_directory_str + "cv_actual_predictions.csv", index = False)

        if self._predicted_probabilities_df is not None:
            # Reorder columns for predicted probabilities
            columns_lst = self._predicted_probabilities_df.filter(regex = "class_.*").columns.tolist()
            self._predicted_probabilities_df = self._predicted_probabilities_df[["modelName", "mcSeed", "foldId"] + columns_lst]
            self._predicted_probabilities_df.to_csv(self.cv_directory_str + "cv_predicted_probabilities.csv", index = False)

        return self._statistics_df, self._predictions_df, self._predicted_probabilities_df

    def plot_cross_validation_statistics(self, plotsDirectoryPath_str):
        """
        This method plots the results of the cross validation.

        Parameters
        ----------
        plotsDirectoryPath_str : string
                Path of the directory where the plots will be stored.

        Returns
        -------
        None
        """
                
        for modelName in self._statistics_df["modelName"].unique().tolist():
            # Plot metrics line plot
            plt.style.use("ggplot")
            plt.figure()
            tmp_df = self._statistics_df.loc[self._statistics_df["modelName"] == modelName, ["foldId"] + self._metrics_names_lst]
            tmp_df.index = tmp_df["foldId"]
            tmp_df.sort_index(inplace = True)
            tmp_df.drop("foldId", axis = 1, inplace = True)
            tmp_df.plot.line()
            metrics_str = ""
            for metric_name in self._metrics_names_lst:
                metrics_str += "Avg " + metric_name + ": " + str(np.asscalar(self._aggregatedStatistics_df.loc[self._aggregatedStatistics_df["modelName"] == modelName, metric_name + " mean"].values)) + "; " + metric_name + " std: " + str(np.asscalar(self._aggregatedStatistics_df.loc[self._aggregatedStatistics_df["modelName"] == modelName, metric_name + " std"].values))
                metrics_str += "\n"
            plt.title("Metrics scores wrt foldId for model:\n\"" + modelName + "\"\n" + metrics_str)
            plt.xlabel("FoldId")
            plt.ylabel("Score")
            """if self._statistics_df.loc[self._statistics_df["modelName"] == modelName].shape[0] < 11:
                plt.xticks(self._statistics_df["foldId"].loc[self._statistics_df["modelName"] == modelName].values)
            else:
                plt.xticks([int(t) for t in np.linspace(0, self._statistics_df["foldId"].loc[self._statistics_df["modelName"] == modelName].max(), 11)])"""

            plt.savefig(plotsDirectoryPath_str + "metrics_plot_" + modelName + ".png")
            plt.clf()

            # Plot metrics histograms
            for metricName in self._metrics_names_lst:
                nbBins = min(self._statistics_df[metricName].loc[self._statistics_df["modelName"] == modelName].unique().shape[0], max(5, self._statistics_df.loc[self._statistics_df["modelName"] == modelName].shape[0] // 5))
                plt.figure()
                sns.distplot(self._statistics_df[metricName].loc[self._statistics_df["modelName"] == modelName], bins = nbBins, kde = False)
                plt.title(modelName + ": Histogram of the " + metricName + " score\nMean: " + str(self._statistics_df[metricName].loc[self._statistics_df["modelName"] == modelName].mean()) + "\nStd: " + str(self._statistics_df[metricName].loc[self._statistics_df["modelName"] == modelName].std()))
                plt.xlabel(metricName + " score")
                plt.ylabel("Count")
                
                plt.savefig(plotsDirectoryPath_str + metricName + "_score_histogram_" + modelName + ".png")
                plt.clf()
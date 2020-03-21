#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# First solution for the M5 Forecasting Accuracy competition                  #
#                                                                             #
# This file contains the code needed to execute multiple groupby operations   #
# in parallel.                                                                #
# Developped using Python 3.8.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2020-03-15                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import time
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp

class ParallelGroupBy():
    """
    This class defines allows to execute multiple groupby operations in parallel
    """

    def __init__(self, cpu_cores = -1):
        """
        This is the class' constructor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if cpu_cores > 0:
            self._cpu_cores = cpu_cores
        else:
            self._cpu_cores = mp.cpu_count()
                
    def _worker(self, q_in, iolock, res_dict):        
        while True:
            # Get item
            item = q_in.get()
            
            if item is None:
                break
                
            # Get data back
            X = item[0]
            shift = item[1]
            rolling_size = item[2]
            feature_type = item[3]
            feature_name = item[4]
                
            # Do job
            if feature_type == "shift_t":
                res = X.groupby(["id"])[feature_name].transform(lambda x: x.shift(shift))
            elif feature_type == "rolling_std_t":
                res = X.groupby(["id"])[feature_name].transform(lambda x: x.shift(shift).rolling(rolling_size).std())
            elif feature_type == "rolling_mean_t":
                res = X.groupby(["id"])[feature_name].transform(lambda x: x.shift(shift).rolling(rolling_size).mean())
            elif feature_type == "rolling_skew_t":
                res = X.groupby(["id"])[feature_name].transform(lambda x: x.shift(shift).rolling(rolling_size).skew())
            elif feature_type == "rolling_kurt_t":
                res = X.groupby(["id"])[feature_name].transform(lambda x: x.shift(shift).rolling(rolling_size).kurt())
            elif feature_type == "rolling_max_t":
                res = X.groupby(["id"])[feature_name].transform(lambda x: x.shift(shift).rolling(rolling_size).max())
            else:
                res = None
            
            # Save result
            if feature_type == "shift_t":
                feature_name += "_" + feature_type + str(shift)
            else:
                feature_name += "_" + feature_type + str(rolling_size)
                
            res_dict[feature_name] = res
                                    
    def run(self, data_df):
        q_in = mp.Queue()
        iolock = mp.Lock()
        manager = mp.Manager()
        res_dict = manager.dict()
                
        # define the processes
        workers_pool = [mp.Process(target = self._worker, args = (q_in, iolock, res_dict)) for i in range(self._cpu_cores)]
        
        # process images with n_cores - 1 process
        for p in workers_pool:
            p.start()
            
        # put data into input queue
        tmp_data_df = data_df[["id", "demand"]]
        sizes_lst = [28, 29, 30]
        feature_type = "shift_t"
        feature_name = "demand"
        for shift_size in sizes_lst:
            q_in.put((tmp_data_df.copy(), shift_size, 0, feature_type, feature_name))
            
        sizes_lst = [7, 30, 60, 90, 180]
        feature_type = "rolling_std_t"
        for rolling_size in sizes_lst:
            q_in.put((tmp_data_df.copy(), 28, rolling_size, feature_type, feature_name))
            
        feature_type = "rolling_mean_t"
        for rolling_size in sizes_lst:
            q_in.put((tmp_data_df.copy(), 28, rolling_size, feature_type, feature_name))
            
        q_in.put((tmp_data_df.copy(), 28, 30, "rolling_skew_t", feature_name))
        q_in.put((tmp_data_df.copy(), 28, 30, "rolling_kurt_t", feature_name))
        
        tmp_data_df = data_df[["id", "sell_price"]]
        feature_name = "sell_price"
        
        q_in.put((tmp_data_df.copy(), 1, 0, "shift_t", feature_name))
        q_in.put((tmp_data_df.copy(), 1, 365, "rolling_max_t", feature_name))
        q_in.put((tmp_data_df.copy(), 0, 7, "rolling_std_t", feature_name))
        q_in.put((tmp_data_df.copy(), 0, 30, "rolling_std_t", feature_name))
        
        # tell workers we're done
        for _ in range(self._cpu_cores):  
            q_in.put(None)
                        
        for p in workers_pool:
            p.join()
                                        
        orig_col_lst = data_df.columns.tolist()
        for feature, data in dict(res_dict).items():
            data_df[feature] = data
            
        data_df.drop(orig_col_lst, axis = 1, inplace = True)

        return data_df
# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
import lightgbm as lgb
from math import ceil
import time
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
import multiprocessing as mp
from joblib import Parallel, delayed
import re

from dev.files_paths import *
from m5_forecasting_accuracy.data_loading.data_loader import DataLoader
from m5_forecasting_accuracy.preprocessing.PreprocessingStep4 import PreprocessingStep4

warnings.filterwarnings('ignore')

# Seed to make all processes deterministic
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
## Multiprocess Runs
def df_parallelize_run(func, t_split, base_test):
    num_cores = mp.cpu_count()
    df = pd.concat(Parallel(n_jobs = num_cores, max_nbytes = None)(delayed(func)(ts, base_test.copy()) for ts in t_split), axis = 1)
    return df

# Helper to load data by store ID
# Read data
def get_data_by_store(store):
    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(BASE), pd.read_pickle(PRICE).iloc[:,2:], pd.read_pickle(CALENDAR).iloc[:,2:]], axis=1)
    
    # Leave only relevant store
    df = df[df['store_id']==store]

    # With memory limits we have to read lags and mean encoding features
    # separately and drop items that we don't need.
    # As our Features Grids are aligned we can use index to keep only necessary rows
    # Alignment is good for us as concat uses less memory than merge.
    df2 = pd.read_pickle(MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
    
    df3 = pd.read_pickle(LAGS).iloc[:,3:]
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2 # to not reach memory limit 
    
    df = pd.concat([df, df3], axis=1)
    del df3 # to not reach memory limit 
    
    # Create features list
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET]+features]
    
    # Skipping first n rows
    df = df[df['d']>=START_TRAIN].reset_index(drop=True)
    
    return df, features

# Recombine Test set after training
def get_base_test():
    base_test = pd.DataFrame()

    for store_id in STORES_IDS:
        temp_df = pd.read_pickle("E:/tmp/kernels/aux_model/" + 'test_' + store_id + '.pkl')
        temp_df['store_id'] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    
    return base_test


# Helper to make dynamic rolling lags
def make_lag_roll(LAG_DAY, base_test):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]

# Call to main
if __name__ == "__main__":
    start_time2 = time.time()
    # Vars
    VER = 1                          # Our model version
    SEED = 42                        # We want all things
    seed_everything(SEED)            # to be as deterministic 
    lgb_params = {
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
    lgb_params["seed"] = SEED        # as possible
    N_CORES = psutil.cpu_count()     # Available CPU cores

    #LIMITS and const
    TARGET      = "sales"            # Our target
    START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
    END_TRAIN   = 1913               # End day of our train set
    P_HORIZON   = 28                 # Prediction horizon
    USE_AUX     = True               # Use or not pretrained models

    #FEATURES to remove
    ## These features lead to overfit
    ## or values not present in test set
    remove_features = ["id","state_id","store_id", "date","wm_yr_wk","d",TARGET]
    mean_features   = ["enc_cat_id_mean","enc_cat_id_std", "enc_dept_id_mean","enc_dept_id_std", "enc_item_id_mean","enc_item_id_std"] 

    #PATHS for Features
    ORIGINAL = "D:/Projets_Data_Science/Competitions/Kaggle/M5_Forecasting_Accuracy/data/raw/"
    BASE     = "E:/tmp/kernels/grid_part_1.pkl"
    PRICE    = "E:/tmp/kernels/grid_part_2.pkl"
    CALENDAR = "E:/tmp/kernels/grid_part_3.pkl"
    LAGS     = "E:/tmp/kernels/lags_df_28.pkl"
    MEAN_ENC = "E:/tmp/kernels/mean_encoding_df.pkl"

    # AUX(pretrained) Models paths
    AUX_MODELS = "E:/tmp/kernels/aux_model/"

    #STORES ids
    STORES_IDS = pd.read_csv(ORIGINAL + "sales_train_validation.csv")["store_id"]
    STORES_IDS = list(STORES_IDS.unique())

    #SPLITS for lags creation
    SHIFT_DAY = 28
    N_LAGS = 15
    LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
    ROLS_SPLIT = []
    for i in [1,7,14]:
        for j in [7,14,30,60]:
            ROLS_SPLIT.append([i,j])

    """
    with open("E:/M5_Forecasting_Accuracy_cache/orig_train_data.pkl", "rb") as f:
        grid_df, features_columns, store_id, base_test = pickle.load(f)
    """

    day_to_predict = 1
    date_to_predict = "2016-04-25"
    train_test_date_split = "2016-04-24"
    eval_start_date = "2016-03-27"
    dl = DataLoader()
    training_set_df, target_df, testing_set_df, truth_df, orig_target_df, sample_submission_df = dl.load_data_v3(CALENDAR_PATH_str, SELL_PRICES_PATH_str, SALES_TRAIN_PATH_str, SAMPLE_SUBMISSION_PATH_str, day_to_predict, train_test_date_split, enable_validation = False, first_day = 1, max_lags = 57, shift_target = False)
    gc.collect()

    print("Statistics for all data:")
    print("    Training set shape:", training_set_df.shape)
    print("    Testing set shape:", testing_set_df.shape)

    prp = PreprocessingStep4(dt_col = "date", keep_last_train_days = 209) #366
    y_train = target_df["demand"].reset_index(drop = True)
    training_set_df = prp.fit_transform(training_set_df, y_train) # y is not used here
    gc.collect()

    with open("E:/M5_Forecasting_Accuracy_cache/processed_data_24052020.pkl", "wb") as f:
        pickle.dump((training_set_df, testing_set_df, target_df, y_train, prp), f)

    """
    with open("E:/M5_Forecasting_Accuracy_cache/processed_data_24052020.pkl", "rb") as f:
        training_set_df, testing_set_df, target_df, y_train, prp = pickle.load(f)

    training_set_df["d"] = training_set_df["d"].str.replace("d_", "").apply(lambda x: int(x))
    testing_set_df["d"] = testing_set_df["d"].str.replace("d_", "").apply(lambda x: int(x))
    tmp = pd.concat([training_set_df, testing_set_df], axis = 0)
    tmp = tmp.loc[tmp["store_id"] == store_id] # Should be (4788267, 23)
    tmp = tmp.loc[tmp["d"] < 1914]
    grid_df = grid_df.loc[grid_df["d"] < 1914]
    print("tmp.shape:", tmp.shape)
    print("grid_df.shape:", grid_df.shape)

    grid_df.columns = [c.replace("tmp_", "") for c in grid_df.columns.tolist()]
    grid_df.columns = [re.sub("(mean|std|sum)(_[0-9]+)$", "\\1_28\\2", c) if re.match("rolling_(mean|std|sum)(_[0-9]+)$", c) else c for c in grid_df.columns.tolist()]

    tmp = tmp[list(set(tmp.columns.tolist()) & set(grid_df.columns.tolist()))]
    tmp.sort_values(["id", "d"], ascending = True, inplace = True)
    tmp = tmp.reset_index(drop = True)

    grid_df.sort_values(["id", "d"], ascending = True, inplace = True)
    grid_df = grid_df.reset_index(drop = True)

    for col in tmp.columns:
        try:
            if not tmp[col].equals(grid_df[col]):
                print(col, "not equal!")
        except:
            print("failed for:", col)

    feats_lst = ["rolling_std_28_7", "rolling_std_28_14", "rolling_std_28_30", "rolling_std_28_60", "rolling_std_28_180", "rolling_mean_1_7", "rolling_mean_1_14", "rolling_mean_1_30", "rolling_mean_1_60", 
    "rolling_mean_7_7", "rolling_mean_7_14", "rolling_mean_7_30", "rolling_mean_7_60", "rolling_mean_14_7", "rolling_mean_14_14", "rolling_mean_14_30", "rolling_mean_14_60"]

    price_momentum_m not equal!
    enc_dept_id_mean not equal!
    enc_item_id_std not equal!
    price_std not equal!
    enc_dept_id_std not equal!
    enc_cat_id_std not equal!
    item_nunique not equal!
    rolling_std_28_60 not equal!
    rolling_std_28_30 not equal!
    price_momentum_y not equal!
    enc_item_id_mean not equal!
    enc_cat_id_mean not equal!

    ['tm_y', 'tm_dw', 'price_momentum_m', 'd', 'price_min', 'enc_dept_id_mean', 'price_momentum', 'price_std', 'enc_dept_id_std', 'price_mean', 'rolling_std_28_60', 'tm_wm', 'tm_d', 'price_momentum_y', 'tm_m', 'rolling_std_28_30', 'price_norm', 'sell_price', 'price_max', 'enc_cat_id_mean', 'enc_cat_id_std', 'enc_item_id_mean', 'tm_w_end', 'price_nunique', 'enc_item_id_std', 'release', 'item_nunique', 'tm_w']

    tm_y is ok but not in correct data type
    tm_dw is ok but not in correct data type
    d is ok but not in correct data type
    tm_wm is ok but not in correct data type
    tm_d is ok but not in correct data type
    tm_m is ok but not in correct data type
    sell_price is ok but not in correct data type
    tm_w_end is ok but not in correct data type
    tm_w is ok but not in correct data type

    # For test:
    store_id not equal!
    rolling_std_28_7 not equal!
    rolling_std_28_14 not equal!
    rolling_std_28_30 not equal!
    rolling_std_28_60 not equal!
    rolling_std_28_180 not equal!
    rolling_mean_1_7 not equal!
    rolling_mean_1_14 not equal!
    rolling_mean_1_30 not equal!
    rolling_mean_1_60 not equal!
    rolling_mean_7_7 not equal!
    rolling_mean_7_14 not equal!
    rolling_mean_7_30 not equal!
    rolling_mean_7_60 not equal!
    rolling_mean_14_7 not equal!
    rolling_mean_14_14 not equal!
    rolling_mean_14_30 not equal!
    rolling_mean_14_60 not equal!
    """

    # Train Models
    for store_id in STORES_IDS:
        print('Train', store_id)
    
        # Get grid for current store
        #grid_df, features_columns = get_data_by_store(store_id)
        grid_df = training_set_df.loc[training_set_df["store_id"] == store_id]
        y_store_train = y_train.loc[training_set_df["store_id"] == store_id]
    
        # Masks for 
        # Train (All data less than 1913)
        # "Validation" (Last 28 days - not real validation set)
        # Test (All data greater than 1913 day, with some gap for recursive features)   
        X_store_train = grid_df.loc[grid_df["d"] <= END_TRAIN]
        X_store_train.drop(["id", "d", "store_id"], axis = 1, inplace = True)
        X_store_valid = grid_df.loc[(grid_df["d"] <= END_TRAIN) & (grid_df["d"] > (END_TRAIN - P_HORIZON))]
        X_store_valid.drop(["id", "d", "store_id"], axis = 1, inplace = True)
        features_columns = grid_df.columns.tolist()
        train_data = lgb.Dataset(X_store_train, label = y_store_train.loc[grid_df["d"] <= END_TRAIN])
        valid_data = lgb.Dataset(X_store_valid, label = y_store_train.loc[(grid_df["d"] <= END_TRAIN) & (grid_df["d"] > (END_TRAIN - P_HORIZON))])
    
        # Saving part of the dataset for later predictions
        # Removing features that we need to calculate recursively 
        grid_df = grid_df.loc[grid_df["d"] > (END_TRAIN - 100)].reset_index(drop = True)
        keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
        grid_df = grid_df[keep_cols]
        grid_df.to_pickle("E:/tmp/kernels/aux_model/" + 'test_' + store_id + '.pkl')
        del grid_df
    
        # Launch seeder again to make lgb training 100% deterministic
        # with each "code line" np.random "evolves" 
        # so we need (may want) to "reset" it
        seed_everything(SEED)
        estimator = lgb.train(lgb_params, train_data, valid_sets = [valid_data], verbose_eval = 100)
    
        # Save model - it's not real '.bin' but a pickle file
        # estimator = lgb.Booster(model_file='model.txt')
        # can only predict with the best iteration (or the saving iteration)
        # pickle.dump gives us more flexibility
        # like estimator.predict(TEST, num_iteration=100)
        # num_iteration - number of iteration want to predict with, 
        # NULL or <= 0 means use best iteration
        model_name = "E:/tmp/kernels/aux_model/" + 'lgb_model_' + store_id + '_v' + str(VER) + '.bin'
        pickle.dump(estimator, open(model_name, 'wb'))

        # Remove temporary files and objects 
        # to free some hdd space and ram memory
        #get_ipython().system('rm train_data.bin')
        del train_data, valid_data, estimator
        gc.collect()
    
        # "Keep" models features for predictions
        MODEL_FEATURES = features_columns
        #MODEL_FEATURES = ['store_id', 'd', 'item_id', 'dept_id', 'cat_id', 'release', 'sell_price', 'price_max', 'price_min', 'price_std', 'price_mean', 'price_norm', 'price_nunique', 'item_nunique', 'price_momentum', 'price_momentum_m', 'price_momentum_y', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'tm_d', 'tm_w', 'tm_m', 'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end', 'enc_cat_id_mean', 'enc_cat_id_std', 'enc_dept_id_mean', 'enc_dept_id_std', 'enc_item_id_mean', 'enc_item_id_std', 'sales_lag_28', 'sales_lag_29', 'sales_lag_30', 'sales_lag_31', 'sales_lag_32', 'sales_lag_33', 'sales_lag_34', 'sales_lag_35', 'sales_lag_36', 'sales_lag_37', 'sales_lag_38', 'sales_lag_39', 'sales_lag_40', 'sales_lag_41', 'sales_lag_42', 'rolling_mean_28_7', 'rolling_std_28_7', 'rolling_mean_28_14', 'rolling_std_28_14', 'rolling_mean_28_30', 'rolling_std_28_30', 'rolling_mean_28_60', 'rolling_std_28_60', 'rolling_mean_28_180', 'rolling_std_28_180', 'rolling_mean_1_7', 'rolling_mean_1_14', 'rolling_mean_1_30', 'rolling_mean_1_60', 'rolling_mean_7_7', 'rolling_mean_7_14', 'rolling_mean_7_30', 'rolling_mean_7_60', 'rolling_mean_14_7', 'rolling_mean_14_14', 'rolling_mean_14_30', 'rolling_mean_14_60']
        print("MODEL_FEATURES", MODEL_FEATURES)

    del training_set_df, target_df, truth_df
    gc.collect()

    prp.num_cores = 16

    # Predict
    # Create Dummy DataFrame to store predictions
    all_preds = pd.DataFrame()

    # Join back the Test dataset with a small part of the training data to make recursive features
    #base_test = get_base_test()
    TARGET = "shifted_demand"
    testing_set_df[TARGET] = np.nan

    # Timer to measure predictions time 
    main_time = time.time()

    # Loop over each prediction day as rolling lags are the most timeconsuming we will calculate it for whole day
    for PREDICT_DAY in range(1,29):    
        print('Predict | Day:', PREDICT_DAY)
        start_time = time.time()

        # Make temporary grid to calculate rolling lags
        grid_df = testing_set_df.copy()
        grid_df = prp.transform(grid_df)
        
        ids_df_lst = []
        for store_id in STORES_IDS:
            # Read all our models and make predictions for each day/store pairs
            model_path = 'lgb_model_' + store_id + '_v' + str(VER) + '.bin' 
            if USE_AUX:
                model_path = AUX_MODELS + model_path
                
            estimator = pickle.load(open(model_path, 'rb'))
            grid_df2 = grid_df.loc[(grid_df["d"] == (END_TRAIN + PREDICT_DAY)) & (grid_df["store_id"] == store_id)].drop(["id", "d", "store_id"], axis = 1)
            ids_df = grid_df[["id", "d"]].loc[(grid_df["d"] == (END_TRAIN + PREDICT_DAY)) & (grid_df["store_id"] == store_id)]
            preds = estimator.predict(grid_df2)
            print("For day:", PREDICT_DAY, "and store:", store_id, ": pred.min() =", preds.min(), "; pred.max() =", preds.max())
            ids_df = ids_df.assign(tmp_demand = preds)
            ids_df_lst.append(ids_df)

        ids_df = pd.concat(ids_df_lst, axis = 0)
        ids_df["d"] = ids_df["d"].apply(lambda x: "d_" + str(x))
        testing_set_df = testing_set_df.merge(ids_df, how = "left", on = ["id", "d"])
        testing_set_df[TARGET].loc[~testing_set_df["tmp_demand"].isnull()] = testing_set_df["tmp_demand"].loc[~testing_set_df["tmp_demand"].isnull()]
        testing_set_df.drop("tmp_demand", axis = 1, inplace = True)
    
        # Make good column naming and add 
        # to all_preds DataFrame        
        temp_df = testing_set_df[['id', TARGET]].loc[testing_set_df['d'] == "d_" + str(END_TRAIN + PREDICT_DAY)]
        temp_df.columns = ['id', 'F' + str(PREDICT_DAY)]
        if 'id' in list(all_preds):
            all_preds = all_preds.merge(temp_df, on = ['id'], how = 'left')
        else:
            all_preds = temp_df.copy()
        
        print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60), ' %0.2f min total |' % ((time.time() - main_time) / 60), ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
        del temp_df
    
    all_preds = all_preds.reset_index(drop=True)

    # Export
    # Reading competition sample submission and merging our predictions as we have predictions only for "_validation" data
    # we need to do fillna() for "_evaluation" items
    submission = pd.read_csv(ORIGINAL+'sample_submission.csv')[['id']]
    submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
    submission.to_csv("E:/tmp/kernels/submission_v" + str(VER) + '_01062020.csv', index = False)

    print("*** kaggle script executed in:", time.time() - start_time2, "seconds ***") # 9036.16752243042 seconds
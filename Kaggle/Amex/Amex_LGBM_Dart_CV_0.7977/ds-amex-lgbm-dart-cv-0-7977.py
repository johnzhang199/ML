#!/usr/bin/env python
# coding: utf-8

# # Comments:
#     
# This is an improvement of my baseline, you can find it here: https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7963
# 
# The main difference between this solution and previous one is that we add new features and do seed blend to boost LB. Single 5 kfold model using seed 42 achieve an out of folds CV of 0.7977 and a public leaderboard of 0.799. If we use seed blend (train three different models using seed 42, 52, 62 and then average predictions) the LB boost niceley.
# 
# The main features that boost CV are the following:
# 
# * The difference between last value and the lag1
# * The difference between last value and the average (this features gives a nice boost)
# 
# This feature engineer is done on all the last columns, so we actually add a lot of features, this model used 1368 features.
# 
# I uploaded test predictions to avoid running training and inference
# 
# Next Steps:
# 
# * Could try feature selection, maybe a lot of the feature are just noise, actually I perform permutation importance and I reduce the amount of features to 1000 app and the CV was almost the same. Maybe there is a better feature selection technique that can boost performance.
# 
# * Could try different models, maybe some neural network with the same features or a subset of the features and then blend with LGBM can work, in my experience blending tree models and neural network works great because they are very diverse so the boost is nice
# 
# * Could try more feature engineering, maybe we can create more features that extract the hidden signal of the dataset, actually I would first work on this option and really try to capture all the signal that the dataset has.

# # Preprocessing

# In[ ]:


# ====================================================
# Library
# ====================================================
import gc
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
import numpy as np
import pandas as pd


# ====================================================
# Library
# ====================================================
import os
import gc
import warnings
warnings.filterwarnings('ignore')
import random
import scipy as sp
import numpy as np
import pandas as pd
import joblib
import itertools
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from itertools import combinations

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
import itertools


def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])
    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1] / gini[0] + top_four)
def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True


class model_amex:
    def __init__(self):
        # to speed up set this to true and then only train data will be used to similuate both train andtest and then compute the out-of-sample score using the simultaed test data
        self.DOWN_SAMPLING = True
        # when building the model set this to false so that I can compute the score using simulated test data. real test data from Kaggle do not have label.
        self.USE_TEST_DATA = False
        # number of the data (for amex, userid's that will be sampled
        # self.SIZE_TRAIN = 10000
        # self.SIZE_TEST = 10000
        # self.SIZE_TRAIN = 1000000000
        # self.SIZE_TEST = 500000
        self.PROD = True
        if not self.PROD:
            self.TOTAL_SAMPLING_PCNT = 0.02
            self.NUM_BATCHES_all_feature = 2
        else:
            self.TOTAL_SAMPLING_PCNT = 1
            self.NUM_BATCHES_all_feature = 2

        self.TEST_PCNT = 0.1

        # folder where data are located such as train and test data
        self._DIR_DATA = self._DIR_DATA_DOWN_SAMPLED = r'../mine/'
        self._DIR_DATA_INPUT = r'../mine/'
        self._DIR_DATA_OUTPUT = r'../mine/downsampled' + str(self.TOTAL_SAMPLING_PCNT) + '_' + str(self.TEST_PCNT) + '/'
        if not os.path.exists(self._DIR_DATA_OUTPUT):
            os.makedirs(self._DIR_DATA_OUTPUT)
        # self._DIR_DATA_INPUT = r'/kaggle/input/amex-default-prediction/'
        # self._DIR_DATA = r'/kaggle/working/'

        # self._DIR_DATA_DOWN_SAMPLED = r'/kaggle/input/down-sampled-data/'
        self._FILE_TRAIN = 'train_data.csv'
        self._FILE_TEST = 'test_data.csv'
        self._FILE_TRAIN_LABEL = 'train_labels.csv'
        self._FILE_TRAIN_FEATHER_ORIG = 'train_orig.feather'
        self._FILE_TEST_FEATHER_ORIG = 'test_orig.feather'
        self._DIR_OUTPUT = './output/'

        # list columns that are not numerical, we will need to encode these.
        self.NON_NUM_COLS = ['D_63', 'D_64']
        self.cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66",
                             "D_68"]

        self.COL_UID = "customer_ID"
        self.COL_TIME = 'S_2'
        # self.COL_LABEL = 'target'

        self.seed = 42
        self.n_folds = 5
        self.target = 'target'
        self.boosting_type = 'dart'
        self.metric = 'binary_logloss'

    # ====================================================
    # Get the difference
    # ====================================================
    def get_difference(self, data, num_features):
        df1 = []
        customer_ids = []
        for customer_id, df in tqdm(data.groupby(['customer_ID'])):
            # Get the differences
            diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
            # Append to lists
            df1.append(diff_df1)
            customer_ids.append(customer_id)
        # Concatenate
        df1 = np.concatenate(df1, axis = 0)
        # Transform to dataframe
        df1 = pd.DataFrame(df1, columns = [col + '_diff1' for col in df[num_features].columns])
        # Add customer id
        df1['customer_ID'] = customer_ids
        return df1
    def get_path_train(self):
        if self.DOWN_SAMPLING:
            return self._DIR_DATA_DOWN_SAMPLED + '/' + 'train_down_sampling_' + str(self.TOTAL_SAMPLING_PCNT) + '_' + str(self.TEST_PCNT) + '.feather'
        else:
            return self._DIR_DATA + '/' + self._FILE_TRAIN_FEATHER_ORIG

    def get_path_test(self):
        if self.USE_TEST_DATA:
            return self._DIR_DATA + '/' + self._FILE_TEST_FEATHER_ORIG
        else:
            return self._DIR_DATA_DOWN_SAMPLED + '/' + 'test_down_sampling_from_train_' + str(
                self.TOTAL_SAMPLING_PCNT) + '_' + str(self.TEST_PCNT) + '.feather'

    # ====================================================
# Read & preprocess data and save it to disk
# ====================================================
    def read_preprocess_data(self):
        train = pd.read_feather(self.get_path_train())
        features = train.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()
        cat_features = [
            "B_30",
            "B_38",
            "D_114",
            "D_116",
            "D_117",
            "D_120",
            "D_126",
            "D_63",
            "D_64",
            "D_66",
            "D_68",
        ]
        num_features = [col for col in features if col not in cat_features]
        print('Starting training feature engineer...')
        train_num_agg = train.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
        train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
        train_num_agg.reset_index(inplace = True)
        train_cat_agg = train.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
        train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
        train_cat_agg.reset_index(inplace = True)
        train_labels = pd.read_csv(self._DIR_DATA_INPUT + self._FILE_TRAIN_LABEL)
        # Transform float64 columns to float32
        cols = list(train_num_agg.dtypes[train_num_agg.dtypes == 'float64'].index)
        for col in tqdm(cols):
            train_num_agg[col] = train_num_agg[col].astype(np.float32)
        # Transform int64 columns to int32
        cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == 'int64'].index)
        for col in tqdm(cols):
            train_cat_agg[col] = train_cat_agg[col].astype(np.int32)
        # Get the difference
        train_diff = self.get_difference(train, num_features)
        train = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID').merge(train_diff, how = 'inner', on = 'customer_ID').merge(train_labels, how = 'inner', on = 'customer_ID')
        del train_num_agg, train_cat_agg, train_diff
        gc.collect()
        test = pd.read_feather(self.get_path_test())
        print('Starting test feature engineer...')
        test_num_agg = test.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
        test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
        test_num_agg.reset_index(inplace = True)
        test_cat_agg = test.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
        test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
        test_cat_agg.reset_index(inplace = True)
        # Transform float64 columns to float32
        cols = list(test_num_agg.dtypes[test_num_agg.dtypes == 'float64'].index)
        for col in tqdm(cols):
            test_num_agg[col] = test_num_agg[col].astype(np.float32)
        # Transform int64 columns to int32
        cols = list(test_cat_agg.dtypes[test_cat_agg.dtypes == 'int64'].index)
        for col in tqdm(cols):
            test_cat_agg[col] = test_cat_agg[col].astype(np.int32)
        # Get the difference
        test_diff = self.get_difference(test, num_features)
        test = test_num_agg.merge(test_cat_agg, how = 'inner', on = 'customer_ID').merge(test_diff, how = 'inner', on = 'customer_ID')
        del test_num_agg, test_cat_agg, test_diff
        gc.collect()
        # Save files to disk
        train.to_parquet(self._DIR_DATA_OUTPUT + 'train_fe.parquet')
        test.to_parquet(self._DIR_DATA_OUTPUT + 'test_fe.parquet')



# # Training & Inference

# In[ ]:



# ====================================================
# Configurations
# ====================================================
    if False:
        class CFG:
            input_dir = '/content/data/'
            seed = 42
            n_folds = 5
            target = 'target'
            boosting_type = 'dart'
            metric = 'binary_logloss'

# ====================================================
# Seed everything
# ====================================================
    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

# ====================================================
# Read data
# ====================================================
    def read_data(self):
        train = pd.read_parquet(self._DIR_DATA_OUTPUT + 'train_fe.parquet')
        test = pd.read_parquet(self._DIR_DATA_OUTPUT + 'test_fe.parquet')
        return train, test

# ====================================================
# Amex metric
# ====================================================


# ====================================================
# LGBM amex metric
# ====================================================

# ====================================================
# Train & Evaluate
# ====================================================
    def train_and_evaluate(self, train, test):
        # Label encode categorical features
        cat_features = [
            "B_30",
            "B_38",
            "D_114",
            "D_116",
            "D_117",
            "D_120",
            "D_126",
            "D_63",
            "D_64",
            "D_66",
            "D_68"
        ]
        cat_features = [f"{cf}_last" for cf in cat_features]
        for cat_col in cat_features:
            encoder = LabelEncoder()
            train[cat_col] = encoder.fit_transform(train[cat_col])
            test[cat_col] = encoder.transform(test[cat_col])
        # Round last float features to 2 decimal place
        num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
        num_cols = [col for col in num_cols if 'last' in col]
        for col in num_cols:
            train[col + '_round2'] = train[col].round(2)
            test[col + '_round2'] = test[col].round(2)
        # Get the difference between last and mean
        num_cols = [col for col in train.columns if 'last' in col]
        num_cols = [col[:-5] for col in num_cols if 'round' not in col]
        for col in num_cols:
            try:
                train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
                test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
            except:
                pass
        # Transform float64 and float32 to float16
        num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
        for col in tqdm(num_cols):
            train[col] = train[col].astype(np.float16)
            test[col] = test[col].astype(np.float16)
        # Get feature list
        features = [col for col in train.columns if col not in ['customer_ID', self.target]]
        params = {
            'objective': 'binary',
            'metric': self.metric,
            'boosting': self.boosting_type,
            'seed': self.seed,
            'num_leaves': 100,
            'learning_rate': 0.01,
            'feature_fraction': 0.20,
            'bagging_freq': 10,
            'bagging_fraction': 0.50,
            'n_jobs': -1,
            'lambda_l2': 2,
            'min_data_in_leaf': 40,
            }
        # Create a numpy array to store test predictions
        test_predictions = np.zeros(len(test))
        # Create a numpy array to store out of folds predictions
        oof_predictions = np.zeros(len(train))
        kfold = StratifiedKFold(n_splits = self.n_folds, shuffle = True, random_state = self.seed)
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[self.target])):
            print(' ')
            print('-'*50)
            print(f'Training fold {fold} with {len(features)} features...')
            x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
            y_train, y_val = train[self.target].iloc[trn_ind], train[self.target].iloc[val_ind]
            lgb_train = lgb.Dataset(x_train, y_train, categorical_feature = cat_features)
            lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature = cat_features)
            model = lgb.train(
                params = params,
                train_set = lgb_train,
                num_boost_round = 10500,
                valid_sets = [lgb_train, lgb_valid],
                early_stopping_rounds = 1500,
                verbose_eval = 500,
                feval = lgb_amex_metric
                )
            # Save best model
            joblib.dump(model, self._DIR_DATA_OUTPUT + f'/lgbm_{self.boosting_type}_fold{fold}_seed{self.seed}.pkl')
            # Predict validation
            val_pred = model.predict(x_val)
            # Add to out of folds array
            oof_predictions[val_ind] = val_pred
            # Predict the test set
            test_pred = model.predict(test[features])
            test_predictions += test_pred / self.n_folds
            # Compute fold metric
            score = amex_metric(y_val, val_pred)
            print(f'Our fold {fold} CV score is {score}')
            del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
            gc.collect()
        # Compute out of folds metric
        score = amex_metric(train[self.target], oof_predictions)
        print(f'Our out of folds CV score is {score}')
        # Create a dataframe to store out of folds predictions
        oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[self.target], 'prediction': oof_predictions})
        oof_df.to_csv(self._DIR_DATA_OUTPUT + f'/oof_lgbm_{self.boosting_type}_baseline_{self.n_folds}fold_seed{self.seed}.csv', index = False)
        # Create a dataframe to store test prediction
        test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
        test_df.to_csv(self._DIR_DATA_OUTPUT + f'/test_lgbm_{self.boosting_type}_baseline_{self.n_folds}fold_seed{self.seed}.csv', index = False)

    def compute_score(self):
        train_y_orig = pd.read_csv(self._DIR_DATA_INPUT + self._FILE_TRAIN_LABEL)
        # merged = False
        for run_id in [self._DIR_DATA_OUTPUT + f'/test_lgbm_{self.boosting_type}_baseline_{self.n_folds}fold_seed{self.seed}.csv']:
            # LGB_with_series_feature
            submission = pd.read_csv(run_id)
            submission = submission.groupby(self.COL_UID).mean()['prediction'].reset_index()
            train_y = train_y_orig.merge(submission, on=self.COL_UID)
            score = amex_metric(train_y[self.target], train_y['prediction'])
            print(run_id + ' score: ' + str(score))


# # Read Submission File
# This is the submission file corresponding to the output of the previous pipeline (using the average blend of 3 seeds)

# In[ ]:



if __name__ == '__main__':

    model = model_amex()
    if True:
        # Read & Preprocess Data
        model.read_preprocess_data()

        model.seed_everything(model.seed)
        train, test =  model.read_data()
        model.train_and_evaluate(train, test)
    if False:
        sub = pd.read_csv(model._DIR_DATA_OUTPUT + 'test_lgbm_baseline_5fold_seed_blend.csv')
        sub.to_csv(model._DIR_DATA_OUTPUT + 'test_lgbm_baseline_5fold_seed_blend.csv', index=False)
    if True:
        model.compute_score()
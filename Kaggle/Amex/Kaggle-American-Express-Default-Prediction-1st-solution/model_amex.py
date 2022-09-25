from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
import time
import argparse
from sklearn.model_selection import  StratifiedKFold
import lightgbm as lgb
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import torch.nn as nn
import torch
import argparse
from os import remove
from shutil import move

from utils import TaskDataset
from scheduler import Adam12
from model import Amodel

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
        self.TOTAL_SAMPLING_PCNT = 0.02
        self.TEST_PCNT = 0.1

        # folder where data are located such as train and test data
        self._DIR_DATA = self._DIR_DATA_DOWN_SAMPLED = r'../mine/'
        self._DIR_DATA_INPUT = r'../mine/'
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
        self.COL_LABEL = 'target'

        self.seed = 42
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=str, default=self._DIR_DATA_INPUT + '')
        parser.add_argument("--save_dir", type=str, default='tmp')
        parser.add_argument("--use_apm", action='store_true', default=False)
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--do_train", action='store_true', default=False)
        parser.add_argument("--test", action='store_true', default=False)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--remark", type=str, default='')

        self.args, unknown = parser.parse_known_args()

    def get_path_train_orig(self):
        return self._DIR_DATA_INPUT + '/' + self._FILE_TRAIN

    def get_path_train(self):
        if self.DOWN_SAMPLING:
            return self._DIR_DATA_DOWN_SAMPLED + '/' + 'train_down_sampling_' + str(self.TOTAL_SAMPLING_PCNT) + '_' + str(self.TEST_PCNT) + '.feather'
        else:
            return self._DIR_DATA + '/' + self._FILE_TRAIN_FEATHER_ORIG

        # not used

    def get_path_test(self):
        if self.USE_TEST_DATA:
            return self._DIR_DATA + '/' + self._FILE_TEST_FEATHER_ORIG
        else:
            return self._DIR_DATA_DOWN_SAMPLED + '/' + 'test_down_sampling_from_train_' + str(self.TOTAL_SAMPLING_PCNT) + '_' + str(self.TEST_PCNT) + '.feather'

    if False:
        # not used

        # not used
        def build_label_encoder(self):
            label_encoder_model = dict()
            # manually check to see which file (test or train) contains the full set of values. in amex this is train
            df = pd.read_csv(self.get_path_train_orig(), usecols=self.NON_NUM_COLS)
            print('finished reading the NON_NUM_COLS from train_data')

            for c in self.NON_NUM_COLS:
                label_encoder_model[c] = LabelEncoder().fit(df[c])
                np.save(self._DIR_DATA + 'label_encoder_model_' + c + '.npy', label_encoder_model[c].classes_)

        # not used
        def get_label_encoder(self):
            label_encoder_model = dict()
            if not os.path.exists(self._DIR_DATA + 'label_encoder_model_' + self.NON_NUM_COLS[0] + '.npy'):
                self.build_label_encoder()
            np_load_old = np.load
            # modify the default parameters of np.load
            np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

            for c in self.NON_NUM_COLS:
                _file_path = self._DIR_DATA + 'label_encoder_model_' + c + '.npy'
                label_encoder_model[c] = LabelEncoder()

                label_encoder_model[c].classes_ = np.load(_file_path)
            # restore np.load for future normal usage
            np.load = np_load_old
            return label_encoder_model

    def S1_denoise(self):
        def denoise(df):
            df['D_63'] = df['D_63'].apply(lambda t: {'CR': 0, 'XZ': 1, 'XM': 2, 'CO': 3, 'CL': 4, 'XL': 5}[t]).astype(
                np.int8)
            df['D_64'] = df['D_64'].apply(lambda t: {np.nan: -1, 'O': 0, '-1': 1, 'R': 2, 'U': 3}[t]).astype(np.int8)
            for col in tqdm(df.columns):
                if col not in [self.COL_UID, self.COL_TIME] + self.NON_NUM_COLS:
                    df[col] = np.floor(df[col] * 100)
            return df

        # chunksize = 10 ** 6

        files = [self.get_path_train_orig(), self._DIR_DATA_INPUT + self._FILE_TEST]
        files_out = [self._DIR_DATA + self._FILE_TRAIN_FEATHER_ORIG, self._DIR_DATA + self._FILE_TEST_FEATHER_ORIG]
        for file_in, file_out in zip(files, files_out):
            # FIRST_TIME = True
            # for chunk in pd.read_csv(file_in, chunksize=chunksize):
            chunk = pd.read_csv(file_in)
            # train = pd.read_csv(self.get_path_train_orig())
            chunk = denoise(chunk)
            chunk.to_feather(file_out)
            #     if FIRST_TIME:
            #         chunk.to_csv(file_out, index=False)
            #         FIRST_TIME = False
            #     else:
            #         chunk.to_csv(file_out, index=False, mode='a')

        # del train
        # this fails the 2nd time, will need to process by batch
        # test = pd.read_csv(self._DIR_DATA + self._FILE_TEST)
        # test = denoise(test)
        # test.to_feather()

    def S1pnt1_down_sample(self):
        if self.DOWN_SAMPLING:
            df_orig = pd.read_feather(self._DIR_DATA + self._FILE_TRAIN_FEATHER_ORIG)
            customer_ID_samples = df_orig.groupby(self.COL_UID).count()[self.COL_TIME].to_frame()
            _len = len(customer_ID_samples)
            # rescale self.SIZE_TRAIN + self.SIZE_TEST
            # if _len < self.SIZE_TRAIN + self.SIZE_TEST:
            # self.SIZE_TRAIN, self.SIZE_TEST = int(self.SIZE_TRAIN * _len / (self.SIZE_TRAIN + self.SIZE_TEST)), int(self.SIZE_TEST * _len / (self.SIZE_TRAIN + self.SIZE_TEST))
            customer_ID_samples = customer_ID_samples.sample(
                frac=self.TOTAL_SAMPLING_PCNT)
            msk = np.random.rand(len(customer_ID_samples)) < (1 - self.TEST_PCNT)
            customer_ID_samples_train = customer_ID_samples[msk]
            customer_ID_samples_test = customer_ID_samples[~msk]
            train = df_orig.merge(customer_ID_samples_train, left_on=self.COL_UID, right_index=True,
                                  suffixes=('', '_y')).drop(
                self.COL_TIME + '_y', axis=1)

            train.reset_index().to_feather(self.get_path_train())

            test = df_orig.merge(customer_ID_samples_test, left_on=self.COL_UID, right_index=True,
                                 suffixes=('', '_y')).drop(
                self.COL_TIME + '_y', axis=1)

            test.reset_index().to_feather(self.get_path_test())

    def cat_feature(self, df, lastk):
        one_hot_features = [col for col in df.columns if 'oneHot' in col]
        if lastk is None:
            num_agg_df = df.groupby(self.COL_UID, sort=False)[one_hot_features].agg(['mean', 'std', 'sum', 'last'])
        else:
            num_agg_df = df.groupby(self.COL_UID, sort=False)[one_hot_features].agg(['mean', 'std', 'sum'])
        num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]

        if lastk is None:
            cat_agg_df = df.groupby(self.COL_UID, sort=False)[self.cat_features].agg(['last', 'nunique'])
        else:
            cat_agg_df = df.groupby(self.COL_UID, sort=False)[self.cat_features].agg(['nunique'])
        cat_agg_df.columns = ['_'.join(x) for x in cat_agg_df.columns]

        count_agg_df = df.groupby(self.COL_UID, sort=False)[[self.COL_TIME]].agg(['count'])
        count_agg_df.columns = ['_'.join(x) for x in count_agg_df.columns]
        df = pd.concat([num_agg_df, cat_agg_df, count_agg_df], axis=1).reset_index()
        print('cat feature shape after engineering', df.shape)

        return df

    def one_hot_encoding(self, df, cols, is_drop=True):
        for col in cols:
            print('one hot encoding:', col)
            dummies = pd.get_dummies(pd.Series(df[col]), prefix='oneHot_%s' % col)
            df = pd.concat([df, dummies], axis=1)
        if is_drop:
            df.drop(cols, axis=1, inplace=True)
        return df

    def num_feature(self, df, num_features, lastk):
        if num_features[0][:5] == 'rank_':
            num_agg_df = df.groupby(self.COL_UID, sort=False)[num_features].agg(['last'])
        else:
            if lastk is None:
                num_agg_df = df.groupby(self.COL_UID, sort=False)[num_features].agg(
                    ['mean', 'std', 'min', 'max', 'sum', 'last'])
            else:
                num_agg_df = df.groupby(self.COL_UID, sort=False)[num_features].agg(
                    ['mean', 'std', 'min', 'max', 'sum'])
        num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]
        if num_features[0][:5] != 'rank_':
            for col in num_agg_df.columns:
                num_agg_df[col] = num_agg_df[col] // 0.01
        df = num_agg_df.reset_index()
        print('num feature shape after engineering', df.shape)

        return df

    def diff_feature(self, df, num_features, lastk):
        diff_num_features = [f'diff_{col}' for col in num_features]
        cids = df[self.COL_UID].values
        df = df.groupby(self.COL_UID)[num_features].diff().add_prefix('diff_')
        df.insert(0, self.COL_UID, cids)
        if lastk is None:
            num_agg_df = df.groupby(self.COL_UID, sort=False)[diff_num_features].agg(
                ['mean', 'std', 'min', 'max', 'sum', 'last'])
        else:
            num_agg_df = df.groupby(self.COL_UID, sort=False)[diff_num_features].agg(
                ['mean', 'std', 'min', 'max', 'sum'])
        num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]
        for col in num_agg_df.columns:
            num_agg_df[col] = num_agg_df[col] // 0.01

        df = num_agg_df.reset_index()
        print('diff feature shape after engineering', df.shape)

        #diff_D_108_std at 2% sampling all NaN, leading to NN failing. just fill with 0 if all NaN
        df[df.columns[df.isnull().sum() == len(df)]]=0

        return df

    def S2_manial_feature(self):
        n_cpu = 16
        transform = [['', 'rank_', 'ym_rank_'], [''], ['']]
        if False:
            len(df_train)
        for li, lastk in enumerate([None, 3, 6]):
            for prefix in transform[li]:

                # df = df_orig.copy()

                df = pd.read_feather(self.get_path_train()).append(
                    pd.read_feather(self.get_path_test())).reset_index(drop=True)
                df[self.COL_TIME] = pd.to_datetime(df[self.COL_TIME])
                all_cols = [c for c in list(df.columns) if c not in [self.COL_UID, self.COL_TIME]]
                num_features = [col for col in all_cols if col not in self.cat_features]

                # special coded for amex prefix
                for col in [col for col in df.columns if 'S_' in col or 'P_' in col]:
                    if col != self.COL_TIME:
                        df[col] = df[col].fillna(0)

                if lastk is not None:
                    prefix = f'last{lastk}_' + prefix
                    print('all df shape', df.shape)
                    df['rank'] = df.groupby(self.COL_UID)[self.COL_TIME].rank(ascending=False)
                    df = df.loc[df['rank'] <= lastk].reset_index(drop=True)
                    df = df.drop(['rank'], axis=1)
                    print(f'last {lastk} shape', df.shape)

                if prefix == 'rank_':
                    cids = df[self.COL_UID].values
                    df = df.groupby(self.COL_UID)[num_features].rank(pct=True).add_prefix('rank_')
                    df.insert(0, self.COL_UID, cids)
                    num_features = [f'rank_{col}' for col in num_features]

                if prefix == 'ym_rank_':
                    cids = df[self.COL_UID].values
                    old = df[self.COL_TIME]
                    df[self.COL_TIME] = df[self.COL_TIME].map(str)
                    df['ym'] = df[self.COL_TIME].apply(lambda x: x[:7])
                    df[self.COL_TIME] = old
                    df = df.groupby('ym')[num_features].rank(pct=True).add_prefix('ym_rank_')
                    num_features = [f'ym_rank_{col}' for col in num_features]
                    df.insert(0, self.COL_UID, cids)

                if prefix in ['', 'last3_']:
                    df = self.one_hot_encoding(df, self.cat_features, False)

                vc = df[self.COL_UID].value_counts(sort=False).cumsum()
                batch_size = int(np.ceil(len(vc) / n_cpu))
                dfs = []
                start = 0
                for i in range(min(n_cpu, int(np.ceil(len(vc) / batch_size)))):
                    vc_ = vc[i * batch_size:(i + 1) * batch_size]
                    dfs.append(df[start:vc_[-1]])
                    start = vc_[-1]

                # for debugging turn of parallel
                if False:
                    pool = ThreadPool(n_cpu)

                if prefix in ['', 'last3_']:
                    if False:
                        cat_feature_df = pd.concat(pool.map(cat_feature, tqdm(dfs, desc='cat_feature'))).reset_index(
                            drop=True)
                    cat_feature_df = self.cat_feature(df, lastk).reset_index(drop=True)

                    cat_feature_df.to_feather(self._DIR_DATA + f'{prefix}cat_feature.feather')

                if prefix in ['', 'last3_', 'last6_', 'rank_', 'ym_rank_']:
                    if False:
                        num_feature_df = pd.concat(pool.map(num_feature, tqdm(dfs, desc='num_feature'))).reset_index(
                            drop=True)
                    num_feature_df = self.num_feature(df, num_features, lastk).reset_index(drop=True)
                    num_feature_df.to_feather(self._DIR_DATA + f'{prefix}num_feature.feather')

                if prefix in ['', 'last3_']:
                    if False:
                        diff_feature_df = pd.concat(pool.map(diff_feature, tqdm(dfs, desc='diff_feature'))).reset_index(
                            drop=True)
                    diff_feature_df = self.diff_feature(df, num_features, lastk).reset_index(drop=True)
                    diff_feature_df.to_feather(self._DIR_DATA + f'{prefix}diff_feature.feather')

                # for debugging turn of parallel
                if False:
                    pool.close()

    def Write_log(self, logFile, text, isPrint=True):
        if isPrint:
            print(text)
        logFile.write(text)
        logFile.write('\n')
        return None

    def amex_metric_mod(self, y_true, y_pred):
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

    def Metric(self, labels, preds):
        return self.amex_metric_mod(labels, preds)

    def Lgb_train_and_predict(self, train, test, config, gkf=False, aug=None, output_root=None, run_id=None):
        if output_root is None:
            output_root = self._DIR_OUTPUT
        if not run_id:
            run_id = 'run_lgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            while os.path.exists(output_root + run_id + '/'):
                time.sleep(1)
                run_id = 'run_lgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_root + f'{self.args.save_dir}/'
        else:
            output_path = output_root + run_id + '/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        os.system(f'copy ./*.py {output_path}'.replace('/', "\\"))
        os.system(f'copy ./*.sh {output_path}'.replace('/', "\\"))
        config['lgb_params']['seed'] = config['seed']
        oof, sub = None, None
        if train is not None:
            log = open(output_path + '/train.log', 'w', buffering=1)
            log.write(str(config) + '\n')
            features = config['feature_name']
            params = config['lgb_params']
            rounds = config['rounds']
            verbose = config['verbose_eval']
            early_stopping_rounds = config['early_stopping_rounds']
            folds = config['folds']
            seed = config['seed']
            oof = train[[self.COL_UID]]
            oof[self.COL_LABEL] = 0

            all_valid_metric, feature_importance = [], []
            if gkf:
                tmp = train[[self.COL_UID, self.COL_LABEL]].drop_duplicates(self.COL_UID).reset_index(drop=True)
                skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                split = skf.split(tmp, tmp[self.COL_LABEL])
                new_split = []
                for trn_index, val_index in split:
                    trn_uids = tmp.loc[trn_index, self.COL_UID].values
                    val_uids = tmp.loc[val_index, self.COL_UID].values
                    new_split.append((train.loc[train[self.COL_UID].isin(trn_uids)].index,
                                      train.loc[train[self.COL_UID].isin(val_uids)].index))
                split = new_split

                # skf = GroupKFold(n_splits=folds)
                # split = skf.split(train,train[self.COL_LABEL],train[self.COL_UID])
            else:
                skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                split = skf.split(train, train[self.COL_LABEL])
            for fold, (trn_index, val_index) in enumerate(split):
                evals_result_dic = {}
                train_cids = train.loc[trn_index, self.COL_UID].values
                if aug:
                    train_aug = aug.loc[aug[self.COL_UID].isin(train_cids)]
                    trn_data = lgb.Dataset(train.loc[trn_index, features].append(train_aug[features]),
                                           label=train.loc[trn_index, self.COL_LABEL].append(train_aug[self.COL_LABEL]))
                else:
                    trn_data = lgb.Dataset(train.loc[trn_index, features], label=train.loc[trn_index, self.COL_LABEL])

                val_data = lgb.Dataset(train.loc[val_index, features], label=train.loc[val_index, self.COL_LABEL])
                model = lgb.train(params,
                                  train_set=trn_data,
                                  num_boost_round=rounds,
                                  valid_sets=[trn_data, val_data],
                                  evals_result=evals_result_dic,
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose_eval=verbose
                                  )
                model.save_model(output_path + '/fold%s.ckpt' % fold)

                valid_preds = model.predict(train.loc[val_index, features], num_iteration=model.best_iteration)
                oof.loc[val_index, self.COL_LABEL] = valid_preds

                for i in range(len(evals_result_dic['valid_1'][params['metric']]) // verbose):
                    self.Write_log(log, ' - %i round - train_metric: %.6f - valid_metric: %.6f\n' % (
                        i * verbose, evals_result_dic['training'][params['metric']][i * verbose],
                        evals_result_dic['valid_1'][params['metric']][i * verbose]))
                all_valid_metric.append(self.Metric(train.loc[val_index, self.COL_LABEL], valid_preds))
                self.Write_log(log, '- fold%s valid metric: %.6f\n' % (fold, all_valid_metric[-1]))

                importance_gain = model.feature_importance(importance_type='gain')
                importance_split = model.feature_importance(importance_type='split')
                feature_name = model.feature_name()
                feature_importance.append(pd.DataFrame(
                    {'feature_name': feature_name, 'importance_gain': importance_gain,
                     'importance_split': importance_split}))

            feature_importance_df = pd.concat(feature_importance)
            feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
            feature_importance_df = feature_importance_df.sort_values(by=['importance_gain'], ascending=False)
            feature_importance_df.to_csv(output_path + '/feature_importance.csv', index=False)

            mean_valid_metric = np.mean(all_valid_metric)
            global_valid_metric = self.Metric(train[self.COL_LABEL].values, oof[self.COL_LABEL].values)
            self.Write_log(log,
                           'all valid mean metric:%.6f, global valid metric:%.6f' % (
                               mean_valid_metric, global_valid_metric))

            oof.to_csv(output_path + '/oof.csv', index=False)

            log.close()
            if os.path.exists(output_path + '/train_%.6f.log' % mean_valid_metric):
                remove(output_path + '/train_%.6f.log' % mean_valid_metric)
            move(output_path + '/train.log', output_path + '/train_%.6f.log' % mean_valid_metric)

            log_df = pd.DataFrame({'run_id': [run_id], 'mean metric': [round(mean_valid_metric, 6)],
                                   'global metric': [round(global_valid_metric, 6)], 'remark': [self.args.remark]})
            if not os.path.exists(output_root + '/experiment_log.csv'):
                log_df.to_csv(output_root + '/experiment_log.csv', index=False)
            else:
                log_df.to_csv(output_root + '/experiment_log.csv', index=False, header=None, mode='a')

        if test is not None:
            sub = test[[self.COL_UID]]
            sub['prediction'] = 0
            for fold in range(folds):
                model = lgb.Booster(model_file=output_path + '/fold%s.ckpt' % fold)
                test_preds = model.predict(test[features], num_iteration=model.best_iteration)
                sub['prediction'] += (test_preds / folds)
            sub[[self.COL_UID, 'prediction']].to_csv(output_path + '/submission.csv.zip', compression='zip',
                                                     index=False)
        if self.args.save_dir in output_path:
            os.rename(output_path, output_root + run_id + '/')
        return oof, sub, (mean_valid_metric, global_valid_metric)

    def S3_series_feature(self):
        train = pd.read_feather(self.get_path_train())

        test = pd.read_feather(self.get_path_test())

        eps = 1e-3

        train_y = pd.read_csv(self._DIR_DATA_INPUT + self._FILE_TRAIN_LABEL)
        train = train.merge(train_y, how='left', on=self.COL_UID)

        print(train.shape, test.shape)

        lgb_config = {
            'lgb_params': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting': 'dart',
                'max_depth': -1,
                'num_leaves': 64,
                'learning_rate': 0.035,
                'bagging_freq': 5,
                'bagging_fraction': 0.7,
                'feature_fraction': 0.7,
                'min_data_in_leaf': 256,
                'max_bin': 63,
                'min_data_in_bin': 256,
                # 'min_sum_heassian_in_leaf': 10,
                'tree_learner': 'serial',
                'boost_from_average': 'false',
                'lambda_l1': 0.1,
                'lambda_l2': 30,
                'num_threads': 24,
                'verbosity': 1,
            },
            'feature_name': [col for col in train.columns if col not in [self.COL_UID, self.COL_LABEL, self.COL_TIME]],
            'rounds': 4500,
            'early_stopping_rounds': 100,
            'verbose_eval': 50,
            'folds': 5,
            'seed': self.seed
        }

        self.Lgb_train_and_predict(train, test, lgb_config, gkf=True, aug=None, run_id='LGB_with_series_feature')

    def compute_score(self):
        train_y_orig = pd.read_csv(self._DIR_DATA + self._FILE_TRAIN_LABEL)
        # merged = False
        for run_id in ['LGB_with_series_feature', 'LGB_with_manual_feature', 'LGB_with_manual_feature_and_series_oof',
                       'NN_with_series', 'NN_with_series_and_all_feature','']:
            # LGB_with_series_feature
            if len(run_id) > 0:
                _dir = self._DIR_OUTPUT + run_id + '/'
                submission = pd.read_csv(_dir + 'submission.csv.zip')
            else:
                _dir = self._DIR_OUTPUT
                submission = pd.read_csv(_dir + 'final_submission.csv.zip')
            submission.to_csv(_dir + 'submission.csv', index=False)
            submission = submission.groupby(self.COL_UID).mean()['prediction'].reset_index()
            train_y = train_y_orig.merge(submission, on=self.COL_UID)
            score = self.amex_metric_mod(train_y[self.COL_LABEL], train_y['prediction'])
            print(run_id + ' score: ' + str(score))

    def GreedyFindBin(self, distinct_values, counts, num_distinct_values, max_bin, total_cnt, min_data_in_bin=3):
        # INPUT:
        #   distinct_values 保存特征取值的数组，特征取值单调递增
        #   counts 特征的取值对应的样本数目
        #   num_distinct_values 特征取值的数量
        #   max_bin 分桶的最大数量
        #   total_cnt 样本数量
        #   min_data_in_bin 桶包含的最小样本数

        # bin_upper_bound就是记录桶分界的数组
        bin_upper_bound = list();
        assert (max_bin > 0)

        # 特征取值数比max_bin数量少，直接取distinct_values的中点放置
        if num_distinct_values <= max_bin:
            cur_cnt_inbin = 0
            for i in range(num_distinct_values - 1):
                cur_cnt_inbin += counts[i]
                # 若一个特征的取值比min_data_in_bin小，则累积下一个取值，直到比min_data_in_bin大，进入循环。
                if cur_cnt_inbin >= min_data_in_bin:
                    # 取当前值和下一个值的均值作为该桶的分界点bin_upper_bound
                    bin_upper_bound.append((distinct_values[i] + distinct_values[i + 1]) / 2.0)
                    cur_cnt_inbin = 0
            # 对于最后一个桶的上界则为无穷大
            cur_cnt_inbin += counts[num_distinct_values - 1];
            bin_upper_bound.append(float('Inf'))
            # 特征取值数比max_bin来得大，说明几个特征值要共用一个bin
        else:
            if min_data_in_bin > 0:
                max_bin = min(max_bin, total_cnt // min_data_in_bin)
                max_bin = max(max_bin, 1)
            # mean size for one bin
            mean_bin_size = total_cnt / max_bin
            rest_bin_cnt = max_bin
            rest_sample_cnt = total_cnt
            # 定义is_big_count_value数组：初始设定特征每一个不同的值的数量都小（false）
            is_big_count_value = [False] * num_distinct_values
            # 如果一个特征值的数目比mean_bin_size大，那么这些特征需要单独一个bin
            for i in range(num_distinct_values):
                # 如果一个特征值的数目比mean_bin_size大，则设定这个特征值对应的is_big_count_value为真。。
                if counts[i] >= mean_bin_size:
                    is_big_count_value[i] = True
                    rest_bin_cnt -= 1
                    rest_sample_cnt -= counts[i]
            # 剩下的特征取值的样本数平均每个剩下的bin：mean size for one bin
            mean_bin_size = rest_sample_cnt / rest_bin_cnt
            upper_bounds = [float('Inf')] * max_bin
            lower_bounds = [float('Inf')] * max_bin

            bin_cnt = 0
            lower_bounds[bin_cnt] = distinct_values[0]
            cur_cnt_inbin = 0
            # 重新遍历所有的特征值（包括数目大和数目小的）
            for i in range(num_distinct_values - 1):
                # 如果当前的特征值数目是小的
                if not is_big_count_value[i]:
                    rest_sample_cnt -= counts[i]
                cur_cnt_inbin += counts[i]

                # 若cur_cnt_inbin太少，则累积下一个取值，直到满足条件，进入循环。
                # need a new bin 当前的特征如果是需要单独成一个bin，或者当前几个特征计数超过了mean_bin_size，或者下一个是需要独立成桶的
                if is_big_count_value[i] or cur_cnt_inbin >= mean_bin_size or \
                        is_big_count_value[i + 1] and cur_cnt_inbin >= max(1.0, mean_bin_size * 0.5):
                    upper_bounds[bin_cnt] = distinct_values[i]  # 第i个bin的最大就是 distinct_values[i]了
                    bin_cnt += 1
                    lower_bounds[bin_cnt] = distinct_values[i + 1]  # 下一个bin的最小就是distinct_values[i + 1]，注意先++bin了
                    if bin_cnt >= max_bin - 1:
                        break
                    cur_cnt_inbin = 0
                    if not is_big_count_value[i]:
                        rest_bin_cnt -= 1
                        mean_bin_size = rest_sample_cnt / rest_bin_cnt
            #             bin_cnt+=1
            # update bin upper bound 与特征取值数比max_bin数量少的操作类似，取当前值和下一个值的均值作为该桶的分界点
            for i in range(bin_cnt - 1):
                bin_upper_bound.append((upper_bounds[i] + lower_bounds[i + 1]) / 2.0)
            bin_upper_bound.append(float('Inf'))
        return bin_upper_bound

    def S4_feature_combined(self):
        def pad_target(x):
            t = np.zeros(13)
            t[:-len(x)] = np.nan
            t[-len(x):] = x
            return list(t)

        root = self.args.root

        oof = pd.read_csv('./output/LGB_with_series_feature/oof.csv')
        sub = pd.read_csv('./output/LGB_with_series_feature/submission.csv.zip')

        tmp1 = oof.groupby(self.COL_UID, sort=False)['target'].agg(lambda x: pad_target(x))
        tmp2 = sub.groupby(self.COL_UID, sort=False)['prediction'].agg(lambda x: pad_target(x))

        tmp = tmp1.append(tmp2)

        tmp = pd.DataFrame(data=tmp.tolist(), columns=['target%s' % i for i in range(1, 14)])

        df = []
        for fn in ['cat', 'num', 'diff', 'rank_num', 'last3_cat', 'last3_num', 'last3_diff', 'last6_num',
                   'ym_rank_num']:
            if len(df) == 0:
                df.append(pd.read_feather(f'{self._DIR_DATA}/{fn}_feature.feather'))
            else:
                df.append(pd.read_feather(f'{self._DIR_DATA}/{fn}_feature.feather').drop([self.COL_UID], axis=1))
            if 'last' in fn:
                df[-1] = df[-1].add_prefix('_'.join(fn.split('_')[:-1]) + '_')

        df.append(tmp)

        df = pd.concat(df, axis=1)
        print(df.shape)
        df.to_feather(f'{self._DIR_DATA}/all_feature.feather')

        del df

        df = pd.read_feather(self.get_path_train()).append(
            pd.read_feather(self.get_path_test())).reset_index(drop=True)
        df = df.drop(['S_2'], axis=1)
        df = self.one_hot_encoding(df, self.cat_features, True)
        for col in tqdm(df.columns):
            if col not in ['customer_ID', 'S_2']:
                df[col] /= 100
            df[col] = df[col].fillna(0)

        df.to_feather(self._DIR_DATA + 'nn_series.feather')

        eps = 1e-3

        dfs = []
        for fn in ['cat', 'num', 'diff', 'rank_num', 'last3_cat', 'last3_num', 'last3_diff', 'last6_num',
                   'ym_rank_num']:
            if len(dfs) == 0:
                dfs.append(pd.read_feather(f'{self._DIR_DATA}/{fn}_feature.feather'))
            else:
                dfs.append(pd.read_feather(f'{self._DIR_DATA}/{fn}_feature.feather').drop([self.COL_UID], axis=1))

            if 'last' in fn:
                dfs[-1] = dfs[-1].add_prefix('_'.join(fn.split('_')[:-1]) + '_')

        for df in dfs:
            for col in tqdm(df.columns):
                if col not in ['customer_ID', 'S_2']:
                    # v_min = df[col].min()
                    # v_max = df[col].max()
                    # df[col] = (df[col]-v_min+eps) / (v_max-v_min+eps)
                    vc = df[col].value_counts().sort_index()
                    if True or len(vc) > 0:
                        bins = self.GreedyFindBin(vc.index.values, vc.values, len(vc), 255, vc.sum())
                        df[col] = np.digitize(df[col], [-np.inf] + bins)
                        df.loc[df[col] == len(bins) + 1, col] = 0
                        df[col] = df[col] / df[col].max()

        tmp = tmp.fillna(0)
        dfs.append(tmp)
        df = pd.concat(dfs, axis=1)

        df.to_feather(self._DIR_DATA + 'nn_all_feature.feather')

    def S5_LGB_main(self):
        root = self.args.root
        seed = self.args.seed

        df = pd.read_feather(f'{self._DIR_DATA}/all_feature.feather')

        train_y = pd.read_csv(f'{self._DIR_DATA_INPUT}/' + self._FILE_TRAIN_LABEL)

        # todo: change this to use train_feather count
        train_feather = pd.read_feather(self.get_path_train())
        train_feather = train_feather.groupby(self.COL_UID).count()[self.COL_TIME]

        train = df[:train_feather.shape[0]]
        train = train.merge(train_y, left_on=self.COL_UID, right_on=self.COL_UID)
        test = df[train_feather.shape[0]:].reset_index(drop=True)
        del df

        print(train.shape, test.shape)

        lgb_config = {
            'lgb_params': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting': 'dart',
                'max_depth': -1,
                'num_leaves': 64,
                'learning_rate': 0.035,
                'bagging_freq': 5,
                'bagging_fraction': 0.75,
                'feature_fraction': 0.05,
                'min_data_in_leaf': 256,
                'max_bin': 63,
                'min_data_in_bin': 256,
                # 'min_sum_heassian_in_leaf': 10,
                'tree_learner': 'serial',
                'boost_from_average': 'false',
                'lambda_l1': 0.1,
                'lambda_l2': 30,
                'num_threads': 24,
                'verbosity': 1,
            },
            'feature_name': [],
            'rounds': 4500,
            'early_stopping_rounds': 100,
            'verbose_eval': 50,
            'folds': 5,
            'seed': seed
        }

        lgb_config = {
            'lgb_params': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting': 'dart',
                'max_depth': -1,
                'num_leaves': 64,
                'learning_rate': 0.035,
                'bagging_freq': 5,
                'bagging_fraction': 0.75,
                'feature_fraction': 0.05,
                'min_data_in_leaf': 256,
                'max_bin': 63,
                'min_data_in_bin': 256,
                # 'min_sum_heassian_in_leaf': 10,
                'tree_learner': 'serial',
                'boost_from_average': 'false',
                'lambda_l1': 0.1,
                'lambda_l2': 30,
                'num_threads': 24,
                'verbosity': 1,
            },
            'feature_name': [col for col in train.columns if col not in [self.COL_UID, self.COL_LABEL,
                                                                         self.COL_TIME] and 'skew' not in col and 'kurt' not in col and 'sub_mean' not in col and 'div_mean' not in col],
            'rounds': 4500,
            'early_stopping_rounds': 100,
            'verbose_eval': 50,
            'folds': 5,
            'seed': seed
        }
        lgb_config['feature_name'] = [col for col in train.columns if
                                      col not in [self.COL_UID, self.COL_LABEL, 'S_2'] and 'target' not in col]
        self.Lgb_train_and_predict(train, test, lgb_config, aug=None, run_id='LGB_with_manual_feature')

        lgb_config['feature_name'] = [col for col in train.columns if
                                      col not in [self.COL_UID, self.COL_LABEL, self.COL_TIME]]
        self.Lgb_train_and_predict(train, test, lgb_config, aug=None, run_id='LGB_with_manual_feature_and_series_oof')

    def use_cuda(self):
        return True

    def NN_train_and_predict(self, train, test, model_class, config, use_series_oof, logit=False,
                             output_root='./output/',
                             run_id=None):
        if not run_id:
            run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            while os.path.exists(output_root + run_id + '/'):
                time.sleep(1)
                run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_root + f'{self.args.save_dir}/'
        else:
            output_path = output_root + run_id + '/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        os.system(f'copy ./*.py {output_path}'.replace('/', "\\"))
        feature_name = config['feature_name']
        obj_max = config['obj_max']
        epochs = config['epochs']
        smoothing = config['smoothing']
        patience = config['patience']
        lr = config['lr']
        batch_size = config['batch_size']
        folds = config['folds']
        seed = config['seed']
        if train is not None:
            train_series, train_feature, train_y, train_series_idx = train

            oof = train_y[[self.COL_UID]]
            oof['fold'] = -1
            oof[self.COL_LABEL] = 0.0
            oof[self.COL_LABEL] = oof[self.COL_LABEL].astype(np.float32)
        else:
            oof = None

        if train is not None:
            log = open(output_path + 'train.log', 'w', buffering=1)
            log.write(str(config) + '\n')

            all_valid_metric = []

            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

            model_num = 0
            train_folds = []

            for fold, (trn_index, val_index) in enumerate(skf.split(train_y, train_y[self.COL_LABEL])):

                train_dataset = TaskDataset(train_series, train_feature, [train_series_idx[i] for i in trn_index],
                                            train_y)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                              collate_fn=train_dataset.collate_fn, num_workers=self.args.num_workers,
                                              pin_memory=True)
                valid_dataset = TaskDataset(train_series, train_feature, [train_series_idx[i] for i in val_index],
                                            train_y)
                valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                              collate_fn=valid_dataset.collate_fn, num_workers=self.args.num_workers)

                model = model_class(223, (6375 + 13) * 2, 1, 3, 128, use_series_oof=use_series_oof)
                if run_id == 'NN_with_series_and_all_feature':
                    model = model_class(223, (len(train_feature.columns)-1)*2, 1, 3, 128, use_series_oof=use_series_oof)

                scheduler = Adam12()

                model.cuda()
                if self.args.use_apm:
                    scaler = amp.GradScaler()
                optimizer = scheduler.schedule(model, 0, epochs)[0]

                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)
                # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5,
                #                                                 max_lr=1e-2, epochs=epochs, steps_per_epoch=len(train_dataloader))
                # torch.optim.Adam(model.parameters(),betas=(0.9, 0.99), lr=lr, weight_decay=0.00001,eps=1e-5)
                gpus = list(range(torch.cuda.device_count()))
                if len(gpus) > 1:
                    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

                loss_tr = nn.BCELoss()
                loss_tr1 = nn.BCELoss(reduction='none')
                if obj_max == 1:
                    best_valid_metric = 0
                else:
                    best_valid_metric = 1e9
                not_improve_epochs = 0
                if self.args.do_train:
                    for epoch in range(epochs):
                        # if epoch <= 13:
                        #     continue
                        np.random.seed(666 * epoch)
                        train_loss = 0.0
                        train_num = 0
                        scheduler.step(model, epoch, epochs)
                        model.train()
                        bar = tqdm(train_dataloader)
                        for data in bar:  # train_dataloader:
                            optimizer.zero_grad()
                            if self.use_cuda():
                                for k in data:
                                    data[k] = data[k].cuda()
                            y = data['batch_y']
                            if self.args.use_apm:
                                with amp.autocast():
                                    outputs = model(data)
                                    # loss_series = loss_tr1(series_outputs,y.repeat(1,13))
                                    # loss_series = (loss_series * data['batch_mask']).sum() / data['batch_mask'].sum()
                                    # if epoch < 30:
                                    #     loss = loss_series
                                    # else:
                                    loss = loss_tr(outputs,
                                                   y)  # + loss_series # 0.5 * (loss_tr(outputs,y) + loss_feature(feature,y))
                                if str(loss.item()) == 'nan': continue
                                scaler.scale(loss).backward()
                                torch.nn.utils.clip_grad_norm(model.parameters(), clipnorm)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                outputs = model(data)
                                loss = loss_tr(outputs, y)
                                loss.backward()
                                optimizer.step()
                            # scheduler.step()
                            train_num += data['batch_feature'].shape[0]
                            train_loss += data['batch_feature'].shape[0] * loss.item()
                            bar.set_description('loss: %.4f' % (loss.item()))

                        train_loss /= train_num

                        # eval
                        model.eval()
                        valid_preds = []
                        for data in tqdm(valid_dataloader):

                            # for data in valid_dataloader:
                            if self.use_cuda():
                                for k in data:
                                    data[k] = data[k].cuda()
                            with torch.no_grad():
                                if logit:
                                    outputs = model(data).sigmoid()
                                    # feature,outputs = model(data)
                                    # outputs = outputs.sigmoid()
                                else:
                                    outputs = model(data)
                                    # feature,outputs = model(data)
                            valid_preds.append(outputs.detach().cpu().numpy())

                        valid_preds = np.concatenate(valid_preds).reshape(-1)
                        valid_Y = train_y.iloc[val_index][self.COL_LABEL].values  # oof train
                        valid_mean = np.mean(valid_preds)
                        valid_metric = self.Metric(valid_Y, valid_preds)

                        if obj_max * (valid_metric) > obj_max * best_valid_metric:
                            if len(gpus) > 1:
                                torch.save(model.module.state_dict(), output_path + 'fold%s.ckpt' % fold)
                            else:
                                torch.save(model.state_dict(), output_path + 'fold%s.ckpt' % fold)
                            not_improve_epochs = 0
                            best_valid_metric = valid_metric
                            self.Write_log(log,
                                           '[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, valid_mean:%.6f' % (
                                               epoch, optimizer.param_groups[0]['lr'], train_loss, valid_metric,
                                               valid_mean))
                        else:
                            not_improve_epochs += 1
                            self.Write_log(log,
                                           '[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, valid_mean:%.6f, NIE +1 ---> %s' % (
                                               epoch, optimizer.param_groups[0]['lr'], train_loss, valid_metric,
                                               valid_mean,
                                               not_improve_epochs))
                            if not_improve_epochs >= patience:
                                break

                # state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, torch.device('cuda' if torch.cuda.is_available() else 'cpu') )
                state_dict = torch.load(output_path + 'fold%s.ckpt' % fold,
                                        torch.device('cuda' if self.use_cuda() else 'cpu'))

                model = model_class(223, (6375 + 13) * 2, 1, 3, 128, use_series_oof=use_series_oof)
                if run_id == 'NN_with_series_and_all_feature':
                    model = model_class(223, (len(train_feature.columns)-1)*2, 1, 3, 128, use_series_oof=use_series_oof)
                if self.use_cuda():
                    model.cuda()
                model.load_state_dict(state_dict)
                if len(gpus) > 1:
                    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

                model.eval()

                valid_preds = []
                valid_Y = []
                for data in tqdm(valid_dataloader):

                    # for data in valid_dataloader:
                    if self.use_cuda():
                        for k in data:
                            data[k] = data[k].cuda()
                    with torch.no_grad():
                        if logit:
                            outputs = model(data).sigmoid()
                            # feature,outputs = model(data)
                            # outputs = outputs.sigmoid()
                        else:
                            outputs = model(data)
                            # feature,outputs = model(data)
                    valid_preds.append(outputs.detach().cpu().numpy())
                    if False:
                        valid_Y.append(y.detach().cpu().numpy())

                valid_preds = np.concatenate(valid_preds).reshape(-1)
                valid_Y = train_y.iloc[val_index][self.COL_LABEL].values  # oof train
                valid_mean = np.mean(valid_preds)
                valid_metric = self.Metric(valid_Y, valid_preds)
                self.Write_log(log,
                               '[fold %s] best_valid_metric: %.6f, best_valid_mean: %.6f' % (
                                   fold, valid_metric, valid_mean))

                all_valid_metric.append(valid_metric)
                oof.iloc[val_index, oof.columns.get_loc(self.COL_LABEL)] = valid_preds
                oof.iloc[val_index, oof.columns.get_loc('fold')] = fold
                train_folds.append(fold)

            mean_valid_metric = np.mean(all_valid_metric)
            self.Write_log(log, 'all valid mean metric:%.6f' % (mean_valid_metric))
            oof.loc[oof['fold'].isin(train_folds)].to_csv(output_path + 'oof.csv', index=False)

            if test is None:
                log.close()
                # os.rename(output_path + 'train.log', output_path + 'train_%.6f.log' % mean_valid_metric)
                if os.path.exists(output_path + '/train_%.6f.log' % mean_valid_metric):
                    remove(output_path + '/train_%.6f.log' % mean_valid_metric)
                move(output_path + '/train.log', output_path + '/train_%.6f.log' % mean_valid_metric)


            log_df = pd.DataFrame(
                {'run_id': [run_id], 'folds': folds, 'metric': [round(mean_valid_metric, 6)], 'lb': [np.nan],
                 'remark': [config['remark']]})
            if not os.path.exists(output_root + 'experiment_log.csv'):
                log_df.to_csv(output_root + 'experiment_log.csv', index=False)
            else:
                log_df.to_csv(output_root + 'experiment_log.csv', index=False, mode='a', header=None)

        if test is not None:
            if train is None:
                log = open(output_path + 'test.log', 'w', buffering=1)
                self.Write_log(log, str(config) + '\n')
            test_series, test_feature, test_series_idx = test

            sub = test_feature[-len(test_series_idx):][[self.COL_UID]].reset_index(drop=True)
            sub['prediction'] = 0

            test_dataset = TaskDataset(test_series, test_feature, test_series_idx)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                         collate_fn=test_dataset.collate_fn, num_workers=self.args.num_workers)
            models = []
            for fold in range(folds):
                if not os.path.exists(output_path + 'fold%s.ckpt' % fold):
                    continue
                model = model_class(223, (6375 + 13) * 2, 1, 3, 128, use_series_oof=use_series_oof)
                if run_id == 'NN_with_series_and_all_feature':
                    model = model_class(223, (len(train_feature.columns)-1)*2, 1, 3, 128, use_series_oof=use_series_oof)
                if self.use_cuda():
                    model.cuda()
                    state_dict = torch.load(output_path + 'fold%s.ckpt' % fold, torch.device('cuda'))
                else:
                    state_dict = torch.load(output_path + 'fold%s.ckpt' % fold, torch.device('cpu'))

                model.load_state_dict(state_dict)
                if len(gpus) > 1:
                    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

                model.eval()
                models.append(model)
            print('model count:', len(models))
            test_preds = []
            with torch.no_grad():
                for data in tqdm(test_dataloader):

                    # for data in test_dataloader:
                    if self.use_cuda():

                        for k in data:
                            data[k] = data[k].cuda()

                    if logit:
                        # outputs = model(data).sigmoid()
                        outputs = torch.stack([m(data).sigmoid() for m in models], 0).mean(0)
                        # feature,outputs = model(data)
                        # outputs = outputs.sigmoid()
                    else:
                        # outputs = model(data)
                        outputs = torch.stack([m(data) for m in models], 0).mean(0)
                        # feature,outputs = model(data)
                    test_preds.append(outputs.cpu().detach().numpy())
            test_preds = np.concatenate(test_preds).reshape(-1)
            test_mean = np.mean(test_preds)
            self.Write_log(log, 'test_mean: %.6f' % (test_mean))
            sub['prediction'] = test_preds
            sub.to_csv(output_path + 'submission.csv.zip', index=False, compression='zip')
        else:
            sub = None

        if self.args.save_dir in output_path:
            os.rename(output_path, output_root + run_id + '/')
        return oof, sub

    def S6_NN_main(self, first_train ):
        x = datetime.datetime.now()
        print('start: ', x)
        df = pd.read_feather(self._DIR_DATA + 'nn_series.feather')
        y = pd.read_csv(self._DIR_DATA_INPUT + 'train_labels.csv')

        # todo: change this to use train_feather count
        if True:
            train_feather = pd.read_feather(self.get_path_train())
            train_feather = train_feather.groupby('customer_ID').count()['S_2']
        # df_uid = df.groupby(self.COL_UID).count()['index']
        y = y.merge(train_feather, left_on=self.COL_UID, right_index=True).drop(columns='S_2')

        f = pd.read_feather(self._DIR_DATA + 'nn_all_feature.feather')
        df['idx'] = df.index
        series_idx = df.groupby(self.COL_UID, sort=False).idx.agg(['min', 'max'])
        series_idx['feature_idx'] = np.arange(len(series_idx))
        df = df.drop(['idx'], axis=1)
        print(f.head())
        if True:
            self.args.do_train = True
            self.args.batch_size = 512
            # https://github.com/pytorch/pytorch/issues/2341
            self.args.num_workers = 0
            #set folds to 5 when running real runs, orig setting
            folds = 2
        nn_config = {
            'id_name': self.COL_UID,
            'feature_name': [],
            'label_name': self.COL_LABEL,
            'obj_max': 1,
            'epochs': 10,
            'smoothing': 0.001,
            'clipnorm': 1,
            'patience': 100,
            'lr': 3e-4,
            'batch_size': 256,
            'folds': folds,
            'seed': self.args.seed,
            'remark': self.args.remark
        }
        if first_train:
            self.NN_train_and_predict([df, f, y, series_idx.values[:y.shape[0]]],
                                      [df, f, series_idx.values[y.shape[0]:]],
                                      Amodel, nn_config, use_series_oof=False, run_id='NN_with_series')

        # torch.backends.cudnn.enabled = False
        self.NN_train_and_predict([df, f, y, series_idx.values[:y.shape[0]]], [df, f, series_idx.values[y.shape[0]:]],
                                  Amodel, nn_config, use_series_oof=True, run_id='NN_with_series_and_all_feature')
        x = datetime.datetime.now()
        print('end: ', x)
    def S7_ensemble(self):
        p0 = pd.read_csv('./output/LGB_with_manual_feature/submission.csv.zip')
        p1 = pd.read_csv('./output/LGB_with_manual_feature_and_series_oof/submission.csv.zip')
        p2 = pd.read_csv('./output/NN_with_series/submission.csv.zip')
        p3 = pd.read_csv('./output/NN_with_series_and_all_feature/submission.csv.zip')

        p0['prediction'] = p0['prediction'] * 0.3 + p1['prediction'] * 0.35 + p2['prediction'] * 0.15 + p3[
            'prediction'] * 0.1

        p0.to_csv('./output/final_submission.csv.zip', index=False, compression='zip')
if __name__ == '__main__':

    model = model_amex()
    if False:
        model.S1_denoise()
    if False:
        model.S1pnt1_down_sample()

    if False:
        model.S2_manial_feature()
        model.S3_series_feature()
    if False:
         model.S4_feature_combined()
    if False:
        model.S5_LGB_main()
    if True:
        model.S6_NN_main(first_train=True)
        model.S7_ensemble()
        model.compute_score()
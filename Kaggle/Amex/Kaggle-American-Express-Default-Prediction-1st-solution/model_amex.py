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

class model_amex:
    def __init__(self):
        # to speed up set this to true and then only train data will be used to similuate both train andtest and then compute the out-of-sample score using the simultaed test data
        self.DOWN_SAMPLING = True
        # when building the model set this to false so that I can compute the score using simulated test data. real test data from Kaggle do not have label.
        self.USE_TEST_DATA = False
        # number of the data (for amex, userid's that will be sampled
        self.SIZE_TRAIN = 10000
        self.SIZE_TEST = 10000

        # folder where data are located such as train and test data
        self._DIR_DATA = r'../mine/'
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
        parser.add_argument("--root", type=str, default=self._DIR_DATA + '')
        parser.add_argument("--save_dir", type=str, default='tmp')
        parser.add_argument("--use_apm", action='store_true', default=False)
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--do_train", action='store_true', default=False)
        parser.add_argument("--test", action='store_true', default=False)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--remark", type=str, default='')
    
        self.args, unknown = parser.parse_known_args()

    def get_path_train_orig(self):
        return self._DIR_DATA + '/' + self._FILE_TRAIN

    def get_path_train(self):
        if self.DOWN_SAMPLING:
            return self._DIR_DATA + '/' + 'train_down_sampling_' + str(self.SIZE_TRAIN) + '.feather'
        else:
            return self._DIR_DATA + '/' + self._FILE_TRAIN_FEATHER_ORIG

    # not used
    def get_path_test(self):
        if self.USE_TEST_DATA:
            return self._DIR_DATA + '/' + self._FILE_TEST_FEATHER_ORIG
        else:
            return self._DIR_DATA + '/' + 'test_down_sampling_from_train_' + str(self.SIZE_TEST) + '.feather'

    if False:
        #not used


        #not used
        def build_label_encoder(self):
            label_encoder_model = dict()
            #manually check to see which file (test or train) contains the full set of values. in amex this is train
            df = pd.read_csv(self.get_path_train_orig(), usecols=self.NON_NUM_COLS)
            print('finished reading the NON_NUM_COLS from train_data')

            for c in self.NON_NUM_COLS:
                label_encoder_model[c] = LabelEncoder().fit(df[c])
                np.save(self._DIR_DATA + 'label_encoder_model_' + c + '.npy', label_encoder_model[c].classes_)

        #not used
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

        train = pd.read_csv(self.get_path_train_orig())
        train = denoise(train)
        train.to_feather(self._DIR_DATA + self._FILE_TRAIN_FEATHER_ORIG)

        del train
        #this fails the 2nd time, will need to process by batch
        test = pd.read_csv(self._DIR_DATA + self._FILE_TEST)
        test = denoise(test)
        test.to_feather(self._DIR_DATA + self._FILE_TEST_FEATHER_ORIG)

    def S1pnt1_down_sample(self):
        if self.DOWN_SAMPLING:

            df_orig = pd.read_feather(self._DIR_DATA + self._FILE_TRAIN_FEATHER_ORIG)
            customer_ID_samples = df_orig.groupby(self.COL_UID).count()[self.COL_TIME].to_frame().sample(n=self.SIZE_TRAIN+self.SIZE_TEST)

            df_orig = df_orig.merge(customer_ID_samples, left_on=self.COL_UID, right_index=True, suffixes=('', '_y')).drop(
                self.COL_TIME+'_y', axis=1)
            msk = np.random.rand(len(df_orig)) < self.SIZE_TRAIN/(self.SIZE_TRAIN+self.SIZE_TEST)
            train = df_orig[msk]
            train.reset_index().to_feather(self.get_path_train())
            test = df_orig[~msk]
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

    def num_feature(self,df, num_features, lastk):
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
                all_cols = [c for c in list(df.columns) if c not in [self.COL_UID, self.COL_TIME]]
                num_features = [col for col in all_cols if col not in self.cat_features]

                #special coded for amex prefix
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
                    df['ym'] = df[self.COL_TIME].apply(lambda x: x[:7])
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

    def Write_log(self,logFile, text, isPrint=True):
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
            os.mkdir(output_path)
        os.system(f'copy ./*.py {output_path}')
        os.system(f'copy ./*.sh {output_path}')
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
                      'all valid mean metric:%.6f, global valid metric:%.6f' % (mean_valid_metric, global_valid_metric))

            oof.to_csv(output_path + '/oof.csv', index=False)

            log.close()
            os.rename(output_path + '/train.log', output_path + '/train_%.6f.log' % mean_valid_metric)

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
            sub[[self.COL_UID, 'prediction']].to_csv(output_path + '/submission.csv.zip', compression='zip', index=False)
        if self.args.save_dir in output_path:
            os.rename(output_path, output_root + run_id + '/')
        return oof, sub, (mean_valid_metric, global_valid_metric)

    def S3_series_feature(self):
        train = pd.read_feather(self.get_path_train())

        test = pd.read_feather(self.get_path_test())

        eps = 1e-3

        train_y = pd.read_csv(self._DIR_DATA + self._FILE_TRAIN_LABEL)
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
        run_id = 'LGB_with_series_feature'
        # LGB_with_series_feature
        _dir = self._DIR_OUTPUT + run_id + '/'
        submission = pd.read_csv(_dir+'submission.csv.zip')
        train_y = pd.read_csv(self._DIR_DATA + self._FILE_TRAIN_LABEL)
        train_y = train_y.merge(submission, on=self.COL_UID)
        score = self.amex_metric_mod(train_y[self.COL_LABEL], train_y['prediction'])
        print(run_id + ' score: ' + str(score))

if __name__ == '__main__':

    model = model_amex()
    if False:
        model.S1_denoise()
        model.S1pnt1_down_sample()
        model.S2_manial_feature()
        model.S3_series_feature()
    model.compute_score()
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm

# from utils import *
from utils import NN_train_and_predict
from model import *
import datetime
import argparse

def train():

    _DIR_DATA = r'C:\John\git\vas\kaggle\americanExpress/'
    if False:
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=str, default=_DIR_DATA + '')
        parser.add_argument("--save_dir", type=str, default='tmp')
        parser.add_argument("--use_apm", action='store_true', default=False)
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--do_train", action='store_true', default=False)
        parser.add_argument("--test", action='store_true', default=False)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--remark", type=str, default='')

        args, unknown = parser.parse_known_args()
        root = args.root
        seed = args.seed

        id_name = 'customer_ID'
        label_name = 'target'


    x = datetime.datetime.now()
    print('start: ', x)
    df =  pd.read_feather(_DIR_DATA + 'nn_series.feather')
    y = pd.read_csv(_DIR_DATA + 'train_labels.csv')

#todo: change this to use train_feather count
    if False:
        train_feather = pd.read_feather(f'{root}/train.feather')
        train_feather = train_feather.groupby ('customer_ID').count()['S_2']
    df_uid = df.groupby ('customer_ID').count()['index']
    y = y.merge(df_uid, left_on='customer_ID', right_index=True).drop(columns='index')


    f = pd.read_feather(_DIR_DATA + 'nn_all_feature.feather')
    df['idx'] = df.index
    series_idx = df.groupby('customer_ID',sort=False).idx.agg(['min','max'])
    series_idx['feature_idx'] = np.arange(len(series_idx))
    df = df.drop(['idx'],axis=1)
    print(f.head())
    nn_config = {
        'id_name':id_name,
        'feature_name':[],
        'label_name':label_name,
        'obj_max': 1,
        'epochs': 10,
        'smoothing': 0.001,
        'clipnorm': 1,
        'patience': 100,
        'lr': 3e-4,
        'batch_size': 256,
        'folds': 5,
        'seed': seed,
        'remark': args.remark
    }
    if False:
        args.do_train=True
        args.batch_size=512
    #https://github.com/pytorch/pytorch/issues/2341
        args.num_workers=4
    if True:
        NN_train_and_predict([df,f,y,series_idx.values[:y.shape[0]]],[df,f,series_idx.values[y.shape[0]:]],Amodel,nn_config,use_series_oof=False,run_id='NN_with_series')

    NN_train_and_predict([df,f,y,series_idx.values[:y.shape[0]]],[df,f,series_idx.values[y.shape[0]:]],Amodel,nn_config,use_series_oof=True,run_id='NN_with_series_and_all_feature')
    x = datetime.datetime.now()
    print('end: ', x)
if __name__ == '__main__':
    train()
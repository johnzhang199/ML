import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def denoise(df):
    df['D_63'] = df['D_63'].apply(lambda t: {'CR':0, 'XZ':1, 'XM':2, 'CO':3, 'CL':4, 'XL':5}[t]).astype(np.int8)
    df['D_64'] = df['D_64'].apply(lambda t: {np.nan:-1, 'O':0, '-1':1, 'R':2, 'U':3}[t]).astype(np.int8)
    for col in tqdm(df.columns):
        if col not in ['customer_ID','S_2','D_63','D_64']:
            df[col] = np.floor(df[col]*100)
    return df
_DIR = r'C:\John\git\vas\kaggle\americanExpress/'

train = pd.read_csv(_DIR + '/train_data.csv')
train = denoise(train)
train.to_feather(_DIR + 'train_orig.feather')

del train

test = pd.read_csv(_DIR + 'test_data.csv')
test = denoise(test)
test.to_feather(_DIR + 'test_orig.feather')

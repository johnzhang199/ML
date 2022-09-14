#Parameter grid search with xgboost
#feature engineering is not so useful and the LB is so overfitted/underfitted
#so it is good to trust your CV

#go xgboost, go mxnet, go DMLC! http://dmlc.ml

#Credit to Shize's R code and the python re-implementation

import pandas as pd
import numpy as np
import xgboost as xgb
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split

from sklearn.model_selection import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

# from sklearn.metrics import precision_recall_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_data():

    path = r'C:\John\job\2022\migo\migo_interview_dataset_20220316.csv/migo_interview_dataset_20220316.csv'

    df = pd.read_csv(path)

    df = pd.concat([df,pd.get_dummies(df.channel, prefix='channel')], axis=1)
    df = pd.concat([df,pd.get_dummies(df.state, prefix='state')], axis=1)



    X = df.drop(['loan_defaulted','state','channel'
                 ], axis=1)

    y = df['loan_defaulted']

    # convert labels into binary values

    y[y == False] = 0

    y[y == True] = 1


    return X,y
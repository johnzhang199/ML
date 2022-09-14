# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#add Title feature
########################################################################################################################################
FOLDER = '/kaggle/input/titanic/'
FOLDER = r'C:\John\git\vas\kaggle\titanic/'
df_train = pd.read_csv(FOLDER + 'train.csv')
df_test = pd.read_csv(FOLDER + 'test.csv')
combined = [df_train, df_test]

for df in combined:
    df_tmp = df['Name'].str.extract(' ([A-Za-z]+)\.')
    df['Title'] = df_tmp

#impute
########################################################################################################################################
# df_train
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
median_columns_list = ['Age', 'Fare']
mode_columns_list = ['Embarked']
ct = ColumnTransformer(
    [("median_imp", SimpleImputer(strategy = 'median'), median_columns_list),
     ("mode_imp", SimpleImputer(strategy = 'most_frequent'), mode_columns_list)])
i=0
for df in    combined:
    df[median_columns_list+mode_columns_list] = pd.DataFrame(ct.fit_transform(df), index=df.index, columns=median_columns_list+mode_columns_list)
    print(df)
    combined[i] = df
    i += 1
df_train2 = combined[0]
combined[0][median_columns_list+mode_columns_list].to_csv(FOLDER+'imputed.csv')
df_test2 = combined[1]
#onehot encoder
########################################################################################################################################
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title']
OH_encoder.fit(pd.concat([df_train[cat_cols], df_test[cat_cols]]))
i=0
for df in    combined:
    OH_cols = pd.DataFrame(OH_encoder.transform(df[cat_cols]))
    OH_cols.columns = OH_encoder.get_feature_names_out()
    # One-hot encoding removed index; put it back
    OH_cols.index = df.index

    # Remove categorical columns (will replace with one-hot encoding)
    df = df.drop(cat_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    df = pd.concat([df, OH_cols], axis=1)
    combined[i] = df
    i += 1
    print(df)
#     print(df)
df_train2 = combined[0]
df_test2 = combined[1]
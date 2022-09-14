#Parameter grid search with xgboost
#feature engineering is not so useful and the LB is so overfitted/underfitted
#so it is good to trust your CV

#go xgboost, go mxnet, go DMLC! http://dmlc.ml

#Credit to Shize's R code and the python re-implementation

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt


from GridSearchFunc import gridSearch


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



path_train = r'C:\John\git\vas\kaggle\titanic/train.csv'

df_train = pd.read_csv(path_train)

path_test = r'C:\John\git\vas\kaggle\titanic/test.csv'

df_test = pd.read_csv(path_test)

df_train.groupby('Pclass').mean()['Survived']
df_train.groupby('Sex').mean()['Survived']

df_train['Family'] = df_train['SibSp'] + df_train['Parch']

df_train.groupby('SibSp', as_index=False).agg(['count','mean'])[['Survived']].sort_values(('Survived', 'mean'), ascending=False)
df_train.groupby('Parch', as_index=False).agg(['count','mean'])[['Survived']].sort_values(('Survived', 'mean'), ascending=False)
df_train.groupby('Family', as_index=False).agg(['count','mean'])[['Survived']].sort_values(('Survived', 'mean'), ascending=False)


combine = [df_train, df_test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df_train.groupby('Title', as_index=False).agg(['count','mean'])[['Survived']].sort_values(('Survived', 'mean'), ascending=False)


df_train.head()

grid = sns.FacetGrid(df_train, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

x_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']
y_col = 'Survived'
df_train2 = df_train[x_cols + [y_col]]


cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title']


# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(df_train2[cat_cols]))
OH_columns = OH_encoder.get_feature_names_out()
OH_cols_traincolumns = OH_columns

# One-hot encoding removed index; put it back
OH_cols_train.index = df_train2.index



# Remove categorical columns (will replace with one-hot encoding)
df_train2 = df_train2.drop(cat_cols, axis=1)

# Add one-hot encoded columns to numerical features
df_train2 = pd.concat([df_train2, OH_cols_train], axis=1)
# print(OH_X_train)


# for c in cat_cols:
#     df_train2 = pd.concat([df_train2,pd.get_dummies(df_train[c], prefix=c)], axis=1)
#     df_train2 = df_train2.drop([c
#              ], axis=1)

# df_train = pd.concat([df_train,pd.get_dummies(df_train.state, prefix='state')], axis=1)



X = df_train2.drop([y_col
             ], axis=1)
y = df_train2[y_col]
stat = y.to_frame().describe()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have
#much fun of fighting against overfit
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {
    'booster': ['gbtree'],#,'gblinear'],
    # 'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              # 'learning_rate': [0.01,0.02,0.03,0.04,0.05], #so called `eta` value
              'learning_rate': [0.09],#[0.01 * x for x in range(1,21)],#[0.01,0.02,0.03,0.04,0.050.05], #so called `eta` value
                'min_child_weight': [1.45],#[ 1.3 + 0.01*x for x in range(1,21)],
              'max_depth': [10],#[ x for x in range(3,11)],#[6],
              'gamma': [0],#[ 0.01*x for x in range(30)],#[11],
              'silent': [1],
              'subsample': [0.9],#[ 0.1*x for x in range(5,11)],#[0.8],
              'colsample_bytree': [0.5],#[ 0.1*x for x in range(5,11)],#
              'n_estimators': [17],#[x for x in range(3,20)],#[1000], #number of trees, change it to 1000 for better results
              'missing':[-999],
              # 'seed': [1337],
               'scale_pos_weight' : [1/stat.loc['mean'][0]]}

xgb_model = xgb.XGBClassifier()

clf = gridSearch(xgb_model, parameters, X, y, test_size = 0)

df_test['Pclass'].unique()

x_cols_test = ['PassengerId'] + x_cols
df_test2 = df_test[x_cols_test]

df_test2.columns
OH_cols_valid = pd.DataFrame(OH_encoder.transform(df_test2[cat_cols]))
OH_cols_valid.index = df_test2.index
OH_cols_valid.columns = OH_columns
num_df_test2 = df_test2.drop(cat_cols, axis=1)
df_test2 = pd.concat([num_df_test2, OH_cols_valid], axis=1)

# y_col = 'Survived'

# cat_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin']
# for c in cat_cols:
#     df_test2 = pd.concat([df_test2,pd.get_dummies(df_test[c], prefix=c)], axis=1)
#     df_test2 = df_test2.drop([c
#              ], axis=1)

# df_test = pd.concat([df_test,pd.get_dummies(df_test.state, prefix='state')], axis=1)





test_class = clf.predict(df_test2.drop('PassengerId', axis=1))

df_test2['Survived'] = test_class
# df_test2.head()
df_test2.to_csv(r'C:\John\git\vas\kaggle\titanic/submissionFull.csv', index=False)
df_test2[['PassengerId','Survived']].to_csv(r'C:\John\git\vas\kaggle\titanic/submission.csv', index=False)
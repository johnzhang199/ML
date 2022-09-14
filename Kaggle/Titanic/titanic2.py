import pandas as pd

pd.set_option('display.max_rows', 10)

#load data
PATH_ = r'C:\John\git\vas\kaggle\titanic/'
path_train = PATH_ + 'train.csv'
path_test = PATH_ + 'test.csv'
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

#access nan situation
df_train.info()

#access outliers
#add Title

combine = [df_train, df_test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # dataset['Family'] = dataset['SibSp'] + dataset[]

x_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']
y_col = 'Survived'
# df_train2 = df_train[x_cols + [y_col]]


#let's convert all categoricals into onehot
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title']

from sklearn.model_selection import train_test_split

# import XGBClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# declare parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'alpha': 10,
    'learning_rate': 1.0,
    'n_estimators': 100
}

# instantiate the classifier
xgb_clf = XGBClassifier(**params)

y = df_train[y_col]

#fit for each feature
for c in x_cols:
    print (c)
    df_train2 = df_train[[c]]
    if c in cat_cols:
        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(df_train2[[c]]))
        OH_columns = OH_encoder.get_feature_names_out()
        OH_cols_train.columns = OH_columns

        # One-hot encoding removed index; put it back
        OH_cols_train.index = df_train2.index



        # Remove categorical columns (will replace with one-hot encoding)
        # df_train2 = df_train2.drop(c, axis=1)

        # Add one-hot encoded columns to numerical features
        df_train2 = OH_cols_train

    X = df_train2

#split into test and train from train data
# split X and y into training and testing sets


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    #figure out the importance of the features by fitting all features into a xgboost then we can add one feature at a time to see where our out of sample score goes donw meaning we are over fitting

    # fit the classifier to the training data
    xgb_clf.fit(X_train, y_train)

    # make predictions on test data

    y_pred = xgb_clf.predict(X_test)

    # compute and print accuracy score


    print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
pass
#
# import xgboost as xgb
# import matplotlib.pyplot as plt  # for plotting facilities
# xgb.plot_importance(xgb_clf)
# plt.figure(figsize = (16, 12))
# plt.show()
#
# X_test.columns
#
#
# from xgboost import cv
#
# params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                 'max_depth': 5, 'alpha': 10}
# data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
# xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
#                     num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
#
# #feature importance
# import matplotlib.pyplot as plt  # for plotting facilities
# xgb.plot_importance(xgb_cv)
# plt.figure(figsize = (16, 12))
# plt.show()

import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa

_DIR = r'C:\John\git\vas\kaggle\americanExpress/'

def get_nrows_train_with_label(nrows):
    if False:
        pf = ParquetFile(_DIR + 'test.parquet')
        first_ten_rows = next(pf.iter_batches(batch_size = nrows))
        df = pa.Table.from_batches([first_ten_rows]).to_pandas()
    df = pd.read_csv(_DIR + 'train_data.csv', nrows=nrows)
    # print(df.head())

    targets = pd.read_csv(_DIR + 'train_labels.csv')

    # pass

    train = df.merge(targets, on='customer_ID')#, how='left')
# train.target = train.target.astype('int8')

    # print(train.head())
    return train


df_train_with_label = get_nrows_train_with_label(100000)

# print(df_train_with_label.target.unique())
if False:
    df_train_segmented = dict()
    PREFIXES = ['D','S','P','B','R']
    for p in PREFIXES:
        filter_col = [col for col in df_train_with_label if col.startswith(p+'_')]
        filter_col.sort(key=lambda x: int(x[2:]))
    #     print(filter_col)
        df_train_segmented[p] = df_train_with_label[filter_col]

# Let's use D_39 as the variable to predict label using LSTM


#for simplicity let's keep only customers with 13 rows
# df_train_with_label.groupby('customer_ID').count()['target'].reset_index().groupby('target').count()['customer_ID']
customers_with_13_rows = df_train_with_label.groupby('customer_ID').count()['target'] == 13
customers_with_13_rows = customers_with_13_rows[customers_with_13_rows]


# df_train_with_label2.columns

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

df_train_with_label2 = df_train_with_label.merge(customers_with_13_rows, left_on='customer_ID', right_index=True, suffixes=('', '_y')).drop('target_y',axis=1)



n_steps = 13

df_train_with_label2.S_2 = pd.to_datetime( df_train_with_label2.S_2 )

df_train_with_label2.sort_values(['customer_ID', 'S_2'])

df_train_with_label2.set_index(['customer_ID', 'S_2'], inplace=True)

# df_train_with_label2['D_39'] = df_train_with_label2.groupby('customer_ID')['D_39'].transform(lambda x: x.ffill())
for c in df_train_with_label2:
    df_train_with_label2[c] = df_train_with_label2.groupby('customer_ID')[c].transform(lambda x: x.ffill())
    df_train_with_label2[c] = df_train_with_label2.groupby('customer_ID')[c].transform(lambda x: x.bfill())

#ffill then bfill for same customerID
#very slow below compared to above
if False:
    for id, new_df in df_train_with_label2.groupby ('customer_ID'):
        df_train_with_label2.loc[id,:] = df_train_with_label2.loc[id,:].fillna(method='ffill')
        df_train_with_label2.loc[id,:] =   df_train_with_label2.loc[id,:].fillna(method='bfill')
    # break
print('done')
# y.shape
df_train_with_label2.max()

max_rows = pd.get_option('display.max_rows')
pd.set_option('display.max_rows', None)
print(df_train_with_label2.isnull().sum().sort_index())
pd.set_option('display.max_rows', max_rows)

from sklearn.model_selection import train_test_split


weight0 = df_train_with_label2.groupby('target').count()['D_39']
weight1 = weight0/weight0.sum()


# yhat = model.predict_classes(X_test)
# yhat=np.argmax(predict_x,axis=1)

# from sklearn.metrics import accuracy_score

# print('LSTM model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, yhat)))
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# COMPETITION METRIC FROM Konstantin Yakovlev
# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
import numpy as np
def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

#split customer_ID into test amd train
ids = df_train_with_label2.reset_index()['customer_ID'].unique()
Id_train, Id_test, y_train, y_test = train_test_split(ids, ids, test_size = 0.3, random_state = 0)

pd.DataFrame(Id_train, columns=['customer_ID'])
df_train_with_label_train = df_train_with_label2.reset_index().merge(pd.DataFrame(Id_train, columns=['customer_ID']), on='customer_ID').set_index(['customer_ID', 'S_2'])#, how='left')
df_train_with_label_test = df_train_with_label2.reset_index().merge(pd.DataFrame(Id_test, columns=['customer_ID']), on='customer_ID').set_index(['customer_ID', 'S_2'])#, how='left')

y = df_train_with_label_train.groupby('customer_ID').mean()['target'].to_numpy()
y_test = df_train_with_label_test.groupby('customer_ID').mean()['target'].to_numpy()


FEATURES = ['D_39','D_47']
# FEATURES = ['D_47']
X= df_train_with_label_train[FEATURES].to_numpy().reshape(-1,n_steps,len(FEATURES))
X_test= df_train_with_label_test[FEATURES].to_numpy().reshape(-1,n_steps,len(FEATURES))

model = Sequential()
model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_steps, len(FEATURES))))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')#, metrics=['accuracy'])
history = model.fit(X, y, epochs=10, verbose=1, sample_weight=sample_weight)
yhat=np.round((model.predict(X_test)>=0.5)).astype(int).reshape(-1)
print(amex_metric_mod(y_test, yhat))

max_rows = pd.get_option('display.max_rows')
pd.set_option('display.max_rows', None)
print(df_train_with_label2.isnull().sum().sort_index())
pd.set_option('display.max_rows', max_rows)

sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 1] = weight1[1]
sample_weight[y_train == 0] = weight1[0]

# plot learning curves
from matplotlib import pyplot
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()
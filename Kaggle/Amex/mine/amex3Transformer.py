import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime
import winsound
from matplotlib import pyplot
import os
# from sklearn.preprocessing import OneHotEncoder

def Beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)


_DIR = r'C:\John\git\vas\kaggle\americanExpress/'
n_classes = 2
n_steps = 13
CAT_COLS = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
NON_NUM_COLS = [ 'D_63', 'D_64']
TOTAL_TRAIN_BATCH = 1
df_train_with_label_test = pd.DataFrame()



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


# COMPETITION METRIC FROM Konstantin Yakovlev
# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
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


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def get_median_s_2(df_train_with_label2):
    df_train_with_label2_dates = df_train_with_label2.reset_index()
    df_train_with_label2_dates['no'] = df_train_with_label2_dates.index % n_steps
    # df['date'].astype('datetime64[ns]').quantile(0.5, interpolation="midpoint")
    df_train_with_label2_dates['S_2_UNIX'] =    df_train_with_label2_dates['S_2'].apply(lambda x: time.mktime(x.timetuple()))
    df_train_with_label2_dates.info()
    df_train_with_label2_dates = df_train_with_label2_dates.groupby('no').median()['S_2_UNIX']
    # df_train_with_label2_dates = df_train_with_label2_dates.groupby('no').mean()['S_2_UNIX']
    df_train_with_label2_dates = df_train_with_label2_dates.to_frame()
    df_train_with_label2_dates.columns = ['S_2_UNIX']
    df_train_with_label2_dates['S_2'] =  df_train_with_label2_dates['S_2_UNIX'].apply( datetime.fromtimestamp)
    return df_train_with_label2_dates[['S_2']]

def get_seq_no(dt):
    df_train_with_label2_dates = pd.DataFrame(columns=['S_2'], data = {
        datetime.strptime('2017/03/17', '%Y/%m/%d'),
        datetime.strptime('2017-04-16', '%Y-%m-%d'),
        datetime.strptime('2017-05-17', '%Y-%m-%d'),
        datetime.strptime('2017-06-16', '%Y-%m-%d'),
        datetime.strptime('2017-07-17', '%Y-%m-%d'),
        datetime.strptime('2017-08-17', '%Y-%m-%d'),
        datetime.strptime('2017-09-16', '%Y-%m-%d'),
        datetime.strptime('2017-10-17', '%Y-%m-%d'),
        datetime.strptime('2017-11-17', '%Y-%m-%d'),
        datetime.strptime('2017-12-16', '%Y-%m-%d'),
        datetime.strptime('2018-01-17', '%Y-%m-%d'),
        datetime.strptime('2018-02-15', '%Y-%m-%d'),
        datetime.strptime('2018-03-17', '%Y-%m-%d'),
    })
    df_train_with_label2_dates = df_train_with_label2_dates.sort_values('S_2').reset_index()[['S_2']]
    time_diff = df_train_with_label2_dates - dt
    min_distince = time_diff.apply(abs).min()
    idx = time_diff.apply(abs) == min_distince
    return time_diff.loc[idx.S_2].index[0]

def pad_series(df_train_with_label2_less_than13):
    df_train_with_label2_less_than13_13 = pd.DataFrame()
    customer_ID_prev = None
    # no_prev = None
    df_customer = None
    no_range = range(n_steps)

    def add_customer(df_customer):
        df_customer = df_customer.reset_index().rename(columns={'level_0': 'customer_ID', 'level_1': 'S_2'})
        df_customer['no'] = df_customer.index + n_steps - df_customer.index.max() - 1
        df_customer.set_index(['customer_ID', 'no'], inplace=True)

        df_customer = df_customer.reindex(
            pd.MultiIndex.from_product([[customer_ID_prev], no_range], names=['customer_ID', 'no']))
        return pd.concat([df_train_with_label2_less_than13_13, df_customer])

    for index, row in df_train_with_label2_less_than13.iterrows():
        # print(type(row))
        # print(index[0])
        customer_ID_cur = index[0]
        df_customer_cur = row.to_frame().transpose()
        if df_customer is None:
            df_customer  = df_customer_cur
        elif index[0] == customer_ID_prev:
            df_customer = pd.concat([df_customer, df_customer_cur])
        else:
            # print(df_customer)
            df_train_with_label2_less_than13_13 = add_customer(df_customer)
            df_customer = df_customer_cur
            # df_customer.index = index

            # df_customer = None
            # no_prev = None
            # break

        customer_ID_prev = index[0]
    df_train_with_label2_less_than13_13 = add_customer(df_customer)
    return df_train_with_label2_less_than13_13
        # print(customer_ID_cur)
        # print(index[0])
callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
# label_encoder = LabelEncoder()

# array(['CR', 'CO', 'CL', 'XM', 'XZ', 'XL'], dtype=object), test and train
# df['D_63'].unique()
#array([nan, 'U', 'O', 'R'], dtype=object) for D_64 test
# array(['O', 'R', nan, 'U', '-1'], dtype=object) train
# df['D_64'].unique()
# we load train and get the label model first


label_encoder_model = dict()
if os.path.exists(_DIR + 'label_encoder_model_D_63.npy'):

    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    for c in NON_NUM_COLS:
        _file_path = _DIR + 'label_encoder_model_' + c + '.npy'
        label_encoder_model[c] = LabelEncoder()

        label_encoder_model[c].classes_ = np.load(_file_path)
    # restore np.load for future normal usage
    np.load = np_load_old
else:

    df = pd.read_csv(_DIR + 'train_data.csv', usecols=NON_NUM_COLS)
    print('finished reading the NON_NUM_COLS from train_data')

    for c in NON_NUM_COLS:
        # label_encoder_model[c] = LabelEncoder().fit(df[c])
        np.save(_DIR + 'label_encoder_model_' + c+'.npy', label_encoder_model[c].classes_)

    if False:
        Beep()
        label_encoder_model[c].transform(df[c])
        for c in NON_NUM_COLS:
            le_name_mapping = dict(zip(label_encoder_model[c].classes_, label_encoder_model[c].transform(label_encoder_model[c].classes_)))
            print(le_name_mapping)

    #remove memeory
    del df

targets = pd.read_csv(_DIR + 'train_labels.csv')

CHUNKSIZE = 10000
CHUNKSIZE = 400000

TRAIN_DATA_FILLED_DOWN_PATH = _DIR + 'train_data_' + str(CHUNKSIZE) + '.csv'
i2 = 0
last_partial_customer = pd.DataFrame()
results_df = pd.DataFrame()

TRAIN_DATA_PATH = TRAIN_DATA_FILLED_DOWN_PATH if os.path.exists(TRAIN_DATA_FILLED_DOWN_PATH) else _DIR + 'train_data.csv'
already_preprocessed = True if os.path.exists(TRAIN_DATA_FILLED_DOWN_PATH) else False
SAVE_FILLED_DOWN_FILE = True
header = True
#process large file in chunk_size
for chunk in pd.read_csv(_DIR + 'train_data.csv', chunksize=CHUNKSIZE):

    print('processing ' + str(i2) + ' chunk of train data file')

    if not  last_partial_customer.empty:
        chunk = pd.concat([last_partial_customer, chunk])
    #save the last customer_id
    last_partial_customer_id = chunk['customer_ID'].iloc[-1]
    last_partial_customer = chunk[chunk['customer_ID']==last_partial_customer_id]
    chunk = chunk[chunk['customer_ID'] != last_partial_customer_id]


    #onlydo filldown and processing if not already odne so before
    if not already_preprocessed:
        df_train_with_label = chunk.merge(targets, on='customer_ID')#, how='left')


    #0.7509843617246207, fillna(0)
    #0.7699122059511343 200k filldown
    # NROWS= 200000
    #0.7573584243776129 for 400k fillna(0)
    # NROWS= 400000
    #0.7106986785962519 fillna(0), 100k
    #0.7289368778032432 for filldown and then up. 100 k
    # NROWS= 100000
    # NROWS=10000
    # df_train_with_label = get_nrows_train_with_label(NROWS)

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
        customers_with_13_rows = df_train_with_label.groupby('customer_ID').count()['target'] == n_steps
        customers_with_13_rows = customers_with_13_rows[customers_with_13_rows]

    # df_train_with_label2.columns

    # from tensorflow.keras import Sequential
    # from tensorflow.keras.layers import LSTM
    # from tensorflow.keras.layers import Dense

        df_train_with_label2 = df_train_with_label.merge(customers_with_13_rows, left_on='customer_ID', right_index=True, suffixes=('', '_y')).drop('target_y',axis=1)




        df_train_with_label2.S_2 = pd.to_datetime( df_train_with_label2.S_2 )

    # df_train_with_label2.sort_values(['customer_ID', 'S_2'], inplace=True)

        df_train_with_label2.set_index(['customer_ID', 'S_2'], inplace=True)

        if False:
            df_train_with_label2_less_than13.to_csv(_DIR+'df_train_with_label2_less_than13.csv')




    # print('done')
    #isntead of pdding all to 13 periods, use 12 periods, 11 periods , etc to train its own model as padded model performs terribly with fillup/down; 0.03 score instead of 0.62 for 10 k rows
            index_2='S_2'
        if False:
            index2 = 'no'
            df_train_with_label2 = df_train_with_label2.reset_index()
            df_train_with_label2['no'] = df_train_with_label2.index % n_steps
            df_train_with_label2.set_index(['customer_ID', 'no'], inplace=True)

            customers_with_less_than_13_rows = df_train_with_label.groupby('customer_ID').count()['target'] < n_steps
            customers_with_less_than_13_rows = customers_with_less_than_13_rows[customers_with_less_than_13_rows]
            df_train_with_label2_less_than13 = df_train_with_label.merge(customers_with_less_than_13_rows, left_on='customer_ID', right_index=True, suffixes=('', '_y')).drop('target_y',axis=1)
            df_train_with_label2_less_than13.S_2 = pd.to_datetime( df_train_with_label2_less_than13.S_2 )
            df_train_with_label2_less_than13.set_index(['customer_ID', 'S_2'], inplace=True)
            df_train_with_label2_less_than13_padded = pad_series(df_train_with_label2_less_than13)

            df_train_with_label2 = pd.concat([df_train_with_label2, df_train_with_label2_less_than13_padded])
            if False:
                df_train_with_label2_less_than13_padded.to_csv(_DIR + 'df_train_with_label2_less_than13_padded.csv')
        # iii = 1
        if False:
            # df_train_with_label2_less_than13['no'] = df_train_with_label2_less_than13['S_2'].apply(get_seq_no)
            df_train_with_label2_less_than13.sort_values(['customer_ID', 'S_2'], inplace=True)

            df_train_with_label2_less_than13.set_index(['customer_ID', 'S_2'], inplace=True)
            _len = len(df_train_with_label2_less_than13)

            df_train_with_label2_less_than13_13 = pd.DataFrame()
            customer_ID_prev = None
            # no_prev = None
            df_customer = None
            no_range = range(n_steps)
            for index, row in df_train_with_label2_less_than13.iterrows():
                # print(type(row))
                # print(index[0])
                customer_ID_cur = index[0]
                df_customer_cur = row.to_frame().transpose()
                if df_customer is None:
                    df_customer  = df_customer_cur
                elif index[0] == customer_ID_prev:
                    df_customer = pd.concat([df_customer, df_customer_cur])
                else:
                    # print(df_customer)
                    df_customer  = df_customer.reset_index().rename(columns={'level_0': 'customer_ID', 'level_1': 'S_2'})
                    df_customer['no'] = df_customer.index + n_steps - df_customer.index.max() - 1
                    df_customer.set_index(['customer_ID', 'no'], inplace=True)

                    df_customer= df_customer.reindex(pd.MultiIndex.from_product([[customer_ID_prev], no_range], names=['customer_ID', 'no']))
                    df_train_with_label2_less_than13_13 = pd.concat([df_train_with_label2_less_than13_13,df_customer])
                    # df_customer = None
                    # df_customer.index = index

                    # df_customer = None
                    no_prev = None
                    break

                customer_ID_prev = index[0]
                # print(customer_ID_cur)
                # print(index[0])



            customer_ID = df_train_with_label2_less_than13.groupby('customer_ID').count()['P_2'].index.tolist()

            pad_series(df_train_with_label2_less_than13.iloc[:1000])


        if False:
            datetime.fromtimestamp(time.mktime(datetime.now().timetuple()))
            datetime.fromtimestamp(df_train_with_label2_dates['S_2_UNIX'][0])


        if False:
            #find the closest no
            # df_train_with_label2_dates = get_median_s_2(df_train_with_label2)
            #
            # time_diff = df_train_with_label2_dates - df_train_with_label2_less_than13['S_2'].iloc[0]
            # min_distince = time_diff.apply(abs).min()
            # idx = time_diff.apply(abs)==min_distince
            # time_diff.loc[idx.S_2].index[0]
            df_train_with_label2_less_than13['no'] = df_train_with_label2_less_than13['S_2'].apply(get_seq_no)
            print(df_train_with_label2_less_than13)

            df_train_with_label2_less_than13_2 = df_train_with_label2_less_than13.set_index(['customer_ID', 'no'])
            df_train_with_label2_less_than13_2.iloc[:1000].to_csv(_DIR+'df_train_with_label2_less_than13_2.csv')

        if False:
            #sort the S_2 for each customer and then set to 12,11, etc
            df_train_with_label2_less_than13.set_index(['customer_ID', 'S_2'], inplace=True)



        # y.shape
        if False:
            df_train_with_label2.max()

            max_rows = pd.get_option('display.max_rows')
            pd.set_option('display.max_rows', None)
            print(df_train_with_label2.isnull().sum().sort_index())
            pd.set_option('display.max_rows', max_rows)

        # df_train_with_label2['D_39'].reshape(len(df_train_with_label2['D_39'])/n_steps, n_steps, 1)




        if False:
            weight0 = df_train_with_label2.groupby('target').count()['D_39']
            weight1 = weight0/weight0.sum()


    # yhat = model.predict_classes(X_test)
    # yhat=np.argmax(predict_x,axis=1)

    # from sklearn.metrics import accuracy_score

    # print('LSTM model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, yhat)))
    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1]*100))


        if False:
            pd.DataFrame(Id_train, columns=['customer_ID'])



        cols_already_filled = []


        if False:
            FEATURES = ['B_' + str(x) for x in range(1,34)] \
                + ['B_' + str(x) for x in range(36,39)] \
                + ['B_' + str(x) for x in range(40,43)] + [             'D_39','D_47']#.4484

        if False:
            df_train_with_label_train.columns
        if i2 == 0:
            FEATURES = [x for x in df_train_with_label2.columns if x != 'target']# and x != 'S_2']
            cnt = len(FEATURES)
#
# \
#            '['B_1','B_2','B_3','B_4'
        if False:
            FEATURES = ['B_' + str(x) for x in range(1,15)] + [             'D_39','D_47']#0.41
            FEATURES = ['B_' + str(x) for x in range(1,20)] + [             'D_39','D_47']#0.02
            FEATURES = ['B_' + str(x) for x in range(1,17)] + [             'D_39','D_47']#0.48
            FEATURES = ['B_' + str(x) for x in range(1,18)] + [             'D_39','D_47']#0.444
            FEATURES = ['B_' + str(x) for x in range(1,18)] + [             'D_39','D_47']#0.444
            # FEATURES = ['B_' + str(x) for x in range(1,16)] + [ 'B_17',            'D_39','D_47']#0.02

        if False:
            FEATURES = [x for x in FEATURES if x not in NON_NUM_COLS]

        if False:
            df_train_with_label_train[FEATURES].isnull().sum()
            df_train_with_label_train[FEATURES].info()

        if False:
            df_train_with_label2[FEATURES].min().min()


# df_train_with_label2['D_39'] = df_train_with_label2.groupby('customer_ID')['D_39'].transform(lambda x: x.ffill())
        if True:
            for i,c in enumerate(FEATURES):
                # print ('fillna for ' + str(i) +'/' +str(cnt )+': ' + c)
                if c in cols_already_filled:
                    continue

                df_train_with_label2[c] = df_train_with_label2.groupby('customer_ID')[c].transform(lambda x: x.ffill())
                df_train_with_label2[c] = df_train_with_label2.groupby('customer_ID')[c].transform(lambda x: x.bfill())
                cols_already_filled.append(c)
            if False:
                print(df_train_with_label2[FEATURES].info(verbose=True, show_counts=True))
            if False:
                Beep()
            print('done filldown and up')
        for c in NON_NUM_COLS:
            df_train_with_label2[c] = label_encoder_model[c].transform(df_train_with_label2[c])


        df_train_with_label2[FEATURES] = df_train_with_label2[FEATURES] - df_train_with_label2[FEATURES].min() + 1

        if False:
            df_train_with_label2[FEATURES] = df_train_with_label2[FEATURES].fillna(0)

        if False:
            if True:
                col_with_nulls = df_train_with_label_train[FEATURES].isnull().sum()
                col_with_nulls = col_with_nulls[col_with_nulls>0]

                col_with_nulls = col_with_nulls.index.tolist()
            else:
                col_with_nulls = ['B_' + str(x) for x in [17,29,42]]
            FEATURES = [x for x in FEATURES if x not in col_with_nulls]

        if False:
            df_train_with_label2[CAT_COLS]
            df_train_with_label2['D_64'].unique()
            df_train_with_label2['D_63'].unique()

            print(df_train_with_label2['D_64'].unique())
            print(label_encoder.inverse_transform(df_train_with_label2['D_64'].unique()))

    #ffill then bfill for same customerID
    #very slow below compared to above
        if False:
            for id, new_df in df_train_with_label2.groupby ('customer_ID'):
                df_train_with_label2.loc[id,:] = df_train_with_label2.loc[id,:].fillna(method='ffill')
                df_train_with_label2.loc[id,:] =   df_train_with_label2.loc[id,:].fillna(method='bfill')
        # break
# print('done')
# FEATURES = ['D_47']

        df_train_with_label3 = df_train_with_label2.fillna(0)

        #now save the filw
        if SAVE_FILLED_DOWN_FILE:
            df_train_with_label3.to_csv(TRAIN_DATA_FILLED_DOWN_PATH,
                         header=header, mode='a')

            header = False

    else:
        df_train_with_label3 = chunk
        df_train_with_label3.S_2 = pd.to_datetime(df_train_with_label3.S_2)

        # df_train_with_label2.sort_values(['customer_ID', 'S_2'], inplace=True)

        df_train_with_label3.set_index(['customer_ID', 'S_2'], inplace=True)
        if i2 == 0:
            FEATURES = [x for x in df_train_with_label3.columns if x != 'target']  # and x != 'S_2']
            cnt = len(FEATURES)

    if i2 < TOTAL_TRAIN_BATCH:
        df_train_with_label_test = pd.concat([df_train_with_label_test,df_train_with_label3])
    if i2 == TOTAL_TRAIN_BATCH-1:
        X_test= df_train_with_label_test[FEATURES].to_numpy().reshape(-1,n_steps,len(FEATURES))
        y_test = df_train_with_label_test.groupby('customer_ID').mean()['target'].to_numpy()

        input_shape = X_test.shape[1:]

        if False and os.path.exists(_DIR + '/modelTransformers'):
            model = keras.models.load_model(_DIR + '/modelTransformers')
        else:
            model = build_model(
                input_shape,
                head_size=256,
                num_heads=4,
                ff_dim=4,
                num_transformer_blocks=4,
                mlp_units=[128],
                mlp_dropout=0.4,
                dropout=0.25,
            )

            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                metrics=["sparse_categorical_accuracy"],
            )
        if False:
            model.summary()
    else:
        df_train_with_label_train = df_train_with_label3
    # split customer_ID into test amd train
        if False:
            ids = df_train_with_label2.reset_index()['customer_ID'].unique()
            Id_train, Id_test, y_train, y_test = train_test_split(ids, ids, test_size=0.3, random_state=0)

            df_train_with_label_train = df_train_with_label3.reset_index().merge(pd.DataFrame(Id_train, columns=['customer_ID']), on='customer_ID').set_index(['customer_ID', 'no'])#, how='left')

        if False:
            df_train_with_label_train[FEATURES].iloc[:1000].to_csv(_DIR+'df_train_with_label_train.csv')

            df_train_with_label_test = df_train_with_label3.reset_index().merge(pd.DataFrame(Id_test, columns=['customer_ID']), on='customer_ID').set_index(['customer_ID', 'no'])#, how='left')

        X= df_train_with_label_train[FEATURES].to_numpy().reshape(-1,n_steps,len(FEATURES))

        y = df_train_with_label_train.groupby('customer_ID').mean()['target'].to_numpy()







        history = model.fit(
            X,
            y,
            validation_split=0.2,
            # epochs = 100,
            epochs=200,
            batch_size=64,
            callbacks=callbacks,
            # verbose = 0
        );print('done fitting')
        # if False:
        res = model.evaluate(X_test, y_test, verbose=1)
        y_predict = model.predict(X_test)
# y_predict = model.predict_classes(X_test)
#         yhat=np.round((y_predict[:,1]>=0.5)).astype(int).reshape(-1)
        yhat=y_predict[:,1].reshape(-1)
        if False:
            print(amex_metric_mod(y_test, yhat))

        y_test_df = pd.DataFrame(y_test, columns=['target'])
        yhat_df = pd.DataFrame(yhat, columns=['prediction'])
        score = amex_metric(y_test_df, yhat_df)
        print(score)

        results_df = pd.concat([results_df, pd.DataFrame(data=[[CHUNKSIZE, i2,  score]], columns=['chunksize','batch_no', 'score'])], ignore_index=True)
        results_df.to_csv(_DIR + 'score_history.csv', index=False)
        if False:
            Beep()
        x1 = 1
# amex_metric

        if False:
            ave_p2 = df_train_with_label.groupby('customer_ID').mean().rename(columns={'P_2': 'prediction'})
            train_labels = df_train_with_label.groupby('customer_ID').max()
            ave_p2['prediction'] = 1.0 - (ave_p2['prediction'] / ave_p2['prediction'].max())
            print(amex_metric(train_labels[['target']], ave_p2))
# model = Sequential()
# model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_steps, len(FEATURES))))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam')#, metrics=['accuracy'])
        if False:
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
        if False:
            pyplot.title('Learning Curves')
            pyplot.xlabel('Epoch')
            pyplot.ylabel('Cross Entropy')
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='val')
            pyplot.legend()
            pyplot.show()
    i2 += 1

x1=1
model.save(_DIR + '/modelTransformers_400k')
x1=1
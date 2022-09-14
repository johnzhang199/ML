#!/usr/bin/env python
# coding: utf-8

# # Time Series GRU TensorFlow Starter Notebook
# In this notebook we present starter code for a time series GRU model and starter code for processing Kaggle's 50GB CSV files into multiple saved NumPy files. Using a time series GRU allows us to use all the provided customer data and not just the customer's last data point. We published plots of time series data [here][1]. In this notebook we
# * Processes the train data from dataframes into 3D NumPy array of dimensions `num_of_customers x 13 x 188`
# * Save processed arrays as multiple NumPy files on disk
# * Next we build and train a GRU from the multiple files on disk
# * We compute validation score and achieve 0.787
# * Finally we process and save test data, infer test, and create a submission
# 
# It is important to note that you **do not need** to process the train and test files every time you run this notebook. Only process the data again when you engineer new features. Otherwise, upload your saved NumPy arrays to a Kaggle dataset (or use my Kaggle dataset [here][2]). Then as you customize and improve your GRU model, set the variable `PROCESS_DATA = False` and `PATH_TO_DATA = [the path to your kaggle dataset]`.
# 
# To view time series EDA which can help give you intuition about feature engineering and improving model architecture, see my other notebook [here][1]. Note in the code below, we partition the GPU into 8GB for RAPIDS (feature engineering) and 8GB for TensorFlow (model build and train).
# 
# [1]: https://www.kaggle.com/cdeotte/time-series-eda
# [2]: https://www.kaggle.com/datasets/cdeotte/amex-data-for-transformers-and-rnns

# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import tensorflow.keras.backend as K
print('Using TensorFlow version',tf.__version__)

# RESTRICT TENSORFLOW TO 8GB OF GPU RAM
# SO THAT WE HAVE 8GB RAM FOR RAPIDS
LIMIT = 8
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*LIMIT)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    print(e)
print('We will restrict TensorFlow to max %iGB GPU RAM'%LIMIT)
print('then RAPIDS can use %iGB GPU RAM'%(16-LIMIT))


# # Process Train Data
# We process both train and test data in chunks. We split train data into 10 parts and process each part separately and save to disk. We split test into 20 parts. This allows us to avoid memory errors during processing. We can also perform processing on GPU which is faster than CPU. Discussions about data preprocessing are [here][1] and [here][2]
# 
# [1]: https://www.kaggle.com/competitions/amex-default-prediction/discussion/327828
# [2]: https://www.kaggle.com/competitions/amex-default-prediction/discussion/328054

# In[ ]:


# LOADING JUST FIRST COLUMN OF TRAIN OR TEST IS SLOW
# INSTEAD YOU CAN LOAD FIRST COLUMN FROM MY DATASET
# OTHERWISE SET VARIABLE TO NONE TO LOAD FROM KAGGLE'S ORIGINAL DATAFRAME
PATH_TO_CUSTOMER_HASHES = \
    r'C:\John\git\vas\kaggle\americanExpress\amex-default-prediction/'

# AFTER PROCESSING DATA ONCE, UPLOAD TO KAGGLE DATASET
# THEN SET VARIABLE BELOW TO FALSE
# AND ATTACH DATASET TO NOTEBOOK AND PUT PATH TO DATASET BELOW
PROCESS_DATA = True
PATH_TO_DATA = './data/'
#PATH_TO_DATA = '../input/amex-data-for-transformers-and-rnns/data/'

# AFTER TRAINING MODEL, UPLOAD TO KAGGLE DATASET
# THEN SET VARIABLE BELOW TO FALSE
# AND ATTACH DATASET TO NOTEBOOK AND PUT PATH TO DATASET BELOW
TRAIN_MODEL = True
PATH_TO_MODEL = r'C:\John\git\vas\kaggle\americanExpress\model/'
#PATH_TO_MODEL = '../input/amex-data-for-transformers-and-rnns/model/'

INFER_TEST = True


# In[ ]:


import cupy#, cudf # GPU LIBRARIES
import numpy as np, pandas as pd # CPU LIBRARIES
import matplotlib.pyplot as plt, gc

_DIR = r'C:\John\git\vas\kaggle\americanExpress/'

# FILL NAN VALUE
NAN_VALUE = -127 # will fit in int8

from pyarrow.parquet import ParquetFile
import pyarrow as pa

hex_to_int = lambda x: int(x, 16)


def read_file(path='', usecols=None):
    # LOAD DATAFRAME
    if usecols is not None:
        df = pd.read_parquet(path, columns=usecols)
    else:
        # df = pd.read_parquet(path)
        pf = ParquetFile(path)
        first_ten_rows = next(pf.iter_batches(batch_size=10000))
        df = pa.Table.from_batches([first_ten_rows]).to_pandas()
    # REDUCE DTYPE FOR CUSTOMER AND DATE
    # df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df['customer_ID'] = df['customer_ID'].str[-16:].apply(hex_to_int).astype('int64')
    df.S_2 = pd.to_datetime(df.S_2)
    # SORT BY CUSTOMER AND DATE (so agg('last') works correctly)
    # df = df.sort_values(['customer_ID','S_2'])
    # df = df.reset_index(drop=True)
    # FILL NAN
    df = df.fillna(NAN_VALUE)
    print('shape of data:', df.shape)

    return df


print('Reading train data...')
TRAIN_PATH = _DIR + 'train.parquet'
train = read_file(path=TRAIN_PATH)
if PROCESS_DATA:
    # LOAD TARGETS
    targets = pd.read_csv(_DIR + 'train_labels.csv')
    targets['customer_ID'] = targets['customer_ID'].str[-16:].apply(hex_to_int).astype('int64')

    print(f'There are {targets.shape[0]} train targets')
    
    # GET TRAIN COLUMN NAMES
    # train = pd.read_csv('../input/amex-default-prediction/train_data.csv', nrows=1)
    T_COLS = train.columns
    print(f'There are {len(T_COLS)} train dataframe columns')
    
    # GET TRAIN CUSTOMER NAMES (use pandas to avoid memory error)
    # if PATH_TO_CUSTOMER_HASHES:
    #     train = pd.read_parquet(f'{PATH_TO_CUSTOMER_HASHES}train_customer_hashes.pqt')
    # else:
    #     train = pd.read_csv('/raid/Kaggle/amex/train_data.csv', usecols=['customer_ID'])
    #     train['customer_ID'] = train['customer_ID'].apply(lambda x: int(x[-16:],16) ).astype('int64')
    customers = train.drop_duplicates().sort_index().values.flatten()
    print(f'There are {len(customers)} unique customers in train.')


# In[ ]:


# CALCULATE SIZE OF EACH SEPARATE FILE
def get_rows(customers, train, NUM_FILES = 10, verbose = ''):
    chunk = len(customers)//NUM_FILES
    if verbose != '':
        print(f'We will split {verbose} data into {NUM_FILES} separate files.')
        print(f'There will be {chunk} customers in each file (except the last file).')
        print('Below are number of rows in each file:')
    rows = []

    for k in range(NUM_FILES):
        if k==NUM_FILES-1: cc = customers[k*chunk:]
        else: cc = customers[k*chunk:(k+1)*chunk]
        s = train.loc[train.customer_ID.isin(cc)].shape[0]
        rows.append(s)
    if verbose != '': print( rows )
    return rows

if PROCESS_DATA:
    NUM_FILES = 10
    rows = get_rows(customers, train, NUM_FILES = NUM_FILES, verbose = 'train')


# # Preprocess and Feature Engineering
# The function below processes the data. Discussions describing the process are [here][1] and [here][2]. Currently the code below uses [RAPIDS][3] and GPU to
# * Reduces memory usage of customer_ID column by converting to int64
# * Reduces memory usage of date time column (then deletes the column).
# * We fill NANs
# * Label encodes the categorical columns
# * We reduce memory usage dtypes of columns
# * Converts every customer into a 3D array with sequence length 13 and feature length 188
# 
# To improve this model, try adding new feautures. The columns have been rearanged to have the 11 categorical features first. This makes building the TensorFlow model later easier. We can also try adding Standard Scaler. Currently the data is being used without scaling from the original Kaggle train data. 
# 
# [1]: https://www.kaggle.com/competitions/amex-default-prediction/discussion/327828
# [2]: https://www.kaggle.com/competitions/amex-default-prediction/discussion/328054
# [3]: https://rapids.ai/

# In[ ]:


def feature_engineer(train, PAD_CUSTOMER_TO_13_ROWS = True, targets = None):
        
    # REDUCE STRING COLUMNS 
    # from 64 bytes to 8 bytes, and 10 bytes to 3 bytes respectively
    train['customer_ID'] = train['customer_ID'].str[-16:].apply(hex_to_int).astype('int64')
    train.S_2 = pd.to_datetime( train.S_2 )
    train['year'] = (train.S_2.dt.year-2000).astype('int8')
    train['month'] = (train.S_2.dt.month).astype('int8')
    train['day'] = (train.S_2.dt.day).astype('int8')
    del train['S_2']
        
    # LABEL ENCODE CAT COLUMNS (and reduce to 1 byte)
    # with 0: padding, 1: nan, 2,3,4,etc: values
    d_63_map = {'CL':2, 'CO':3, 'CR':4, 'XL':5, 'XM':6, 'XZ':7}
    train['D_63'] = train.D_63.map(d_63_map).fillna(1).astype('int8')

    d_64_map = {'-1':2,'O':3, 'R':4, 'U':5}
    train['D_64'] = train.D_64.map(d_64_map).fillna(1).astype('int8')
    
    CATS = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_66', 'D_68']
    OFFSETS = [2,1,2,2,3,2,3,2,2] #2 minus minimal value in full train csv
    # then 0 will be padding, 1 will be NAN, 2,3,4,etc will be values
    for c,s in zip(CATS,OFFSETS):
        train[c] = train[c] + s
        train[c] = train[c].fillna(1).astype('int8')
    CATS += ['D_63','D_64']
    
    # ADD NEW FEATURES HERE
    # EXAMPLE: train['feature_189'] = etc etc etc
    # EXAMPLE: train['feature_190'] = etc etc etc
    # IF CATEGORICAL, THEN ADD TO CATS WITH: CATS += ['feaure_190'] etc etc etc
    
    # REDUCE MEMORY DTYPE
    SKIP = ['customer_ID','year','month','day']
    for c in train.columns:
        if c in SKIP: continue
        if str( train[c].dtype )=='int64':
            train[c] = train[c].astype('int32')
        if str( train[c].dtype )=='float64':
            train[c] = train[c].astype('float32')
            
    # PAD ROWS SO EACH CUSTOMER HAS 13 ROWS
    if PAD_CUSTOMER_TO_13_ROWS:
        tmp = train[['customer_ID']].groupby('customer_ID').customer_ID.agg('count')
        more = np.array([],dtype='int64')
        for j in range(1,13):
            i = tmp.loc[tmp==j].index.values
            more = np.concatenate([more,np.repeat(i,13-j)])
        df = \
            train.iloc[:len(more)].copy().fillna(0)
        df = df * 0 - 1 #pad numerical columns with -1
        df[CATS] = (df[CATS] * 0).astype('int8') #pad categorical columns with 0
        df['customer_ID'] = more
        train = pd.concat([train,df],axis=0,ignore_index=True)
        
    # ADD TARGETS (and reduce to 1 byte)
    if targets is not None:
        train = train.merge(targets,on='customer_ID',how='left')
        train.target = train.target.astype('int8')
        
    # FILL NAN
    train = train.fillna(-0.5) #this applies to numerical columns
    
    # SORT BY CUSTOMER THEN DATE
    train = train.sort_values(['customer_ID','year','month','day']).reset_index(drop=True)
    train = train.drop(['year','month','day'],axis=1)
    
    # REARRANGE COLUMNS WITH 11 CATS FIRST
    COLS = list(train.columns[1:])
    COLS = ['customer_ID'] + CATS + [c for c in COLS if c not in CATS]
    train = train[COLS]
    
    return train


# In[ ]:


if PROCESS_DATA:
    # CREATE PROCESSED TRAIN FILES AND SAVE TO DISK        
    for k in range(NUM_FILES):

        # READ CHUNK OF TRAIN CSV FILE
        skip = int(np.sum( rows[:k] ) + 1) #the plus one is for skipping header
        train = pd.read_csv(_DIR+'train_data.csv', nrows=rows[k],
                              skiprows=skip, header=None, names=T_COLS)

        # FEATURE ENGINEER DATAFRAME
        train = feature_engineer(train, targets = targets)

        # SAVE FILES
        print(f'Train_File_{k+1} has {train.customer_ID.nunique()} customers and shape',train.shape)
        tar = train[['customer_ID','target']].drop_duplicates().sort_index()
        if not os.path.exists(PATH_TO_DATA): os.makedirs(PATH_TO_DATA)
        tar.to_parquet(f'{PATH_TO_DATA}targets_{k+1}.pqt',index=False)
        data = train.iloc[:,1:-1].values.reshape((-1,13,188))
        cupy.save(f'{PATH_TO_DATA}data_{k+1}',data.astype('float32'))

    # CLEAN MEMORY
    del train, tar, data
    del targets
    gc.collect()


# # Build Model
# We will just input the sequence data into a basic GRU. We will follow that we two dense layers and finally a sigmoid output to predict default. Try improving the model architecture.

# In[ ]:


# SIMPLE GRU MODEL
def build_model():
    
    # INPUT - FIRST 11 COLUMNS ARE CAT, NEXT 177 ARE NUMERIC
    inp = tf.keras.Input(shape=(13,188))
    embeddings = []
    for k in range(11):
        emb = tf.keras.layers.Embedding(10,4)
        embeddings.append( emb(inp[:,:,k]) )
    x = tf.keras.layers.Concatenate()([inp[:,:,11:]]+embeddings)
    
    # SIMPLE RNN BACKBONE
    x = tf.keras.layers.GRU(units=128, return_sequences=False)(x)
    x = tf.keras.layers.Dense(64,activation='relu')(x)
    x = tf.keras.layers.Dense(32,activation='relu')(x)
    
    # OUTPUT
    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    
    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer = opt)
    
    return model


# In[ ]:


# CUSTOM LEARNING SCHEUDLE
def lrfn(epoch):
    lr = [1e-3]*5 + [1e-4]*2 + [1e-5]*1
    return lr[epoch]
LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = False)


# # Competition Metric Code
# The code below is from Konstantin Yakovlev's discussion post [here][1]
# 
# [1]: https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534

# In[ ]:


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


# # Train Model
# We train 5 folds for 8 epochs each. We save the 5 fold models for test inference later. If you only want to infer without training, then set variable `TRAIN_MODEL = False` in the beginning of this notebook.

# In[ ]:


if TRAIN_MODEL:
    # SAVE TRUE AND OOF
    true = np.array([])
    oof = np.array([])
    VERBOSE = 2 # use 1 for interactive 

    for fold in range(5):

        # INDICES OF TRAIN AND VALID FOLDS
        valid_idx = [2*fold+1, 2*fold+2]
        train_idx = [x for x in [1,2,3,4,5,6,7,8,9,10] if x not in valid_idx]

        print('#'*25)
        print(f'### Fold {fold+1} with valid files', valid_idx)

        # READ TRAIN DATA FROM DISK
        X_train = []; y_train = []
        for k in train_idx:
            X_train.append( np.load(f'{PATH_TO_DATA}data_{k}.npy'))
            y_train.append( pd.read_parquet(f'{PATH_TO_DATA}targets_{k}.pqt') )
        X_train = np.concatenate(X_train,axis=0)
        y_train = pd.concat(y_train).target.values
        print('### Training data shapes', X_train.shape, y_train.shape)

        # READ VALID DATA FROM DISK
        X_valid = []; y_valid = []
        for k in valid_idx:
            X_valid.append( np.load(f'{PATH_TO_DATA}data_{k}.npy'))
            y_valid.append( pd.read_parquet(f'{PATH_TO_DATA}targets_{k}.pqt') )
        X_valid = np.concatenate(X_valid,axis=0)
        y_valid = pd.concat(y_valid).target.values
        print('### Validation data shapes', X_valid.shape, y_valid.shape)
        print('#'*25)

        # BUILD AND TRAIN MODEL
        K.clear_session()
        model = build_model()
        h = model.fit(X_train,y_train, 
                      validation_data = (X_valid,y_valid),
                      batch_size=512, epochs=8, verbose=VERBOSE,
                      callbacks = [LR])
        if not os.path.exists(PATH_TO_MODEL): os.makedirs(PATH_TO_MODEL)
        model.save_weights(f'{PATH_TO_MODEL}gru_fold_{fold+1}.h5')

        # INFER VALID DATA
        print('Inferring validation data...')
        p = model.predict(X_valid, batch_size=512, verbose=VERBOSE).flatten()

        print()
        print(f'Fold {fold+1} CV=', amex_metric_mod(y_valid, p) )
        print()
        true = np.concatenate([true, y_valid])
        oof = np.concatenate([oof, p])
        
        # CLEAN MEMORY
        del model, X_train, y_train, X_valid, y_valid, p
        gc.collect()

    # PRINT OVERALL RESULTS
    print('#'*25)
    print(f'Overall CV =', amex_metric_mod(true, oof) )
    K.clear_session()


# # Process Test Data
# We process the test data in the same way as train data.

# In[ ]:


if PROCESS_DATA:
    # GET TEST COLUMN NAMES
    test = pd.read_csv(_DIR+'test_data.csv', nrows=1)
    T_COLS = test.columns
    print(f'There are {len(T_COLS)} test dataframe columns')
    
    # GET TEST CUSTOMER NAMES (use pandas to avoid memory error)
    # if PATH_TO_CUSTOMER_HASHES:
    #     test = pd.read_parquet(f'{PATH_TO_CUSTOMER_HASHES}test_customer_hashes.pqt')
    # else:
    # test = pd.read_csv('/raid/Kaggle/amex/test_data.csv', usecols=['customer_ID'])
    test['customer_ID'] = test['customer_ID'].apply(lambda x: int(x[-16:],16) ).astype('int64')
    customers = test.drop_duplicates().sort_index().values.flatten()
    print(f'There are {len(customers)} unique customers in test.')


# In[ ]:


NUM_FILES = 20
if PROCESS_DATA:
    # CALCULATE SIZE OF EACH SEPARATE FILE
    rows = get_rows(customers, test, NUM_FILES = NUM_FILES, verbose = 'test')


# In[ ]:


if PROCESS_DATA:
    # SAVE TEST CUSTOMERS INDEX
    test_customer_hashes = cupy.array([],dtype='int64')
    
    # CREATE PROCESSED TEST FILES AND SAVE TO DISK
    for k in range(NUM_FILES):

        # READ CHUNK OF TEST CSV FILE
        skip = int(np.sum( rows[:k] ) + 1) #the plus one is for skipping header
        test = pd.read_csv(_DIR+'test_data.csv', nrows=rows[k],
                              skiprows=skip, header=None, names=T_COLS)

        # FEATURE ENGINEER DATAFRAME
        test = feature_engineer(test, targets = None)
        
        # SAVE TEST CUSTOMERS INDEX
        cust = test[['customer_ID']].drop_duplicates().sort_index().values.flatten()
        test_customer_hashes = cupy.concatenate([test_customer_hashes,cust])

        # SAVE FILES
        print(f'Test_File_{k+1} has {test.customer_ID.nunique()} customers and shape',test.shape)
        data = test.iloc[:,1:].values.reshape((-1,13,188))
        cupy.save(f'{PATH_TO_DATA}test_data_{k+1}',data.astype('float32'))
        
    # SAVE CUSTOMER INDEX OF ALL TEST FILES
    cupy.save(f'{PATH_TO_DATA}test_hashes_data', test_customer_hashes)

    # CLEAN MEMORY
    del test, data
    gc.collect()


# # Infer Test Data
# We infer the test data from our saved fold models. If you don't wish to infer test but you only want your notebook to compute a validation score to evaluate model changes, then set variable `INFER_TEST = False` in the beginning of this notebook. Also if you wish to infer from previously trained models, then add the path to the Kaggle dataset in the variable `PATH_TO_MODEL` in the beginning of this notebook.

# In[ ]:


if INFER_TEST:
    # INFER TEST DATA
    start = 0; end = 0
    sub = pd.read_csv('../input/amex-default-prediction/sample_submission.csv')
    
    # REARANGE SUB ROWS TO MATCH PROCESSED TEST FILES
    sub['hash'] = sub['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    test_hash_index = cupy.load(f'{PATH_TO_DATA}test_hashes_data.npy')
    sub = sub.set_index('hash').loc[test_hash_index].reset_index(drop=True)
    
    for k in range(NUM_FILES):
        # BUILD MODEL
        K.clear_session()
        model = build_model()
        
        # LOAD TEST DATA
        print(f'Inferring Test_File_{k+1}')
        X_test = np.load(f'{PATH_TO_DATA}test_data_{k+1}.npy')
        end = start + X_test.shape[0]

        # INFER 5 FOLD MODELS
        model.load_weights(f'{PATH_TO_MODEL}gru_fold_1.h5')
        p = model.predict(X_test, batch_size=512, verbose=0).flatten() 
        for j in range(1,5):
            model.load_weights(f'{PATH_TO_MODEL}gru_fold_{j+1}.h5')
            p += model.predict(X_test, batch_size=512, verbose=0).flatten()
        p /= 5.0

        # SAVE TEST PREDICTIONS
        sub.loc[start:end-1,'prediction'] = p
        start = end
        
        # CLEAN MEMORY
        del model, X_test, p
        gc.collect()


# # Create Submission

# In[ ]:


if INFER_TEST:
    sub.to_csv('submission.csv',index=False)
    print('Submission file shape is', sub.shape )
    display( sub.head() )


# In[ ]:


if INFER_TEST:
    # DISPLAY SUBMISSION PREDICTIONS
    plt.hist(sub.to_pandas().prediction, bins=100)
    plt.title('Test Predictions')
    plt.show()


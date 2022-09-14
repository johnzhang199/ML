df_train_with_label2.to_csv(_DIR+'df_train_with_label2.csv')

df_train_with_label2.to_csv(_DIR+'df_train_with_label2_filldown_up.csv')

df_train_with_label2.to_csv(_DIR+'df_train_with_label2_filldown_up_labelEncoded_above0.csv')

df_train_with_label3.to_csv(_DIR+'df_train_with_label2_filldown_up_labelEncoded_above0_fill0.csv')


ids13 = pd.DataFrame(data=customers_with_13_rows.reset_index()['customer_ID'].unique(), columns=['customer_ID'])
df_train_with_label_train = df_train_with_label_train.reset_index().merge(ids13, left_on = 'customer_ID', right_on= 'customer_ID').set_index(['customer_ID','no'])
df_train_with_label_test = df_train_with_label_test.reset_index().merge(ids13, left_on = 'customer_ID', right_on= 'customer_ID').set_index(['customer_ID','no'])



df = pd.read_csv(_DIR + 'train_data.csv', usecols=NON_NUM_COLS)
# df.info()
# array(['CR', 'CO', 'CL', 'XM', 'XZ', 'XL'], dtype=object), test and train
df['D_63'].unique()
#array([nan, 'U', 'O', 'R'], dtype=object) for D_64 test
# array(['O', 'R', nan, 'U', '-1'], dtype=object) train

df['D_64'].unique()

df['D_63'].unique()
#array([nan, 'U', 'O', 'R'], dtype=object) for D_64 test
# array(['O', 'R', nan, 'U', '-1'], dtype=object) train

df['D_64'].unique()


df_train_with_label2['D_63'].unique()
df_train_with_label2['D_64'].unique()



np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

label_encoder_model2 = dict()
if os.path.exists(_DIR + 'label_encoder_model_D_63.npy'):
    for c in NON_NUM_COLS:
        _file_path = _DIR + 'label_encoder_model_' + c + '.npy'
        label_encoder_model2[c] = LabelEncoder()

        label_encoder_model2[c].classes_ = np.load(_file_path)
# restore np.load for future normal usage
np.load = np_load_old

for c in NON_NUM_COLS:
    df_train_with_label2[c] = label_encoder_model2[c].transform(df_train_with_label2[c])

    label_encoder_model2[c].transform(df_train_with_label2[c])


# model = ...  # Get model (Sequential, Functional Model, or Model subclass)
#2,530,000
model.save(_DIR + '/modelTransformers')

reconstructed_model = keras.models.load_model(_DIR + '/modelTransformers')
        history = reconstructed_model.fit(
            X,
            y,
            validation_split=0.2,
            epochs = 100,
            # epochs=200,
            batch_size=64,
            callbacks=callbacks,
            verbose = 0
        );print('done fitting')
        # if False:
        res = reconstructed_model.evaluate(X_test, y_test, verbose=1)
        y_predict = reconstructed_model.predict(X_test)
# y_predict = model.predict_classes(X_test)
#         yhat=np.round((y_predict[:,1]>=0.5)).astype(int).reshape(-1)
        yhat=y_predict[:,1].reshape(-1)
        if False:
            print(amex_metric_mod(y_test, yhat))

        y_test_df = pd.DataFrame(y_test, columns=['target'])
        yhat_df = pd.DataFrame(yhat, columns=['prediction'])
        print(amex_metric(y_test_df, yhat_df))


if os.path.exists(_DIR + '/modelTransformers'):
    model = keras.models.load_model(_DIR + '/modelTransformers')

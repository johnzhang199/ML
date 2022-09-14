import pandas as pd

#function to us to convert categorial col's to onehot format.
def add_dummies(df, cat_cols):
    for c in cat_cols:
        df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1)
        df = df.drop(c, axis=1)
    return df


pd.set_option('display.max_columns', None)


path = r'C:\John\job\2022\migo\migo_interview_dataset_20220316.csv/migo_interview_dataset_20220316.csv'

df = pd.read_csv(path)

df = add_dummies(df, ['channel', 'state'])



pass

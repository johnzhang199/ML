#!/usr/bin/env python
# coding: utf-8

# based on https://www.kaggle.com/code/cdeotte/time-series-eda

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# LOAD LIBRARIES
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# LOAD TRAIN DATA AND MERGE TARGETS ONTO FEATURES
_DIR = r'C:\John\git\vas\kaggle\americanExpress/'
df = pd.read_csv(_DIR + 'train_data.csv', nrows=100_000)
df.S_2 = pd.to_datetime(df.S_2)
df2 = pd.read_csv(_DIR + 'train_labels.csv')
df = df.merge(df2,on='customer_ID',how='left')


# In[4]:


def plot_time_series(prefix='D', cols=None, display_ct=32):
    
    # DETERMINE WHICH COLUMNS TO PLOT
    if cols is not None and len(cols)==0: cols = None
    if cols is None:
        COLS = df.columns[2:-1]
        COLS = np.sort( [int(x[2:]) for x in COLS if x[0]==prefix] )
        COLS = [f'{prefix}_{x}' for x in COLS]
        print('#'*25)
        print(f'Plotting all {len(COLS)} columns with prefix {prefix}')
        print('#'*25)
    else:
        COLS = [f'{prefix}_{x}' for x in cols]
        print('#'*25)
        print(f'Plotting {len(COLS)} columns with prefix {prefix}')
        print('#'*25)

    # ITERATE COLUMNS
    for c in COLS:

        # CONVERT DATAFRAME INTO SERIES WITH COLUMN
        tmp = df[['customer_ID','S_2',c,'target']].copy()
        tmp2 = tmp.groupby(['customer_ID','target'])[['S_2',c]].agg(list).reset_index()
        tmp3 = tmp2.loc[tmp2.target==1]
        tmp4 = tmp2.loc[tmp2.target==0]

        # FORMAT PLOT
        spec = gridspec.GridSpec(ncols=2, nrows=1,
                             width_ratios=[3, 1], wspace=0.1,
                             hspace=0.5, height_ratios=[1])
        fig = plt.figure(figsize=(20,10))
        ax0 = fig.add_subplot(spec[0])

        # PLOT 32 DEFAULT CUSTOMERS AND 32 NON-DEFAULT CUSTOMERS
        t0 = []; t1 = []
        for k in range(display_ct):
            try:
                # PLOT DEFAULTING CUSTOMERS
                row = tmp3.iloc[k]
                ax0.plot([x.to_pydatetime() for x in row.S_2],row[c],'-o',color='blue')
                t1 += row[c]
                # PLOT NON-DEFAULT CUSTOMERS
                row = tmp4.iloc[k]
                ax0.plot(row.S_2,row[c],'-o',color='orange')
                t0 += row[c]
            except:
                pass
        plt.title(f'Feature {c} (Key: BLUE=DEFAULT, orange=no default)',size=18)

        # PLOT HISTOGRAMS
        ax1 = fig.add_subplot(spec[1])
        try:
            # COMPUTE BINS
            t = t0+t1; mn = np.nanmin(t); mx = np.nanmax(t)
            if mx==mn:
                mx += 0.01; mn -= 0.01
            bins = np.arange(mn,mx+(mx-mn)/20,(mx-mn)/20 )
            # PLOT HISTOGRAMS
            if np.sum(np.isnan(t1))!=len(t1):
                ax1.hist(t1,bins=bins,orientation="horizontal",alpha = 0.8,color='blue')
            if np.sum(np.isnan(t0))!=len(t0):
                ax1.hist(t0,bins=bins,orientation="horizontal",alpha = 0.8,color='orange')
        except:
            pass
        plt.show()


# In[5]:


# LEAVE LIST BLANK TO PLOT ALL
plot_time_series('D',[39,41,47,45,46,48,54,59,61,62,75,96,105,112,124])


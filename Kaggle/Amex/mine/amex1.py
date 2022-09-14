#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[43]:


#if SAMPLE, then only read first few rows to save time
SAMPLE = True
NROWS = 1000
_DIR = '/kaggle/input/amex-default-prediction/'
PATH_sample_submission = _DIR + 'sample_submission.csv'
PATH_train_data = _DIR + 'train_data.csv'
PATH_test_data = _DIR + 'test_data.csv'
PATH_train_labels = _DIR + 'train_labels.csv'
if SAMPLE:
    df_submission = pd.read_csv(PATH_sample_submission, nrows = NROWS)
    df_train = pd.read_csv(PATH_train_data, nrows = NROWS)
    df_test = pd.read_csv(PATH_test_data, nrows = NROWS)
    df_labels = pd.read_csv(PATH_train_labels, nrows = NROWS)
if True:
    print(df_submission.head())
    print(df_train.head())
    print(df_test.head())
    print(df_labels.head())


# In[43]:


combined = [df_submission, df_train, df_test, df_labels]
for d in combined:
    columns = d.columns.tolist()
    columns.sort()
    d = d[columns]


# In[43]:


df_submission


# In[43]:


max_columns = pd.get_option('display.max_columns')
if True:
    pd.set_option('display.max_columns', None)
    print ('submission')
    print(df_submission.head())
    print ('*' * 100)
    print ('train')
    print(df_train.head())
    print ('*' * 100)
    print ('test')
    print(df_test.head())
    print ('*' * 100)
    print ('label')
    print(df_labels.head())
pd.set_option('display.max_columns', max_columns)


# In[3]:


print(df_labels.head())


# In[4]:


# print(df_labels.mean())


# In[5]:


# print(df_train.describe())


# In[6]:


# filter_col = [col for col in df_train if col.startswith('D_')]
# filter_col.sort(key=lambda x: int(x[2:]))
# print(filter_col)
# 


# In[7]:


df_train_segmented = dict()
PREFIXES = ['D','S','P','B','R']
for p in PREFIXES:
    filter_col = [col for col in df_train if col.startswith(p+'_')]
    filter_col.sort(key=lambda x: int(x[2:]))
#     print(filter_col)
    df_train_segmented[p] = df_train[filter_col]


# In[30]:


# print(df_train_segmented['D'])
# print(df_train_segmented)
# print('test')
CAT_COLS = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']


# In[8]:


# df_describe = df_train[filter_col].describe()
# print(df_train[filter_col].median())
# df_train[filter_col].describe()[['mean']].plot()


# In[9]:


# df_describe.loc['mean',:].plot()


# In[10]:


# df_train[filter_col].median().plot()


# In[11]:





# In[35]:


#IMPORTANT: Intentionally plotted different ways for learning purposes only. 

#optional plotting w/pandas: https://pandas.pydata.org/pandas-docs/stable/visualization.html

#we will use matplotlib.pyplot: https://matplotlib.org/api/pyplot_api.html

#to organize our graphics will use figure: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure
#subplot: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot
#and subplotS: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html?highlight=matplotlib%20pyplot%20subplots#matplotlib.pyplot.subplots

#graph distribution of quantitative data
plt.figure(figsize=[26,12])

# i = 0
for p in PREFIXES:
    # plt.subplot(231)
    # plt.boxplot(x=df_train_segmented[PREFIXES[i]], showmeans = True, meanline = True)
    cols_not_cat = [c for c in df_train_segmented[p] if c not in CAT_COLS]
    plt.boxplot(x=df_train_segmented[p][cols_not_cat], showmeans = True, meanline = True,
               labels = cols_not_cat)
    # plt.boxplot(x=df_train_segmented[PREFIXES[i]][['D_39','D_41']], showmeans = True, meanline = True)
    # plt.boxplot(x=df_train_segmented[PREFIXES[i]][['D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48',
    #        'D_49', 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55', 'D_56', 'D_58',
    #        'D_59', 'D_60', 'D_61', 'D_62', 'D_63', 'D_64', 'D_65', 'D_66', 'D_68',
    #        'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_76', 'D_77',
    #        'D_78', 'D_79', 'D_80', 'D_81', 'D_82', 'D_83', 'D_84', 'D_86', 'D_87',
    #        'D_88', 'D_89', 'D_91', 'D_92', 'D_93', 'D_94', 'D_96', 'D_102',
    #        'D_103', 'D_104', 'D_105', 'D_106', 'D_107', 'D_108', 'D_109', 'D_110',
    #        'D_111', 'D_112', 'D_113', 'D_114', 'D_115', 'D_116', 'D_117', 'D_118',
    #        'D_119', 'D_120', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126',
    #        'D_127', 'D_128', 'D_129', 'D_130', 'D_131', 'D_132', 'D_133', 'D_134',
    #        'D_135', 'D_136', 'D_137', 'D_138', 'D_139', 'D_140', 'D_141', 'D_142',
    #        'D_143', 'D_144', 'D_145']], showmeans = True, meanline = True)

    plt.title(p)
# plt.ylabel('Fare ($)')
if False:


    plt.subplot(232)
    plt.boxplot(data1['Age'], showmeans = True, meanline = True)
    plt.title('Age Boxplot')
    plt.ylabel('Age (Years)')

    plt.subplot(233)
    plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)
    plt.title('Family Size Boxplot')
    plt.ylabel('Family Size (#)')

    plt.subplot(234)
    plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 
             stacked=True, color = ['g','r'],label = ['Survived','Dead'])
    plt.title('Fare Histogram by Survival')
    plt.xlabel('Fare ($)')
    plt.ylabel('# of Passengers')
    plt.legend()

    plt.subplot(235)
    plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 
             stacked=True, color = ['g','r'],label = ['Survived','Dead'])
    plt.title('Age Histogram by Survival')
    plt.xlabel('Age (Years)')
    plt.ylabel('# of Passengers')
    plt.legend()

    plt.subplot(236)
    plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 
             stacked=True, color = ['g','r'],label = ['Survived','Dead'])
    plt.title('Family Size Histogram by Survival')
    plt.xlabel('Family Size (#)')
    plt.ylabel('# of Passengers')
    plt.legend()


# In[36]:


df_train_segmented['S']


# In[28]:


df_train_segmented[PREFIXES[i]].columns


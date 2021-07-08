# -*- coding: utf-8 -*-
"""
@filename: OutlierHandling
@dataset: outliers.csv
@author: cyruslentin
"""

# hides all warnings
import warnings
warnings.filterwarnings('ignore')

# imports
import pandas as pd
import numpy as np
import utils

# read dataset
df = pd.read_csv('./data/outliers.csv')

# create master copy
dfm = df.copy()

###########################################################
# replace outliers with nulls
###########################################################

# info
print("\n*** Structure ***")
print(df.info())

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier index
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers with nulls
lol, uol = utils.OutlierLimits(df['icol'])
print("Lower Outlier Limit:",lol)
print("Upper Outlier Limit:",uol)
df['icol'] = np.where(df['icol'] < lol, None, df['icol'])
df['icol'] = np.where(df['icol'] > uol, None, df['icol'])

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier index
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))


###########################################################
# replace outliers with upper mean or median
###########################################################

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier index
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers with nulls
lol, uol = utils.OutlierLimits(df['jcol'])
print("Lower Outlier Limit:",lol)
print("Upper Outlier Limit:",uol)
df['jcol'] = np.where(df['jcol'] < lol, None, df['jcol'])
df['jcol'] = np.where(df['jcol'] > uol, None, df['jcol'])

# now fill nas with mean or median
print(df['jcol'].mean())
df['jcol'] = df['jcol'].fillna(df['jcol'].mean())
print(df['jcol'].mean())
# convert to int if column was originally int
df['jcol'] = df['jcol'].astype(int)

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier index
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))


###########################################################
# replace outliers with upper limit / lower
###########################################################

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier index
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers with nulls
lol, uol = utils.OutlierLimits(df['kcol'])
print("Lower Outlier Limit:",lol)
print("Upper Outlier Limit:",uol)
df['kcol'] = np.where(df['kcol'] < lol, lol, df['kcol'])
df['kcol'] = np.where(df['kcol'] > uol, uol, df['kcol'])
# convert to int if column was originally int
#df['kcol'] = df['kcol'].astype(int)

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier index
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

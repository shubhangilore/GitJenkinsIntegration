# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 00:18:39 2018
@filename: ClassificationKnn
@dataset: Iris
@author: cyruslentin
"""

# hides all warnings
import warnings
warnings.filterwarnings('ignore')

# imports
import pandas as pd
import utils

# read dataset
df = pd.read_csv('./data/Iris.csv')

# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())

# drop Id
# change as required
print("\n*** Transformation ***")
df = df.drop('Id', axis=1)
# store class variable  
# change as required
clsVars = "Species"
print("Done ...")

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# get unique Species names
print("\n*** Unique Species - Categoric Alpha***")
lnLabels = df[clsVars].unique()
print(lnLabels)

# convert string / categoric to numeric
print("\n*** Unique Species - Categoric Numeric ***")
df[clsVars] = pd.Categorical(df[clsVars])
df[clsVars] = df[clsVars].cat.codes
lnCCodes = df[clsVars].unique()
print(lnCCodes)

# master df
dfm = df.copy()


# Data Rescaling
# Your preprocessed data may contain attributes with a mixtures of scales for 
# various quantities such as dollars, kilograms and sales volume.

# Many machine learning methods expect or are more effective if the data attributes 
# have the same scale. 
# Two popular data scaling methods are normalization and standardization.

# Data Normalization
# Normalization refers to rescaling real valued numeric attributes into the range 0 and 1.
# It is useful to scale the input attributes for a model that relies on the 
# magnitude of values, such as distance measures used in k-nearest neighbors 
# and in the preparation of coefficients in regression.
#
# The MinMaxScaler transforms features by scaling each feature to a given range. 
# This range can be set by specifying the feature_range parameter (default at (0,1))
#
# manually
# normalized = (x-min(x))/(max(x)-min(x))

# preparing for normalization / standadrization
df = dfm.copy()

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev
print('\n*** StdDev In Columns ***')
print(df.std())

# normalize data
print('\n*** Normalize Data ***')
df = utils.NormalizeData(df, clsVars)
print('Done ...')

# check variance
print('*** Variance In Columns ***')
print(df.var())

# check std dev
print('\n*** StdDev In Columns ***')
print(df.std())


# Data Standardization
# Standardization refers to shifting the distribution of each attribute to have 
# a mean of zero and a standard deviation of one (unit variance).
# 
# It is useful to standardize attributes for a model that relies on the 
# distribution of attributes such as Gaussian processes.
#
# Sklearn its main scaler, the StandardScaler, uses a strict definition of standardization 
# to standardize data. It purely centers the data by using the below formula
#
# manually
# standardized = (x-mean(x))/stdev(x)

# preparing for normalization / standadrization
df = dfm.copy()

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev
print('\n*** StdDev In Columns ***')
print(df.std())

# standardize data
print('\n*** Standardize Data ***')
df = utils.StandardizeData(df, clsVars)
print('Done ...')

# check variance
print('*** Variance In Columns ***')
print(df.var())

# check std dev
print('\n*** StdDev In Columns ***')
print(df.std())

# Absoulute Scaled
# The MaxAbsScaler works very similarly to the MinMaxScaler but automatically 
# scales the data to a [-1,1] range based on the absolute maximum. This scaler 
# is meant for data that is already centered at zero or sparse data. It does not 
# shift/center the data, and thus does not destroy any sparsity.

# manually
# scaled = x / max(abs(x))

# preparing for normalization / standadrization
df = dfm.copy()

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev
print('\n*** StdDev In Columns ***')
print(df.std())

# MaxAbsScaledData
print('\n*** MaxAbsScaledData Data ***')
df = utils.MaxAbsScaledData(df, clsVars)
print('Done ...')

# check variance
print('*** Variance In Columns ***')
print(df.var())

# check std dev
print('\n*** StdDev In Columns ***')
print(df.std())

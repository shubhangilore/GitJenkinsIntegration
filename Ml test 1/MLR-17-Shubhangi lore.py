#!/usr/bin/env python
# coding: utf-8

# In[164]:


# hides all warnings
import warnings
warnings.filterwarnings('ignore')

# imports
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
# seaborn
import seaborn as sns
# utils
import utils


# In[165]:


df=pd.read_csv('california-housing.csv')


# In[166]:


##############################################################
# Exploratory Data Analytics
##############################################################

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


# In[167]:


##############################################################
# Dependent Variable 
##############################################################

# store dep variable  
# change as required
depVars = 'median_house_value'
print("\n*** Dep Vars ***")
print(depVars)


# In[168]:


##############################################################
# Data Transformation
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
df = df.drop('ocean_proximity', axis=1)
print("Done ...")


# In[169]:


# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))


# In[170]:


# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())


# In[171]:


# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required


# In[172]:


df = df.drop('educated', axis=1)


# In[173]:


print(df['total_rooms'].mean())
print(df['total_rooms'].median())


# In[174]:


print(df['total_bedrooms'].mean())
print(df['total_bedrooms'].median())


# In[175]:


df['total_rooms'].fillna(df['total_rooms'].median(),inplace=True)
df['total_bedrooms'].fillna(df['total_bedrooms'].median(),inplace=True)


# In[176]:


df['median_income'] = df['median_income'].interpolate(method ='linear', limit_direction ='forward') 


# In[177]:


# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 


# In[178]:


# check mean
print('\n*** Mean In Columns ***')
print(df.mean())


# In[179]:


df.std()


# In[180]:


var=df.columns.tolist()


# In[181]:


var=['longitude',
 'latitude',
 'housing_median_age',
 'median_income',
 'random_income',]


# In[182]:


# normalize data not doing this because model has no effect
print('\n*** Normalize Data ***')
#df = utils.NormalizeData(df,var )
print('Done ...')


# In[183]:


df.std()


# In[184]:


# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.3f}'.format
df.corr().style.background_gradient(cmap='RdYlGn',axis=None)


# In[185]:


#handling mutlicorr
df=df.drop(['total_rooms','households'],axis=1)


# In[186]:


df=df.drop(['latitude'],axis=1)


# In[187]:


# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.3f}'.format
df.corr().style.background_gradient(cmap='RdYlGn',axis=None)


# In[188]:


##############################################################
# Visual Data Analytics
##############################################################

# check relation with corelation - heatmap
print("\n*** Heat Map ***")
plt.figure(figsize=(8,8))
ax = sns.heatmap(df.corr(), annot=True, cmap="PiYG")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()

# boxplot
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
for colName in colNames:
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# histograms
# https://www.qimacros.com/histogram-excel/how-to-determine-histogram-bin-interval/
# plot histograms
print('\n*** Histograms ***')
colNames = df.columns.tolist()
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# scatterplots
# plot Sscatterplot
print('\n*** Scatterplot ***')
colNames = df.columns.tolist()
colNames.remove(depVars)
print(colName)
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.regplot(data=df, x=depVars, y=colName, color= 'b', scatter_kws={"s": 5})
    plt.title(depVars + ' v/s ' + colName)
    plt.show()


# In[189]:


###############################
# Split Train & Test
###############################

# split into data & target
print("\n*** Prepare Data ***")
dfTrain = df.sample(frac=0.8, random_state=707)
dfTest=df.drop(dfTrain.index)
print("Train Count:",len(dfTrain.index))
print("Test Count :",len(dfTest.index))


# In[190]:


##############################################################
# Model Creation & Fitting 
##############################################################

# all cols except dep var 
print("\n*** Regression Data ***")
allCols = dfTrain.columns.tolist()
print(allCols)
allCols.remove(depVars)
print(allCols)
print("Done ...")

# regression summary for feature
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(dfTrain[allCols])
y = dfTrain[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())


# In[191]:


# remove columns with p-value > 0.05
# change as required
print("\n*** Drop Cols ***")
allCols.remove('longitude')
allCols.remove('random_income')
print(allCols)


# In[192]:


# regression summary for feature
print("\n*** Regression Summary Again ***")
X = sm.add_constant(dfTrain[allCols])
y = dfTrain[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())


# In[193]:


# train data
print("\n*** Regression Data For Train ***")
X_train = dfTrain[allCols].values
y_train = dfTrain[depVars].values
# print
print(X_train.shape)
print(y_train.shape)
print(type(X_train))
print(type(y_train))
print("Done ...")

# test data
print("\n*** Regression Data For Test ***")
X_test = dfTest[allCols].values
y_test = dfTest[depVars].values
print(X_test.shape)
print(y_test.shape)
print(type(X_test))
print(type(y_test))
print("Done ...")


# In[194]:


###############################
# Auto Select Best Regression
###############################

# imports 
print("\n*** Import Regression Libraries ***")
# normal linear regression
from sklearn.linear_model import LinearRegression 
# ridge regression from sklearn library 
from sklearn.linear_model import Ridge 
# import Lasso regression from sklearn library 
from sklearn.linear_model import Lasso 
# import model 
from sklearn.linear_model import ElasticNet 
print("Done ...")
  
# empty lists
print("\n*** Init Empty Lists ***")
lModels = []
lModelAdjR2 = []
lModelRmses = []
lModelScInd = []
lModelCoefs = []
print("Done ...")

# list model name list
print("\n*** Init Models Lists ***")
lModels.append(("LinearRegression", LinearRegression()))
lModels.append(("RidgeRegression ", Ridge(alpha = 10)))
lModels.append(("LassoRegression ", Lasso(alpha = 1)))
lModels.append(("ElasticNet      ", ElasticNet(alpha = 1)))
print("Done ...")

# iterate through the models list
for vModelName, oModelObject in lModels:
    # create model object
    model = oModelObject
    # print model vals
    print("\n*** "+vModelName)
    # fit or train the model
    model.fit(X_train, y_train) 
    # predict train set 
    y_pred = model.predict(X_train)
    dfTrain[vModelName] = y_pred
    # predict test set 
    y_pred = model.predict(X_test)
    dfTest[vModelName] = y_pred
    # r-square  
    from sklearn.metrics import r2_score
    r2 = r2_score(dfTrain[depVars], dfTrain[vModelName])
    print("R-Square:",r2)
    # adj r-square  
    adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
              (X_train.shape[0] - X_train.shape[1] - 1)))
    lModelAdjR2.append(adj_r2)
    print("Adj R-Square:",adj_r2)
    # mae 
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(dfTest[depVars], dfTest[vModelName])
    print("MAE:",mae)
    # mse 
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(dfTest[depVars], dfTest[vModelName])
    print("MSE:",mse)
    # rmse 
    rmse = np.sqrt(mse)
    lModelRmses.append(rmse)
    print("RMSE:",rmse)
    # scatter index
    si = rmse/dfTest[depVars].mean()
    lModelScInd.append(si)
    print("SI:",si)

# print key metrics for each model
msg = "%10s %16s %10s %10s" % ("Model Type", "AdjR2", "RMSE", "SI")
print(msg)
for i in range(0,len(lModels)):
    # print accuracy mean & std
    msg = "%16s %10.3f %10.3f %10.3f" % (lModels[i][0], lModelAdjR2[i], lModelRmses[i], lModelScInd[i])
    print(msg)


# find model with best adj-r2 & print details
print("\n*** Best Model ***")
vBMIndex = lModelAdjR2.index(max(lModelAdjR2))
print("Index       : ",vBMIndex)
print("Model Name  : ",lModels[vBMIndex][0])
print("Adj-R-Sq    : ",lModelAdjR2[vBMIndex])
print("RMSE        : ",lModelRmses[vBMIndex])
print("ScatterIndex: ",lModelScInd[vBMIndex])


# In[195]:


##############################################################
# predict from new data 
##############################################################

# create model from full dataset
print(allCols)

# regression summary for feature
print("\n*** Regression Summary Again ***")
X = sm.add_constant(df[allCols])
y = df[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# now create linear regression model
print("\n*** Regression Model ***")
X = df[allCols].values
y = df[depVars].values
model = lModels[vBMIndex][1]
model.fit(X,y)
print(model)

# read dataset
dfp = pd.read_csv('california-housing-prd.csv')

print("\n*** Structure ***")
print(dfp.info())

# drop cols 
# change as required
print("\n*** Drop Cols ***")
dfp = dfp.drop('ocean_proximity', axis=1)
print("Done ...")
print("None ... ")

# transformation
# change as required
print("\n*** Transformation ***")
dfp=dfp.drop(['total_rooms','households'],axis=1)
print("None ... ")


# split X & y
print("\n*** Split Predict Data ***")
X_pred = dfp[allCols].values
print(X_pred)

# predict
print("\n*** Predict Data ***")
p_pred = model.predict(X_pred)
dfp['predict'] = p_pred
print("Done ... ")


# In[196]:


# predicted value we can find the rmse and si because we dont have previous data
p_pred


# In[197]:


# save data to file
dfp.to_csv("prd-17-shubhangi lore.csv", index=False)


# In[ ]:





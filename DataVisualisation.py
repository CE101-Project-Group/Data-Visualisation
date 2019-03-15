# importing the necessary modules to visualise
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy.stats import norm
from scipy import stats
warnings.filterwarnings("ignore")
plt.show()

"""
this will load in the files to train the program, to test the program
and to have a look at some basic mathematical information about the SalePrice
of housing
"""
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train['SalePrice'].describe()
print(train.head(20)) # look at the first 20 entries in the train dataset
print(test.head(20)) # look at the first 20 entries in the test dataset

# drop the id column from the data as they do not have any correlation with the data whatsoever
trainID = train["Id"]
testID = test["Id"]
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

"""
basicInformation()
scatterPlots()
boxPlots()
histogramPlots()
missingData()
corrMatrix()
"""

def basicInformation():
    # graph to look for SalePrice distribution, skewness and its peak value
    sns.distplot(train['SalePrice'])
    print(f"Skewness of graph: {train['SalePrice'].skew()}")
    print(f"Kurtosis (peak value) of graph: {train['SalePrice'].kurt()}")

def scatterPlots():
    # scatter plot of grlivarea/saleprice
    var = 'GrLivArea'
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

    # scatter plot of totalbsmtsf/saleprice
    var = 'TotalBsmtSF'
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

    # scatterplot
    sns.set()
    columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(train[columns], size = 2.5)
    plt.show()

def boxPlots();
    # box plot of overallqual/saleprice
    var = 'OverallQual'
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)

    # box plot of YearBuilt/SalePrice
    var = 'YearBuilt'
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)

def histogramPlots():

    # histogram and normal probability plot of SalePrice
    sns.distplot(train['SalePrice'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(train['SalePrice'], plot=plt)
    plt.show()

    # applying log transformation
    train['SalePrice'] = np.log(train['SalePrice'])

    # transformed histogram and normal probability plot
    sns.distplot(train['SalePrice'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(train['SalePrice'], plot=plt)
    plt.show()

    # histogram and normal probability plot
    sns.distplot(train['GrLivArea'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(train['GrLivArea'], plot=plt)
    plt.show()
    
    # data transformation
    train['GrLivArea'] = np.log(train['GrLivArea'])

    # transformed histogram and normal probability plot
    sns.distplot(train['GrLivArea'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(train['GrLivArea'], plot=plt)

    # histogram and normal probability plot
    sns.distplot(train['TotalBsmtSF'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(train['TotalBsmtSF'], plot=plt)

    #transform data
    train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

    #histogram and normal probability plot
    sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

def missingData():
    # finding missing data within the data set
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(20)

def corrMatrix():
    # correlation matrix of all attributes
    corrmat = train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.9, square=True)
    plt.show()
# Importing all the libraries
import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)

features = pd.read_csv('data/features.csv.zip')
train = pd.read_csv('data/train.csv.zip')
stores = pd.read_csv('data/stores.csv')
test = pd.read_csv('data/test.csv.zip')
cartd = pd.read_csv('data/cartdata.csv')
sample_submission = pd.read_csv('data/sampleSubmission.csv.zip')


def print_head():
    print(features.head())
    print("------------------------------------------------------------\n")
    print(stores.head())
    print("------------------------------------------------------------\n")
    print(train.head())
    print("------------------------------------------------------------\n")
    print(test.head())
    print("------------------------------------------------------------\n")
    print(sample_submission.head())


def data_info():
    print(features.dtypes)
    print("------------------------------------------------------------\n")
    print(train.dtypes)
    print("------------------------------------------------------------\n")
    print(stores.dtypes)
    print("------------------------------------------------------------\n")
    print(test.dtypes)


def merge_data():
    global train, test
    feature_store = features.merge(stores, how='inner', on="Store")
    train = train.merge(feature_store, how='inner', on=['Store', 'Date', 'IsHoliday'])
    test = test.merge(feature_store, how='inner', on=['Store', 'Date', 'IsHoliday'])


def split_date():
    global train, test
    train = train.copy()
    test = test.copy()

    train['Date'] = pd.to_datetime(train['Date'])
    train['Year'] = pd.to_datetime(train['Date']).dt.year
    train['Month'] = pd.to_datetime(train['Date']).dt.month
    train['Week'] = pd.to_datetime(train['Date']).dt.week
    train['Day'] = pd.to_datetime(train['Date']).dt.day
    train.replace({'A': 1, 'B': 2, 'C': 3}, inplace=True)

    test['Date'] = pd.to_datetime(test['Date'])
    test['Year'] = pd.to_datetime(test['Date']).dt.year
    test['Month'] = pd.to_datetime(test['Date']).dt.month
    test['Week'] = pd.to_datetime(test['Date']).dt.week
    test['Day'] = pd.to_datetime(test['Date']).dt.day
    test.replace({'A': 1, 'B': 2, 'C': 3}, inplace=True)
    # print(train.head())
    # print("------------------------------------------------------------\n")
    # print(test.head())


def weekly_sales_plot():
    weekly_sales2010 = train.loc[train['Year'] == 2010].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})
    print(weekly_sales2010)
    weekly_sales2011 = train.loc[train['Year'] == 2011].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})
    weekly_sales2012 = train.loc[train['Year'] == 2012].groupby(['Week']).agg({'Weekly_Sales': ['mean', 'median']})
    plt.figure(figsize=(20, 7))
    sns.lineplot(weekly_sales2010['Weekly_Sales']['mean'].index, weekly_sales2010['Weekly_Sales']['mean'].values)
    sns.lineplot(weekly_sales2011['Weekly_Sales']['mean'].index, weekly_sales2011['Weekly_Sales']['mean'].values)
    sns.lineplot(weekly_sales2012['Weekly_Sales']['mean'].index, weekly_sales2012['Weekly_Sales']['mean'].values)

    plt.grid()
    plt.xticks(np.arange(1, 53, step=1))
    plt.legend(['2010', '2011', '2012'])
    plt.show()


def kp_plot():
    cd = cartd.copy()

    cd['date'] = pd.to_datetime(cd['date'])
    cd['year'] = pd.to_datetime(cd['date']).dt.year
    cd['month'] = pd.to_datetime(cd['date']).dt.month
    cd['week'] = pd.to_datetime(cd['date']).dt.week
    cd['day'] = pd.to_datetime(cd['date']).dt.day
    cd.replace({'A': 1, 'B': 2, 'C': 3}, inplace=True)

    weekly_sales2018 = cd.loc[cd['year'] == 2018].groupby(['week']).agg({'grand_total': 'sum'})
    weekly_sales2019 = cd.loc[cd['year'] == 2019].groupby(['week']).agg({'grand_total': 'sum'})
    weekly_sales2020 = cd.loc[cd['year'] == 2020].groupby(['week']).agg({'grand_total': 'sum'})
    weekly_sales2021 = cd.loc[cd['year'] == 2021].groupby(['week']).agg({'grand_total': 'sum'})
    print(weekly_sales2021['grand_total'])
    plt.figure(figsize=(20, 7))
    sns.lineplot(weekly_sales2018['grand_total'].index, weekly_sales2018['grand_total'].values)
    sns.lineplot(weekly_sales2019['grand_total'].index, weekly_sales2019['grand_total'].values)
    sns.lineplot(weekly_sales2020['grand_total'].index, weekly_sales2020['grand_total'].values)
    sns.lineplot(weekly_sales2021['grand_total'].index, weekly_sales2021['grand_total'].values)

    plt.grid()
    plt.xticks(np.arange(1, 53, step=1))
    plt.ticklabel_format(style='plain', axis='y')
    plt.legend(['2018', '2019', '2020', '2021'])
    plt.show()


def clean_data():
    global train
    Y_train = train['Weekly_Sales']
    targets = Y_train.copy()
    train = train.drop(['Weekly_Sales'], axis=1)
    # Let's also identify the numeric and categorical columns.
    numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train.select_dtypes('object').columns.tolist()
    print(numeric_cols)
    print("------------------------------------------------------------\n")
    print(categorical_cols)
    # Check if there is any null value in train dataframe
    train.isnull().sum()
    # Check if there is any null value test in dataframe
    test.isnull().sum()


if __name__ == '__main__':
    # print_head()
    # data_info()
    merge_data()
    split_date()
    weekly_sales_plot()
    # TODO
    # Data Visualization and Descriptive Statics

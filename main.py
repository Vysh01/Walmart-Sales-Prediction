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
numeric_cols = None
X_train = None
X_test = None
targets = None
train_inputs = None
val_inputs = None
train_targets = None
val_targets = None


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


def clean_impute_data():
    global train, Y_train, targets, numeric_cols, train, test
    Y_train = train['Weekly_Sales']
    targets = Y_train.copy()
    train = train.drop(['Weekly_Sales'], axis=1)
    # Let's also identify the numeric and categorical columns.
    numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train.select_dtypes('object').columns.tolist()
    # print(numeric_cols)
    # print("------------------------------------------------------------\n")
    # print(categorical_cols)
    # Check if there is any null value in train dataframe
    # print(train.isnull().sum())
    # Check if there is any null value test in dataframe
    # print(test.isnull().sum())

    imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
    imputer.fit(train[numeric_cols])
    train[numeric_cols] = imputer.transform(train[numeric_cols])

    print(train.isnull().sum())
    from sklearn.preprocessing import MinMaxScaler
    # Create the scaler
    scaler = MinMaxScaler()
    # Fit the scaler to the numeric columns
    scaler.fit(train[numeric_cols])
    # Transform and replace the numeric columns
    train[numeric_cols] = scaler.transform(train[numeric_cols])
    print(train[numeric_cols].describe().loc[['min', 'max']])
    # 'Date' is irrelevant and Drop it from data.
    train = train.drop(['Date'], axis=1)
    test = test.drop(['Date'], axis=1)


def prepare_dataset():
    # Preparing the dataset:
    global train_inputs, val_inputs, train_targets, val_targets, X_train, X_test
    X_train = train[['Store', 'Dept', 'IsHoliday', 'Size', 'Week', 'Type', 'Year']]
    X_test = test[['Store', 'Dept', 'IsHoliday', 'Size', 'Week', 'Type', 'Year']]
    print(X_train.columns)
    print(X_test.columns)

    # Splitting and training
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(X_train, Y_train, test_size=0.25,
                                                                            random_state=42)


def prediction_xgbregressor():
    global X_test
    # importing XGBRegressor
    from xgboost import XGBRegressor

    # fitting the model
    model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth=4)
    model.fit(train_inputs, train_targets)

    # Finding out importance of features
    # print(X_test.head())
    # importance_df = pd.DataFrame({
    #     'feature': X_test.columns,
    #     'importance': model.feature_importances_
    # }).sort_values('importance', ascending=False)

    # plt.figure(figsize=(10, 6))
    # plt.title('Feature Importance')
    # sns.barplot(data=importance_df.head(10), x='importance', y='feature')
    # plt.show()
    # Make and evaluate predictions:

    x_pred = model.predict(train_inputs)
    x_preds = model.predict(val_inputs)
    print('XGB TRAIN RMSE: {}'.format(mean_squared_error(x_pred, train_targets, squared=False)))
    print('XGB TEST RMSE: {}'.format(mean_squared_error(x_preds, val_targets, squared=False)))


def prediction_rfr():
    model = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=50, max_depth=20, min_samples_split=3,
                                  max_features=0.4, min_samples_leaf=4).fit(train_inputs, train_targets)
    x_pred = model.predict(train_inputs)
    x_preds = model.predict(val_inputs)

    print('RFR 1 TRAIN RMSE: {}'.format(mean_squared_error(x_pred, train_targets, squared=False)))
    print('RFR 1 TEST RMSE: {}'.format(mean_squared_error(x_preds, val_targets, squared=False)))

    model = RandomForestRegressor(n_estimators=58, max_depth=27, min_samples_split=3,
                                  max_features=6).fit(train_inputs, train_targets)
    x_pred = model.predict(train_inputs)
    x_preds = model.predict(val_inputs)

    print('RFR 2 TRAIN RMSE: {}'.format(mean_squared_error(x_pred, train_targets, squared=False)))
    print('RFR 2 TEST RMSE: {}'.format(mean_squared_error(x_preds, val_targets, squared=False)))


if __name__ == '__main__':
    # print_head()
    # data_info()
    merge_data()
    split_date()
    # weekly_sales_plot()
    clean_impute_data()
    prepare_dataset()
    prediction_xgbregressor()
    prediction_rfr()

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


def perform_test(model):
    le = LabelEncoder()
    one = OneHotEncoder()
    test = pd.read_csv('Dataset/test.csv')
    test.City = le.fit_transform(test.City)
    a = pd.DataFrame(one.fit_transform(test[['City Group']]).toarray())
    b = pd.DataFrame(one.fit_transform(test[['Type']]).toarray())
    test = pd.concat([test, a, b], axis=1)
    test.drop(['Type', 'City Group'], axis=1, inplace=True)

    def year(x):
        o = int(x[6:])
        return o

    def month(x):
        o = int(x[3:5])
        return o

    def day(x):
        o = int(x[0:2])
        return o

    test['year'] = test['Open Date'].apply(year)
    test['month'] = test['Open Date'].apply(month)
    test['day'] = test['Open Date'].apply(day)
    test.drop(["Open Date", "P14", "P15", "P16", "P17", "P18", "P24", "P25", "P26", "P30", "P31",
                  "P32", "P33", "P34", "P35", "P36", "P37"], axis=1, inplace=True)

    test.columns = ['Id', 'City', 'P1', 'P2', 'P3', 'P4',
                    'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
                    'P11', 'P12', 'P13', 'P19', 'P20', 'P21', 'P22',
                    'P23', 'P27', 'P28',
                    'P29', 'revenue', 0, 1,
                    2, 3, 4, 'year', 'month', 'day']

    pred = test.drop(['Id', 'revenue'], axis=1)
    yp = model.predict(pred)
    y1 = pd.DataFrame(yp)
    test = pd.concat([test, y1], axis=1)
    test.columns = ['Id', 'City', 'P1', 'P2', 'P3', 'P4',
                    'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
                    'P11', 'P12', 'P13', 'P19', 'P20', 'P21', 'P22',
                    'P23', 'P27', 'P28',
                    'P29', 'revenue', 0, 1,
                    2, 3, 4, 'year', 'month', 'day',
                    'Prediction']
    test.to_csv('sub.csv', columns=["Id", "Prediction"], index=False)


def internal_regression(X_train, Y_train):
    rr = RandomForestRegressor(n_estimators=1000)
    model = rr.fit(X_train, Y_train)
    return model


def read_and_clean_data():
    dataset = pd.read_csv('Dataset/train.csv')

    le = LabelEncoder()
    one = OneHotEncoder()

    dataset.City = le.fit_transform(dataset.City)
    a = pd.DataFrame(one.fit_transform(dataset[['City Group']]).toarray())
    b = pd.DataFrame(one.fit_transform(dataset[['Type']]).toarray())
    dataset = pd.concat([dataset, a, b], axis=1)
    dataset.drop(['Type', 'City Group'], axis=1, inplace=True)

    print(dataset.head())

    def year(x):
        o = int(x[6:])
        return o

    def month(x):
        o = int(x[3:5])
        return o

    def day(x):
        o = int(x[0:2])
        return o

    dataset['year'] = dataset['Open Date'].apply(year)
    dataset['month'] = dataset['Open Date'].apply(month)
    dataset['day'] = dataset['Open Date'].apply(day)
    dataset.drop(["Open Date", "P14", "P15", "P16", "P17", "P18", "P24", "P25", "P26", "P30", "P31",
                  "P32", "P33", "P34", "P35", "P36", "P37"], axis=1, inplace=True)
    print(dataset.head())

    dataset.columns = ['Id', 'City', 'P1', 'P2', 'P3', 'P4',
                    'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
                    'P11', 'P12', 'P13', 'P19', 'P20', 'P21', 'P22',
                    'P23', 'P27', 'P28',
                    'P29', 'revenue', 0, 1,
                    2, 3, 4, 'year', 'month', 'day']

    x = dataset.drop(["Id", "revenue"], axis=1)
    y = dataset["revenue"]

    model = internal_regression(x, y)
    perform_test(model)


if __name__ == '__main__':
    read_and_clean_data()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder


def fix_missing_data(dataset):
    # creates a instance of class
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    dataset[:, 2:3] = imputer.fit(dataset[:, 2:3]).transform(dataset[:, 2:3])
    return dataset


def house_price():
    dataset = pd.read_csv('Dataset/train.csv')
    print(dataset.head())
    dataset = dataset[["MSSubClass", "Neighborhood", "OverallQual", "LotFrontage", "LotArea", "GrLivArea", "GarageArea", "SalePrice"]]
    print(dataset.head())

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    one = OneHotEncoder()
    neighborhood = one.fit_transform(pd.DataFrame(x[:, 1])).toarray()
    # utilities = one.fit_transform(pd.DataFrame(x[:, 2])).toarray()
    print(neighborhood)
    # print(utilities)

    # x = np.append(x, utilities, axis=1)
    x = np.append(x, neighborhood, axis=1)

    xdf = pd.DataFrame(x)
    print(xdf)
    # xdf.drop(xdf.columns[1], axis=1, inplace=True)
    xdf.drop(xdf.columns[1], axis=1, inplace=True)
    print(xdf)
    x = fix_missing_data(xdf.to_numpy())

    regression = RandomForestClassifier(n_estimators=221, random_state=1)
    regression.fit(x, y)

    dataset_test = pd.read_csv('Dataset/test.csv')
    dataset_test = dataset_test[["MSSubClass", "Neighborhood", "OverallQual", "LotFrontage", "LotArea", "GrLivArea", "GarageArea"]]
    dataset_x = dataset_test.iloc[:, :].values
    print(dataset_x)

    one_test = OneHotEncoder()
    neighborhood_test = one_test.fit_transform(pd.DataFrame(dataset_x[:, 1])).toarray()
    dataset_x = np.append(dataset_x, neighborhood_test, axis=1)
    print(dataset_x)
    dataset_x = pd.DataFrame(dataset_x)
    dataset_x.drop(dataset_x.columns[1], axis=1, inplace=True)
    print(dataset_x)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    dataset_x = dataset_x.to_numpy()

    dataset_x[:, 1:-1] = imputer.fit(dataset_x[:, 1:-1]).transform(dataset_x[:, 1:-1])
    prediction = regression.predict(dataset_x)
    t = pd.DataFrame(prediction, columns=["Prediction"])
    print(t)
    t.to_csv('sub.csv', columns=["Prediction"], index=False)


if __name__ == '__main__':
    house_price()

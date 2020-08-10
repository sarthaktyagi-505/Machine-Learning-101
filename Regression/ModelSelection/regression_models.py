import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def multiple_linear_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print('-----------------------------------')
    y_pred = lr.predict(x_test)
    print(r2_score(y_test, y_pred))


def polynomial_linear_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    pf = PolynomialFeatures(degree=5)
    x_train_trans = pf.fit_transform(x_train)
    x_test_trans = pf.transform(x_test)
    lr = LinearRegression()
    lr.fit(x_train_trans, y_train)
    print('-----------------------------------')
    y_pred = lr.predict(x_test_trans)
    print(r2_score(y_test, y_pred))


def support_vector_regression(x, y):
    y = y.reshape(len(y), 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    y_train = sc_y.fit_transform(y_train)
    svr = SVR(kernel='rbf')
    svr.fit(x_train, y_train)
    print('-----------------------------------')
    y_pred = sc_y.inverse_transform(svr.predict(sc_x.transform(x_test)))
    print(r2_score(y_test, y_pred))


def decision_tree_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    dr = DecisionTreeRegressor(random_state=3)
    dr.fit(x_train, y_train)
    print('-----------------------------------')
    y_pred = dr.predict(x_test)
    print(r2_score(y_test, y_pred))


def random_forest_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    rr = RandomForestRegressor(n_estimators=10, random_state=0)
    rr.fit(x_train, y_train)
    print('-----------------------------------')
    y_pred = rr.predict(x_test)
    print(r2_score(y_test, y_pred))


def read_data():
    dataset = pd.read_csv('Data.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Multiple Linear Regression
    multiple_linear_regression(x, y)

    # Polynomial Linear Regression
    polynomial_linear_regression(x, y)

    # Support Vector Regression
    support_vector_regression(x, y)

    # Decision Tree Regression
    decision_tree_regression(x, y)

    # Random Forest Regression
    random_forest_regression(x, y)


if __name__ == '__main__':
    read_data()

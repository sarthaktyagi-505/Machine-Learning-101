import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def clean_and_load_data():
    dataset = pd.read_csv('Dataset/Position_Salaries.csv')
    print(dataset.head(5))

    X = dataset.iloc[:, 1:-1].values
    Y = dataset.iloc[:, -1].values

    lr = LinearRegression()
    lr.fit(X, Y)

    # Polynomial Linear Regression
    # Create matrix of features specifically for powers
    pf = PolynomialFeatures(degree=4)
    X_trans = pf.fit_transform(X)
    lr_poly = LinearRegression()
    lr_poly.fit(X_trans, Y)

    # predicting single salary
    pred_linear = lr.predict([[6.5]])
    pred_poly = lr_poly.predict(pf.fit_transform([[6.5]]))

    print(pred_linear)
    print(pred_poly)


if __name__  == '__main__':
    clean_and_load_data()
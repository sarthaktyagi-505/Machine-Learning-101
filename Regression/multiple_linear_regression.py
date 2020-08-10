import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def backward_elimination(X, Y):
    # Backward elimination code
    # add ones to feature vector to represent x0 from the multiple linear regression equation

    X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

    # x_opt is vector of optimal features
    X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)

    # fit the new model with all the possible predictors
    regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()

    X_opt = np.array(X[:, [0,4]], dtype=float)
    X_train, X_test, Y_train, Y_test = train_test_split(X_opt, Y, test_size=0.2, random_state=0)
    regressor_ols = sm.OLS(endog=Y_train, exog=X_train).fit()
    Y_opt_pred = regressor_ols.predict(X_test)

    print(np.mean(np.abs((Y_test - Y_opt_pred) / Y_test)) * 100)


def regression():
    dataset = pd.read_csv('Dataset/50_Startups.csv')
    print(dataset.head(5))
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    print(np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100)

    backward_elimination(X, Y)


if __name__ == '__main__':
    regression()
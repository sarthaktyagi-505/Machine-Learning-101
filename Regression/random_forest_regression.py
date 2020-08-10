import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def random_forest_regression():
    dataset = pd.read_csv('Dataset/Position_Salaries.csv')
    X = dataset.iloc[:, 1:-1].values
    Y = dataset.iloc[:, -1].values
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X, Y)
    print(regressor.predict([[6.5]]))


if __name__ == '__main__':
    random_forest_regression()
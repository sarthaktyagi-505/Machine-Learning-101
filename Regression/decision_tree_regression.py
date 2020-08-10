import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def decision_tree_regression():
    dataset = pd.read_csv('Dataset/Position_Salaries.csv')
    X = dataset.iloc[:, 1:-1].values
    Y = dataset.iloc[:, -1].values
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X, Y)

    prediction = regressor.predict([[6.5]])
    print(prediction)


if __name__ == '__main__':
    decision_tree_regression()

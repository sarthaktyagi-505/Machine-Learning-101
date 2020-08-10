import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def regression():
    # read data from CSV
    dataset = pd.read_csv('Dataset/Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    regression = LinearRegression()
    regression.fit(X_train, Y_train)

    # Predicting the test dataset
    print(Y_test)
    Y_pred = regression.predict(X_test)

    plt.scatter(X_train, Y_train, color='red')
    plt.plot(X_train, regression.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Training Set)')
    plt.xlabel('years of work ex')
    plt.ylabel('Salary')
    plt.show()

    plt.scatter(X_test, Y_test, color='red')
    plt.plot(X_train, regression.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Testing Set)')
    plt.xlabel('years of work ex')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    regression()

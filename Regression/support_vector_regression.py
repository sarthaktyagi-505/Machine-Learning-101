import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# Support Vector Regression
def svr():
    dataset = pd.read_csv('Dataset/Position_Salaries.csv')
    X = dataset.iloc[:, 1:-1].values
    Y = dataset.iloc[:, -1].values
    # print(X)
    Y = Y.reshape(len(Y), 1)
    # print(Y)

    # Feature scaling is required for SVR since it has implicit relation with the data
    # Standardisation brings the values of vector in range of -3 to 3
    scx = StandardScaler()
    scy = StandardScaler()
    X = scx.fit_transform(X)
    Y = scy.fit_transform(Y)

    # Train the model using the SVR
    # Kernel is Radial Basic Function
    svr = SVR(kernel='rbf')
    svr.fit(X, Y)

    # Predict the value
    # Now we have to reverse the Scaling
    prediction = svr.predict(scx.transform([[6.5]]))
    print(scy.inverse_transform(prediction))


if __name__ == '__main__':
    svr()
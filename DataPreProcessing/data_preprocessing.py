import numpy as np                                            # allows us to work with Arrays
import pandas as pd                                           # allows us to import dataset and create matrix of features etc
from sklearn.impute import SimpleImputer                      # allows us to replace the salary by average salary
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder               # creates a one hot encoding vector
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def fix_missing_data(dataset):
    # creates a instance of class
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # exclude the strings columns
    # apply imputer to dataset, fit method which calculates the mean
    # transform actually replaces the data
    dataset[:, 1:3] = imputer.fit(dataset[:, 1:3]).transform(dataset[:, 1:3])
    return dataset


def pre_process_data():
    dataset = pd.read_csv('DataSets/Data.csv')

    # differentiate features from Dependant variable  factor
    # if you wanna take all the rows just put : which takes all the rows
    # :-1 ends up meaning : is all the columns and -1 means remove the last column
    # -1 is the index of last column

    # Features/Independant Matrix of features
    X = dataset.iloc[:, :-1].values

    # Dependant variable factor(usually the last column in a datset)
    Y = dataset.iloc[:, -1].values

    # fix the missing data
    X = fix_missing_data(X)

    # encode string data using one hot encoding, this data can be something which can be categorised
    # Use 2 classes first one columntransformer class, onehotencoder from sklearn lib
    # transform= enter kind of encoding and column to transform
    # remainder= passthrough means which will not onehotencoded, if we dont give passthrough first 3 columns will be
    # onehotencoded

    # create a instance of columnTransformer class, what columns we want to transform and what not to
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

    # convert normal array to numpy array
    X = np.array(ct.fit_transform(X))

    #   label encoder, it doesnt have to be numpy array, something for binary encoding
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    # Splitting dataset into training set and test set
    # test_test_split splits the dataset
    # split size needs to be passed
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # Feature Scaling, Use Standardisation on X_train and Y_train
    sc = StandardScaler()

    # Notice the fit has already been called so we dont call it again on test dataset since we want
    # both of them to have same scale, other we would be feeding test data to differently scaled input.
    X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
    X_test[:, 3:] = sc.transform(X_test[:, 3:])

    print(X_train)
    print("------")
    print(X_test)
    print("------")


if __name__ == '__main__':
    pre_process_data()

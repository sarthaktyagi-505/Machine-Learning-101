import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('Dataset/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Using Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(accuracy_score(y_test, y_pred))

# Using KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print(accuracy_score(y_test, y_pred))

# Using kernel SVM
svc = SVC(kernel='rbf', random_state=0)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

print(accuracy_score(y_test, y_pred))

# Using Random Forest Classifier
rc = RandomForestClassifier(n_estimators=100, random_state=0)
rc.fit(x_train, y_train)
y_pred = rc.predict(x_test)

print(accuracy_score(y_test, y_pred))
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# there is no dependant variable when are trying to classify using K means clustering.
# Remove features which are not important to classify

dataset = pd.read_csv('Dataset/Mall_Customers.csv')
dataset = dataset.drop("CustomerID", axis=1);
dataset = dataset.drop("Age", axis=1);
dataset = dataset.drop("Genre", axis=1);
x = dataset.to_numpy();
print(x)

# Using elbow method to get number of clusters

wss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wss.append(kmeans.inertia_)

# plt.plot(range(1, 11), wss)
# plt.title("k means")
# plt.xlabel("Clusters")
# plt.ylabel("WCSS")
# plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)

plt.scatter(x[y_kmeans==0, 0], x[y_kmeans==0, 1], s=100, c='red', label ='cluster1')
plt.scatter(x[y_kmeans==1, 0], x[y_kmeans==1, 1], s=100, c='blue', label ='cluster2')
plt.scatter(x[y_kmeans==2, 0], x[y_kmeans==2, 1], s=100, c='green', label ='cluster3')
plt.scatter(x[y_kmeans==3, 0], x[y_kmeans==3, 1], s=100, c='cyan', label ='cluster4')
plt.scatter(x[y_kmeans==4, 0], x[y_kmeans==4, 1], s=100, c='magenta', label ='cluster5')
plt.xlabel("Annual Income $k")
plt.ylabel("Spending Score")

plt.show()










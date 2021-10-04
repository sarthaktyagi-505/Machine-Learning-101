# Machine-Learning-101
# Classficiation
Unlike classification where we predict a continuous number, we use classification to classify the the objects in to different category. It finds its use cases in a lot of medical and marketing fields.

Some main Classification problems are:
1. Logistic Regression
2. K nearest Neighbors
3. Support Vector Machine
4. Kernel SVM
5. Naive Bayes
6. Decision Tree Classification
7. Random Forest Classification
8. 
## Logistic Regression:

[image:FC053676-C461-494D-9EEC-D470354DEE2D-30485-00008A3AC5702301/Screenshot 2020-08-12 at 5.56.23 PM.png]

Applying Linear regression to classification problem.


We stage a likelihood of the person taking an offer. 

Linear Regression at least gives us the range  of people taking the offer. The line in between 1 and 0 makes sense but not the line above or below it.

<img width="690" alt="Screenshot 2020-08-12 at 5 56 23 PM" src="https://user-images.githubusercontent.com/50817904/135798785-26dec48b-6d17-43e0-bd37-09ac0e04ac2d.png">



We cut of the lines from top and below.
Apply a sigmoid function to /y = mx +c/


### Logistic Regression
### log(p/p-1) = b0 + b1x

[image:373A385E-309C-433E-8C26-BEC1B2D01390-30485-00008A8AB1FB07A9/Screenshot 2020-08-12 at 6.02.06 PM.png]


An example with various age groups›



[image:79417CAC-977F-49AE-9C63-D3B7154F6B4B-30485-00008B0FCA400033/Screenshot 2020-08-12 at 6.11.38 PM.png]



We use the probability to have a score. But what if we don’t want probability and ask for prediction.
[image:5686C86C-D96C-44D1-B15E-04409A2E82EF-30485-00008B3377364F87/Screenshot 2020-08-12 at 6.14.11 PM.png]

Anything having probability of less than 0.5 its projected to 0 and anything above is projected upward.

## K Nearest Neighbor:
 

[image:DCAF3709-E5C4-430F-8574-2D865BAB1314-30485-00008FFE5625631C/Screenshot 2020-08-13 at 6.32.17 PM.png]



Rule Guide:
1. Choose the number K of neighbours, default values is 5.
2. Take Manhattan distance or euclidian distance.
3. Count the data points in different category.
4. Assign new data point to the category where you counted most neighbours.



[image:30E3D415-3524-4579-9633-9515923D9DCC-30485-000090287C5FFD49/Screenshot 2020-08-13 at 6.35.18 PM.png]

Euclidean distance

[image:4C8E39E4-8F2D-4B28-AEEC-ABBC7213731C-30485-0000902FE37F3F2D/Screenshot 2020-08-13 at 6.35.49 PM.png]

Take Euclidean distance from 5 points and assign the category.


## Support Vector Machine
[image:47E86BE7-C19C-44E8-9E23-02A282F9CB9D-30485-00009135BDB597EB/Screenshot 2020-08-13 at 6.54.34 PM.png]

[image:19572C5D-90C1-4E51-8777-BAE6519B1AD6-30485-0000914ADADA85B6/Screenshot 2020-08-13 at 6.56.05 PM.png]


SVM tries to pick extreme cases of categories which is risky. If we are differentiating between apples and oranges most of algorithms will look at only most common features SVM looks at the boundary conditions and tries to create a Line to separate them.


[image:00BB686D-F9B1-4C14-B066-C57ADDD6AC63-30485-0000918F8E11C675/Screenshot 2020-08-13 at 7.01.00 PM.png]


## Kernel SVM


What if the Data cannot be separated Linearly. In that case we use Kernel SVM

[image:919D88DA-06EB-471C-9D33-2A06D199B4D6-30485-0000923B7045F1AD/Screenshot 2020-08-13 at 7.35.59 PM.png]


Map data into Linearly separable dataset using Higher Dimension.

[image:C98D2FEE-58D9-4860-B9E8-9458E5D7BDF4-30485-00009278E0266066/Screenshot 2020-08-13 at 7.40.23 PM.png]

Post applying some function, Hyperplane separates the data.

[image:9477AEA2-ED9F-49A0-A7D4-120B43E548EE-30485-0000928E572FBFA9/Screenshot 2020-08-13 at 7.41.56 PM.png]

### Mapping to higher dimension it can lead to more compute power being required

## The Kernel Trick

[image:D9CFC91F-3905-4306-9C0B-326DA39D0BF0-30485-000092C68FE862CF/Screenshot 2020-08-13 at 7.45.56 PM.png]


If landmark is large then we get a value very close to zero. If the landmark is closer to reference point it be smaller and e^0 is 1.


[image:0544F0F5-1B26-4297-91E3-30806DF1AADF-30485-0000930F53E894CA/Screenshot 2020-08-13 at 7.51.09 PM.png]

We use the kernel to separate our data. 
[image:63D5022A-B011-4AFE-872D-5CF2BB4B2A28-30485-000093210F58C61E/Screenshot 2020-08-13 at 7.52.26 PM.png]

Anything out side circle will assigned 0, anything outside will be 1.
Sigma defines how wide can the circumference of cycle can be. By finding the right sigma we find the distinction.

*Types of kernel function:*

[image:1C0B3628-65B3-4239-ACB1-48D984069961-30485-000093E2EB95706F/Screenshot 2020-08-13 at 8.06.18 PM.png]

 
## Non Linear SVR

[image:42256DCF-FAC0-48A8-90C8-7E21C4895877-70070-0000998B7B95BBBF/Screenshot 2020-08-14 at 1.52.48 AM.png]



[image:816A457E-FFE9-42E7-A664-E9826F3D2D57-70070-000099A23C4ECC10/Screenshot 2020-08-14 at 1.54.28 AM.png]


If we project hyperplane which is same as running a linear model in 3 D. We use a RBF vector to create a 3 D plot of data and run Hyplerplane to get minimum error

## Naives Bayes:

[image:C127D25E-1731-4ACC-86B7-E28A74646707-70070-0000BE79B3E38794/Screenshot 2020-08-14 at 10.28.04 PM.png]

How can the above be applied to train a model.

[image:9D752464-EBAF-48E5-8B55-7AE7ABDBC0DB-70070-0000BEAC039EC1DD/Screenshot 2020-08-14 at 10.31.40 PM.png]


## Decision Tree:

[image:7B47BC80-D70F-4BE6-A94F-D7695447961B-70070-00011342A7BB91C7/Screenshot 2020-08-17 at 4.16.31 PM.png]

Splits in such a way to maximise a category in each split.  It is very similar to Regression part of this, difference being in the algorithm to classify or regress.

## Random Forest Classifier:
Ensemble learning is when you take a lot of models and train them together and take their average.
For eg We can have multiple points on the basis of which we can device Decision tress who’s combined average gives us a comparatively good result.

 
## CAP Analysis:
[image:58A9666E-2BC8-4EF2-BCFC-83EA10903D35-40253-0002704245222929/Screenshot 2020-08-24 at 4.32.07 PM.png]



## K Means Clustering
[image:117D6371-123F-40DD-9468-5DC840A1EB40-40253-0002713ED21FB0EC/Screenshot 2020-08-24 at 4.50.17 PM.png]


[image:52DD504D-726F-45F5-B601-10D3F2A28435-40253-000271CB5E090B64/Screenshot 2020-08-24 at 5.00.21 PM.png]

 Move the centroids

[image:FB27CDD2-373F-4853-A06F-E53E8F4F72E7-40253-0002720B38F2AA67/Screenshot 2020-08-24 at 5.04.55 PM.png]

When no new re assignments take place we can assume the algorithm
Has converged.


[image:E36BD6C7-CABC-4900-92A9-A752A778FD15-40253-0002722485B5DC7C/Screenshot 2020-08-24 at 5.06.44 PM.png]


The selection of centroid can hinder the selection of clusters. There is modification in K Means clustering algorithm which is  k means ++ algorithm.

[image:F03D7199-B6BA-4C66-A0FE-82212FFA5C6A-40253-000272F389B05343/Screenshot 2020-08-24 at 5.21.33 PM.png]


When there is only 1 cluster. The value of WCSS will be very large.

[image:74A784FB-C4EA-4D9B-BFDB-2683BA727A89-40253-00027314AE57B26D/Screenshot 2020-08-24 at 5.23.55 PM.png]

When clusters are increased to 2. The WCSS decreases.

[image:CB5F0114-19E8-4C62-A941-A602F203A138-40253-0002731CD50DA3D7/Screenshot 2020-08-24 at 5.24.30 PM.png]


When the number clusters is 3.

[image:2CF7BABC-17AC-4323-B91C-F5234A80F861-40253-00027307985689A6/Screenshot 2020-08-24 at 5.22.59 PM.png]

We can have as many clusters, but how to find optimal fit. 
[image:D3A569AF-4839-4E11-B76D-F8D28A685DBB-40253-0002734CEFEE1F12/Screenshot 2020-08-24 at 5.27.57 PM.png]

When the drop in WCSS becomes less, We see an elbow point which is the number of clusters required for data modelling

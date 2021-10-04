# Machine-Learning-101
# Pre Processing

*Data Pre Processing Notes:*
1. Pandas library is super important for data pre processing.
2. pd.read_csv(‘path/to/dataset’) creates a dataframe .
3. In any dataset with which you train a model
	1. *Features/Independant Variable*: columns with which we predict dependant variable. Mostly present in first columns.
	2. *Labels/Dependant variable factor*: This is usually in last column of dataset and is something which has to be predicted.
4. Missing data can be harmful for training the data, so in turn sometimes we end up filling it using various tools. (Like missing salary can be replaced by average salaries)
5. Encoding categorical data:
	1. Sometimes there are some string categories of data which is difficult for the machine to understand so we encode this data.One simple way is to give numbers to each data. For eg Spain :1, Germany: 2 etc.
6. Better way is one hot encoding is turning the country column to three dfifferent something like.
[1,0,0]: Spain
[0,1,0]: Germany
This will be easier for machine to interpret
7. Label encoding: dependant variable we use Label Encoder.
8. When should be feature scaling done?
	1. It should be done after splitting the dataset into train and test.
	2. Feature scaling is taking the variables in same scale.
	3. Test set is supposed to be brand new, so should not be worked before training.
	4. Feature scaling works on the data before training

9. Feature Scaling:
	1. Allows us to put all the features to same scale
a. Standardisation (- 3 to +3): it works well all the time
b. Normalisation (0 to 1):  recommended when distribution is normal

Go with Standardisation
10. Only apply transform on test dataset



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

<img width="690" alt="Screenshot 2020-08-12 at 5 56 23 PM" src="https://user-images.githubusercontent.com/50817904/135798890-4f4a83be-73f7-4df1-b0da-2399dbac1605.png">

Applying Linear regression to classification problem.


We stage a likelihood of the person taking an offer. 

Linear Regression at least gives us the range  of people taking the offer. The line in between 1 and 0 makes sense but not the line above or below it.

<img width="674" alt="Screenshot 2020-08-12 at 5 59 41 PM" src="https://user-images.githubusercontent.com/50817904/135798938-db459743-2119-4599-b310-e39155e1644e.png">



We cut of the lines from top and below.
Apply a sigmoid function to /y = mx +c/


### Logistic Regression
### log(p/p-1) = b0 + b1x

<img width="698" alt="Screenshot 2020-08-12 at 6 02 06 PM" src="https://user-images.githubusercontent.com/50817904/135798925-2179b1ca-1cd2-4657-809c-2849e648e983.png">


An example with various age groups›



<img width="616" alt="Screenshot 2020-08-12 at 6 11 38 PM" src="https://user-images.githubusercontent.com/50817904/135798957-48689245-2a12-4d10-85de-84bf18bbea91.png">



We use the probability to have a score. But what if we don’t want probability and ask for prediction.
<img width="625" alt="Screenshot 2020-08-12 at 6 14 11 PM" src="https://user-images.githubusercontent.com/50817904/135798971-e05e6e30-a93f-42ab-8bf8-ced228da5d82.png">

Anything having probability of less than 0.5 its projected to 0 and anything above is projected upward.

## K Nearest Neighbor:
 

<img width="698" alt="Screenshot 2020-08-13 at 6 32 17 PM" src="https://user-images.githubusercontent.com/50817904/135798986-02183b7c-e8a5-42df-bea7-8ee3bde89369.png">



Rule Guide:
1. Choose the number K of neighbours, default values is 5.
2. Take Manhattan distance or euclidian distance.
3. Count the data points in different category.
4. Assign new data point to the category where you counted most neighbours.



<img width="692" alt="Screenshot 2020-08-13 at 6 35 18 PM" src="https://user-images.githubusercontent.com/50817904/135799001-badf9a0c-f49f-4140-9dca-20c44fe5d0a8.png">

Euclidean distance

<img width="686" alt="Screenshot 2020-08-13 at 6 35 49 PM" src="https://user-images.githubusercontent.com/50817904/135799007-205a17be-4626-4def-9aa1-0086ea01faf9.png">

Take Euclidean distance from 5 points and assign the category.


## Support Vector Machine
<img width="690" alt="Screenshot 2020-08-13 at 6 54 34 PM" src="https://user-images.githubusercontent.com/50817904/135799017-42206d97-2cb9-487e-a559-ff178a369c9d.png">
<img width="683" alt="Screenshot 2020-08-13 at 6 56 05 PM" src="https://user-images.githubusercontent.com/50817904/135799026-6fc2e73c-eee1-455f-aa25-5278b87dce58.png">


SVM tries to pick extreme cases of categories which is risky. If we are differentiating between apples and oranges most of algorithms will look at only most common features SVM looks at the boundary conditions and tries to create a Line to separate them.


<img width="688" alt="Screenshot 2020-08-13 at 7 01 00 PM" src="https://user-images.githubusercontent.com/50817904/135799034-1f9a4e22-ce07-42ff-9cad-c8770ef21e54.png">


## Kernel SVM


What if the Data cannot be separated Linearly. In that case we use Kernel SVM

<img width="650" alt="Screenshot 2020-08-13 at 7 35 59 PM" src="https://user-images.githubusercontent.com/50817904/135799043-f40afba4-473a-4efd-8084-dd375af262e5.png">


Map data into Linearly separable dataset using Higher Dimension.

<img width="655" alt="Screenshot 2020-08-13 at 7 40 23 PM" src="https://user-images.githubusercontent.com/50817904/135799055-e9b6b7a8-fa25-4307-b2ae-64d60552a422.png">

Post applying some function, Hyperplane separates the data.

<img width="689" alt="Screenshot 2020-08-13 at 7 41 56 PM" src="https://user-images.githubusercontent.com/50817904/135799061-f35c765f-6606-4a4f-aeb2-9214589df121.png">

### Mapping to higher dimension it can lead to more compute power being required

## The Kernel Trick

<img width="683" alt="Screenshot 2020-08-13 at 7 45 56 PM" src="https://user-images.githubusercontent.com/50817904/135799070-5dd0542a-7f59-4e1e-9ed6-939143d83b16.png">


If landmark is large then we get a value very close to zero. If the landmark is closer to reference point it be smaller and e^0 is 1.


<img width="663" alt="Screenshot 2020-08-13 at 7 51 09 PM" src="https://user-images.githubusercontent.com/50817904/135799079-17b7bdcc-e51c-4dc1-9f79-fc9db89584b0.png">

We use the kernel to separate our data. 
<img width="682" alt="Screenshot 2020-08-13 at 7 52 26 PM" src="https://user-images.githubusercontent.com/50817904/135799083-ad257cd1-62b0-4940-8555-f69b139527af.png">

Anything out side circle will assigned 0, anything outside will be 1.
Sigma defines how wide can the circumference of cycle can be. By finding the right sigma we find the distinction.

*Types of kernel function:*

<img width="692" alt="Screenshot 2020-08-13 at 8 06 18 PM" src="https://user-images.githubusercontent.com/50817904/135799090-05cf9bf7-eed9-489a-a40d-c8600045eb42.png">

 
## Non Linear SVR

<img width="736" alt="Screenshot 2020-08-14 at 1 52 48 AM" src="https://user-images.githubusercontent.com/50817904/135799102-28c80a86-52c5-4b18-b117-15324d358dd9.png">



<img width="745" alt="Screenshot 2020-08-14 at 1 54 28 AM" src="https://user-images.githubusercontent.com/50817904/135799115-0194c0b2-bc16-4f82-9557-eced5a71a53b.png">


If we project hyperplane which is same as running a linear model in 3 D. We use a RBF vector to create a 3 D plot of data and run Hyplerplane to get minimum error

## Naives Bayes:

<img width="622" alt="Screenshot 2020-08-14 at 10 28 04 PM" src="https://user-images.githubusercontent.com/50817904/135799125-7e330b73-cec6-4ded-9b68-337ad3d13e86.png">

How can the above be applied to train a model.

<img width="700" alt="Screenshot 2020-08-14 at 10 31 40 PM" src="https://user-images.githubusercontent.com/50817904/135799131-c7b325d6-043a-434e-89ad-ec57a75cf777.png">


## Decision Tree:

<img width="676" alt="Screenshot 2020-08-17 at 4 16 31 PM" src="https://user-images.githubusercontent.com/50817904/135799139-7a636751-edab-4ab2-8cec-dfacd17cd310.png">

Splits in such a way to maximise a category in each split.  It is very similar to Regression part of this, difference being in the algorithm to classify or regress.

## Random Forest Classifier:
Ensemble learning is when you take a lot of models and train them together and take their average.
For eg We can have multiple points on the basis of which we can device Decision tress who’s combined average gives us a comparatively good result.

 
## CAP Analysis:
![Screenshot 2020-08-24 at 4 32 07 PM](https://user-images.githubusercontent.com/50817904/135799147-879a32c3-3c89-4315-9385-8f402a627f7f.png)



## K Means Clustering
![Screenshot 2020-08-24 at 4 50 17 PM](https://user-images.githubusercontent.com/50817904/135799153-ffd114f7-9748-4ec7-8c9d-8a9f935068a3.png)


![Screenshot 2020-08-24 at 5 00 21 PM](https://user-images.githubusercontent.com/50817904/135799157-489fc766-8a4b-44cc-9c17-9b25968857f9.png)

 Move the centroids

![Screenshot 2020-08-24 at 5 04 55 PM](https://user-images.githubusercontent.com/50817904/135799162-e21b05dd-94db-496b-a9ad-e88de96b465b.png)

When no new re assignments take place we can assume the algorithm
Has converged.


![Screenshot 2020-08-24 at 5 06 44 PM](https://user-images.githubusercontent.com/50817904/135799172-3b8bbaa3-4aa9-4acc-88fe-9879da1ab236.png)


The selection of centroid can hinder the selection of clusters. There is modification in K Means clustering algorithm which is  k means ++ algorithm.

![Screenshot 2020-08-24 at 5 21 33 PM](https://user-images.githubusercontent.com/50817904/135799175-28302555-5663-49b3-85b5-4067b544beb4.png)


When there is only 1 cluster. The value of WCSS will be very large.

![Screenshot 2020-08-24 at 5 23 55 PM](https://user-images.githubusercontent.com/50817904/135799183-96c14453-ed50-4910-8476-ffd854ed7ae5.png)

When clusters are increased to 2. The WCSS decreases.

![Screenshot 2020-08-24 at 5 24 30 PM](https://user-images.githubusercontent.com/50817904/135799193-93634bbf-b689-4775-92ed-e203b8a8ad09.png)


When the number clusters is 3.

![Screenshot 2020-08-24 at 5 22 59 PM](https://user-images.githubusercontent.com/50817904/135799201-b7698560-ab24-4551-b0c0-e901619bb8bb.png)

We can have as many clusters, but how to find optimal fit. 
![Screenshot 2020-08-24 at 5 27 57 PM](https://user-images.githubusercontent.com/50817904/135799205-9102f3b8-a185-4f22-ae45-e302b9650aec.png)

When the drop in WCSS becomes less, We see an elbow point which is the number of clusters required for data modelling



# Regression 
Regression model are used for predicting future values of a particular nature.
## Simple Linear Regression:
Y = b0 + b1*X1

Y: Dependant Variable(Something you try to understand how is it dependant on something)
X: Independent Variable which might and might not affect the dependant variable
b1: Coefficient (connector between y and x)
b0: Constant term.

If we are trying to figure out salary for x work ex:
*Salary = b0 + b1 * Experience*

Constant : Intersection on Y axis (Starting salary)
B1: Slope of the line.
<img width="937" alt="Screenshot 2020-08-09 at 7 42 56 PM" src="https://user-images.githubusercontent.com/50817904/135799340-9b143e87-023c-4b11-9643-d3808d736713.png">


 
*Simple Linear Regression:*


<img width="945" alt="Screenshot 2020-08-09 at 7 43 05 PM" src="https://user-images.githubusercontent.com/50817904/135799349-2c3b00f0-59f2-4ef7-b253-533df4aff0a5.png">


Simple Linear Regression will make many such lines from actual to assumed and calculate 

SUM(y - y`)^2 -> min
 

The Line represents the predicted line trying to fit through test dataset.



<img width="640" alt="Screenshot 2020-08-09 at 9 05 56 PM" src="https://user-images.githubusercontent.com/50817904/135799363-fa9fa21c-0067-4e82-8893-45bd6ef182d2.png">





<img width="633" alt="Screenshot 2020-08-09 at 8 58 26 PM" src="https://user-images.githubusercontent.com/50817904/135799377-8972ac1b-1d9e-4e8c-9f3f-d0d812bbfc0b.png">



## Multiple Regression
Much more coefficients as compared to Linear regression
Y = b0 + b1*x1 + b2*x2 + b3*x3…

There are some Independent variables or features which we have to throw out to increase the accuracy of the model. Only important features should be selected.

## P- Value: 
The value at which we discard a feature and assume its not adding any value to model.

Methods to select the correct features:
1. All in
2. Backward Elimination
3. Forward Selection
4. Bidirectional Elimination
5. Score Comparison

Stepwise regression is Backward, Forward and Bidirectional.

## All in:
Throw all the variables to build the model.
It is used for prepping for Backward Elimination

## Backward Elimination:
1. Select significance level to stay in the model. 5%
2. Fit the full model with all possible predictors
3. Consider the predictor with the highest P value
	1. If p > SL go to step 4 else fin.
4. Remove predictor
5. Fit the model without this variable.
6. Go to step 3

## Forward Selection
1. Select significance level to 5 %
2. Fit all the simple regression models. Select 1 with lowest p values
3. Keep the variable and fit all possible models with one extra predictor added to one you already have.
4. Consider the predictor with the lowest P-value.if P value < SL go to step 3 else FIN.
5. Go to step 3.  Keep the previous model

## Bidirectional Elimination
1. STAY 5% and ENTER 5%
2. Perform next step of Forward selection p<ENTER
3. Perform next step of Backward selection p<STAY.go to step 2
4. Until no new variable.

## All Models:
1. Select a criterion of goodness of fit.
2. Construct all possible Regression Models. 2^n-1
3. Select with nest criterion

Backward Elimination os the fastest.

## Polynomial Linear Regression
Y = b0 + b1x1 + b2x2^2 + …
If the data distribution is non linear we need a non linear curve to match the data better. We add vector for powers of the feature

<img width="700" alt="Screenshot 2020-08-10 at 2 23 07 PM" src="https://user-images.githubusercontent.com/50817904/135799402-5d1f14bd-ae9e-4597-a6b1-5459477bb8ce.png">


They are used to describe diseases might spread etc. 
Why we didn’t split the data to training and test set?
We have very few number of observations. So we take all the rows. 

Comparing Linear and Polynomial Models using marplot



<img width="537" alt="Screenshot 2020-08-10 at 3 02 09 PM" src="https://user-images.githubusercontent.com/50817904/135799412-6b71f567-79ba-4ba6-98a6-594e7ad40a8e.png">





<img width="487" alt="Screenshot 2020-08-10 at 3 03 40 PM" src="https://user-images.githubusercontent.com/50817904/135799420-275dfe3e-d69c-4a57-a097-2055559b64f1.png">




## Support Vector Regression
<img width="734" alt="Screenshot 2020-08-10 at 3 48 57 PM" src="https://user-images.githubusercontent.com/50817904/135799430-b5a027c9-9a9c-4303-b5bc-dc49363e3e99.png">



Adds a layer of buffer to Linear regression Line. There are some points which lie out side Epsilon tube, these are slack variables. The min distance from slack variable defines the Linear Buffer line passing through the data.


The ones which have implicit relationship we have to apply feature scaling.

We also apply the feature scaling to Label/ Dependant variable.



## Decision Trees
Classification Tress 
Regression Trees


<img width="773" alt="Screenshot 2020-08-10 at 9 38 41 PM" src="https://user-images.githubusercontent.com/50817904/135799444-a4a13861-665e-4c14-a090-488d8186f62e.png">


Each partition is called a leaf. Algorithm finds splits and final leaves are called terminal leaves.


<img width="703" alt="Screenshot 2020-08-10 at 9 41 51 PM" src="https://user-images.githubusercontent.com/50817904/135799452-16656eb5-78a0-462d-9b00-4f0a4163d04f.png">


The above figure shows how the decision tree is constructed using the splits.

You take average of the points within the terminal leaves which will be assigned to prediction.

<img width="767" alt="Screenshot 2020-08-10 at 9 43 48 PM" src="https://user-images.githubusercontent.com/50817904/135799456-cfe0c10d-094d-4815-a347-4a2a4f3a7c64.png">

Add the above average values to Decision tree and the data coming in will use this decision tree to make predictions.


<img width="766" alt="Screenshot 2020-08-10 at 9 45 21 PM" src="https://user-images.githubusercontent.com/50817904/135799465-b7f2455e-d5f2-4b00-960d-3f935ffd8b60.png">


We don’t have to apply feature scaling for decision tree.
They work with highly complex datasets


## Random Forest Regression
This is version of ensemble learning, which when you take same algorithm multiple times to make it better.

1. Pick random data points from training set.
2. Build a decision tree based on data points selected above
3. Keep building regression trees.
4. Use all of them to predict.


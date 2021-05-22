# [Santander-Customer-Prediction](https://github.com/Deepakit/Santander-Customer-Prediction/blob/main/Santander_prediction.ipynb)

1) In this challenge, we tried to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. In Santander    Customer Transaction Prediction ,we have a binary classification task.

2) Predicted customer transaction using prehistoric data. Used Machine learning Models like Logistic Regression,Random Forest, Naïve Bayes to obtain the outcome.
   Data can be downloaded from [Link](https://www.kaggle.com/c/santander-customer-transaction-prediction)

## Data
We are provided with Train and test data which have 200k samples each and we have 200 anonimyzed numerical columns. 

* Training Data:

![](/Images/Training_data.PNG)

* Testing Data:

![](https://github.com/Deepakit/Santander-Customer-Prediction/blob/main/Images/Testing_data.PNG)

We have 200 numerical variables in both sets , named from var_0 to var_199 and one target, ID column.

## EDA

We take a look at the data provided to us by organizing ,plotting and summarizing the data.
By doing same, we can get an idea how the data is distributed and is there any pattern observed in data.

### Statistics of the data
We use .describe() in python to get a look at the numerical data composition.

![](https://github.com/Deepakit/Santander-Customer-Prediction/blob/main/Images/Describe%20data.PNG)

* Standard Deviation in both train and test is quite significant.
* Mean and other measures are close.

### Target Distribution
![](https://github.com/Deepakit/Santander-Customer-Prediction/blob/main/Images/Target_dist.PNG)

We noted few pointers:
* We are having a unbalanced data, where 90% of the data is no. of customers who will not make a transaction & 10 % of  the data are those who will make a transaction.
* From the violin plots, it seems that there is no relationship between the target and index of the data frame, it is more dominated by zero compare to one's.
* From the jitter plots with violin plots, we can observe that target looks uniform ly distributed over the indexes of the data frame.

### Distribution
Get an idea of this data distribution, we review in the training dataset that we will work with, we review the histogram of the mean values of each record based on the binary target variable.

![](https://github.com/Deepakit/Santander-Customer-Prediction/blob/main/Images/dist_of_mean_over_data.PNG)

As we can see that there is a small variation in the mean of all feature that could explain the target variable

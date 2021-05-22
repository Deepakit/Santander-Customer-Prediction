# [Santander-Customer-Prediction](https://github.com/Deepakit/Santander-Customer-Prediction/blob/main/Santander_prediction.ipynb)

1) In this challenge, we tried to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. In Santander    Customer Transaction Prediction ,we have a binary classification task.

2) Predicted customer transaction using prehistoric data. Used Machine learning Models like Logistic Regression,Random Forest, Naïve Bayes to obtain the outcome.
   Data can be downloaded from [Link](https://www.kaggle.com/c/santander-customer-transaction-prediction)

## Data
We are provided with Train and test data which have 200k samples each and we have 200 anonimyzed numerical columns. 

..* Training Data

![](/Images/Training_data.png)

..* Testing Data

![](/Images/Testing_data.png)

We have 200 numerical variables in both sets , named from var_0 to var_199 and one target, ID column.

## EDA

We take a look at the data provided to us by organizing ,plotting and summarizing the data.
By doing same, we can get an idea how the data is distributed and is there any pattern observed in data.

### Statistics of the data
We use .describe() in python to get a look at the numerical data composition.
![](/Images/Describe data.png)

•	Standard Deviation in both train and test is quite significant.
•	Mean and other measures are close.


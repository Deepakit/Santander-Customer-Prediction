# [Santander-Customer-Prediction](https://github.com/Deepakit/Santander-Customer-Prediction/blob/main/Santander_prediction.ipynb)

1) In this challenge, we tried to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. In Santander    Customer Transaction Prediction ,we have a binary classification task.

2) Predicted customer transaction using prehistoric data. Used Machine learning Models like Logistic Regression,Random Forest, Naïve Bayes to obtain the outcome.
   Data can be downloaded from [Link](https://www.kaggle.com/c/santander-customer-transaction-prediction)

## Data
We are provided with Train and test data which have 200k samples each and we have 200 anonimyzed numerical columns. 

* Training Data:

![](/Images/Training_data.PNG)

* Testing Data:

![](/Images/Testing_data.PNG)

We have 200 numerical variables in both sets , named from var_0 to var_199 and one target, ID column.

## EDA

We take a look at the data provided to us by organizing ,plotting and summarizing the data.
By doing same, we can get an idea how the data is distributed and is there any pattern observed in data.

### Statistics of the data
We use .describe() in python to get a look at the numerical data composition.

![](/Images/Describe%20data.PNG)

* Standard Deviation in both train and test is quite significant.
* Mean and other measures are close.

### Target Distribution
![](/Images/Target_dist.PNG)

We noted few pointers:
* We are having a unbalanced data, where 90% of the data is no. of customers who will not make a transaction & 10 % of  the data are those who will make a transaction.
* From the violin plots, it seems that there is no relationship between the target and index of the data frame, it is more dominated by zero compare to one's.
* From the jitter plots with violin plots, we can observe that target looks uniform ly distributed over the indexes of the data frame.

### Distribution
Get an idea of this data distribution, we review in the training dataset that we will work with, we review the histogram of the mean values of each record based on the binary target variable.

![](/Images/dist_of_mean_over_data.PNG)

As we can see that there is a small variation in the mean of all feature that could explain the target variable

### Missing Value Analysis

We look for possible null values in the dataframe and if found any we will be filling them if number is significant.
```python
print(customer_data.isnull().sum().any())
print(test_data.isnull().sum().any())
```
As we can see there is no missing values in both train as well as in test data.

### Outlier
Outliers in the data may occur due to poor measurement quality or some external reasons. As they may effect in our prediction modelling we have to deal with it. In a simple way we can detect outliers by plotting box plots of the different variables in the data set. We used boxplot method to identify the outliers. It helps by defining the upperlimit and lower limit beyond which any data lying is considered to be an outlier. 
![](/Images/outlier.PNG)
Although, I have only posted here till var_39 but outliers are present in all columns. This can affect the model so we will be removing them.
We calculate the 25 and 75 percentile , and found min and max , and remove all the points less than min and greater than max.

### Feature Selection
Feature selection is very important for modelling . Every dataset have unwanted and good features and both features affect the performance of model. In the classification modelling feature selection is about selecting the independent variables which will be helpful in predicting the target variable. It is also know as Dimensionality Reduction. For numerical data we can use correlation plot.
![](/Images/heatmap_1.PNG)
From the colour of the graph we can see that there isn't much correlation between the variables. So, we have to keep all the 200 columns.

To re-confirm the same I also used PCA on dummy training dataset and plot variance ratio.
![](/Images/plot-variance.PNG)

## Data Preparation
Note we are dealing with a data set very unbalanced, where there is only *10%* of records categorized with __target 1__, so those customers who have made a financial transaction.
To develop a binary classification model we need to have more balanced data since most machine learning algorithms work best when the number of samples in each class is almost the same. This is because most algorithms are designed to maximize accuracy and reduce error, so we'll try to do this in this section before to predict models fit better.
How we have a large dataset with 200,000 records we could undersampling in the data with the balanced target variable. Initially we will test a resampling in a 1:1 ratio but depending on the results we can use other proportions. Keep in mind that with undersampling we might be removing information that may be valuable. This could lead to a lack of fit and poor generalization of the test set.

![](/main/Images/Sampling.PNG)

### Under-Sampling
```python
class_0,class_1 = customer_data.target.value_counts()

df_class_0 = customer_data[customer_data['target']==0]
df_class_1 = customer_data[customer_data['target']==1]

under_df_0 = df_class_0.sample(class_1)
df_train_under = pd.concat([under_df_0,df_class_1],axis=0)

print(df_train_under.target.value_counts())
df_train_under.describe()
```

### Over-Sampling
```python
over_df = resample(df_class_1, replace=True, n_samples=179813,random_state=123)

df_train_over = pd.concat([over_df,df_class_0],axis=0)

len(df_train_over)
print(df_train_over.target.value_counts())
df_train_over.describe()
```

## Modelling
After all early stages of preprocessing, then model the data. So, we have to select best model for this project with the help of some metrics.

In order to automate the performance measures of the different models, we will factor a function to measure the metrics and be able to make comparisons between the different algorithms applied.
```python
def metrics(y_true,y_pred):
    print("Confusion Matrix")
    print(confusion_matrix(y_true,y_pred))
    
    print("Accuracy:", accuracy_score(y_true,y_pred))
    print("Precision:", precision_score(y_true,y_pred))
    print("F1 Score:", f1_score(y_true,y_pred))
    print("Recall:", recall_score(y_true,y_pred))
    
    false_positive_rate,recall,thresholds = roc_curve(y_true,y_pred)
    roc_auc = auc(false_positive_rate,recall)
    
    print("ROC:",roc_auc)
    
    plt.plot(false_positive_rate,recall,'b')
    plt.plot([0,1],[0,1],'r--')
    plt.title("AUC=%0.2f"%roc_auc)
    plt.show()
```
The models we used are:
1) Logistic Regression
   As data provided is linearly separable it provides us with a good opportunity to check our data with the simplest model so to set a benchmark for other models to beat.
2) Random Forest
   This constructor requires more parameters than the decision tree because it is to be told the number of tree models to use, for which the parameter can be used                n_estimators. On the other hand, as selecting the data to be used for each submodel it is a good idea to fix the seed to ensure that the results are repeatable
3) Naive Bayes
   Naive Bayes classifiers are a family of simple "probabilistic classifier" based on applying bayes theorm with strong independence assumptions between the features.We did      not found any dependency between data and dataset itself is too large.

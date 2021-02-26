
#setting working directory

setwd("C:/Programs/Santander")
getwd()

#Loading the libraries
L = c('tidyr','ggplot2','corrgram','usdm','gridExtra','mlbench','caret','lattice','DMwR','rpart','randomForest')

#loading packages
lapply(L, require, character.only = TRUE)
rm(L)

#Loading Datasets
train_df=read.csv('train.csv',header = TRUE)
test_df=read.csv('test.csv',header = TRUE)

#checking dimensions
dim(train_df)
dim(test_df)

#####EXPLORING_DATA#####

#Structure of data
str(train_df)
str(test_df)

#Summary of Data
summary(train_df)
summary(test_df)

head(train_df)


######MISSING_VALUES######

#checking for missing values

sum(is.na(train_df))
sum(is.na(test_df))
#storing numeric data without ID_code 

train_data=train_df[,2:202]
View(train_data)
test_data=test_df[,2:201]
View(test_data)

#factorize the target variable
train_data$target = factor(train_data$target, levels = c(0, 1))

#column names
cnames=colnames(train_df[,3:202])
cnames

######EXPLORATORY_DATA_ANALYSIS######

#Plot graph for o & 1s in target variable
require(gridExtra)
target_count <-table(train_df$target)
target_count/length(train_df$target)*100

##Plotting target variable
plot1<-ggplot(train_df,aes(target))+theme_bw()+geom_bar(stat='count',fill='lightgreen')

#Violin Plot with Jitters
plot2<-ggplot(train_df,aes(x=target,y=1:nrow(train_df)))+theme_bw()+geom_violin(fill='lightblue')+
facet_grid(train_df$target)+geom_jitter(width=0.02)+labs(y='Index')
grid.arrange(plot1,plot2,ncol=2)


#####OUTLIER_ANALYSIS######

#Removing outliers from the train set
for (i in cnames) {
  val = train_data[,i][train_data[,i]%in% boxplot.stats(train_data[,i])$out]
  print(length(val))
  train_data[,i][train_data[,i] %in% val] = NA
  
}
#Again checking for missing data after outliers
apply(train_data, 2,function(x) {sum(is.na(x))})

train_data= drop_na(train_data)

#copying data
data_no_outliers =train_data
dim(train_data)

#Removing outliers from the test set
for (i in cnames) {
  val = test_data[,i][test_data[,i]%in% boxplot.stats(test_data[,i])$out]
  print(length(val))
  test_data[,i][test_data[,i] %in% val] = NA
  
}
#Again checking for missing data after outliers
apply(test_data, 2,function(x) {sum(is.na(x))})

test_data= drop_na(test_data)

#copying data
Test_no_outliers =test_data
dim(test_data)

######FEATURE SELECTION #########

#Correlation

correlationMatrix <-cor(train_data[,1:201])
print(correlationMatrix)
highlyCorrelated <-findCorrelation(correlationMatrix,cutoff = 0.75)
print(highlyCorrelated)

###So we don't have any correlation in the variables so we have to keep all the 200 variables


#######FEATURE_SCALING#######


#normalization on train_data

for(i in cnames){
  print(i)
  train_data[,i] = (train_data[,i] - min(train_data[,i]))/
    (max(train_data[,i] - min(train_data[,i])))
}
View(train_data)


#normalization on test_data

for(i in cnames){
  print(i)
  test_data[,i] = (test_data[,i] - min(test_data[,i]))/
    (max(test_data[,i] - min(test_data[,i])))
}
View(test_data)


#####MODELLING######

library(caret)
#Dividing data into to train and test
set.seed(272)
train_index = createDataPartition(train_data$target,p=0.7,list = FALSE)
train = train_data[train_index,]
test = train_data[-train_index,]



#Fitting Logistic Regression to the Training set

logit_model = glm(formula = target ~ .,family = binomial,data = train)
summary(logit_model)
# Predicting the Test set results
log_pred = predict(logit_model, type = 'response', newdata = test)
y_pred = ifelse(log_pred > 0.5, 1, 0)
y_pred
ConfMatrix_RF = table(test$target, y_pred)

confusionMatrix(ConfMatrix_RF)


#NaiveBayes MOdel

library(caTools)
library(e1071)

#Develop model
NB_model = naiveBayes(target ~ ., data =train)
summary(NB_model)
#predict on test cases 
NB_Pred = predict(NB_model, test[,2:201], type = "class")
NB_Pred
#Look at confusion matrix
Conf_matrix = table(test[,1],NB_Pred)
confusionMatrix(Conf_matrix)

#RANDOM Forest
install.packages("randomForest")
library(randomForest)
library(ggplot2)
library(inTrees)
RF_model = randomForest(target ~ ., train, importance = TRUE, ntree = 100)
treeList = RF2List(RF_model)  
exec = extractRules(treeList, train[,-1])
exec[1:2,]
readableRules = presentRules(exec, colnames(train))
readableRules[1:2,]   
ruleMetric = getRuleMetric(exec, train[,-1], train$target)
ruleMetric[1:2,]
RF_Predictions = predict(RF_model, test[,-1])
ConfMatrix_RF = table(test$target, RF_Predictions)
confusionMatrix(ConfMatrix_RF)



#Predict the test_dataset outcome
Test_Pred = predict(NB_model, test_data[,1:200], type = 'class')
Test_Pred
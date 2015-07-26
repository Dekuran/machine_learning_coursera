## machine learning project using wearable technology exercise data from the Human Activity Recognition project to train a model to detect correct or incorrect exercise techniques
#start by loading useful packages
library(ggplot2)
library(caret)
library(curl)
install.packages("doParallel")
library(doParallel)
registerDoParallel(cores=2)
install.packages("gbm")
library(gbm)
install.packages("randomForest")
library(randomForest)
## let's get the training data now
trainurl<-'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
download.file(trainurl,destfile="train.csv", method="curl")
train<-read.csv("train.csv" )
summary(train)
#removing the rownumbers, timestamps, window, and usernames as variables
exercise<-subset(train,select=-c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))
# removing mostly NA data
few_na<-apply(!is.na(exercise),2,sum)>19621
exercise<-exercise[,few_na]
#splitting the training data into training and testing data to allow some pre-testing before using the real test set
inTrain<-createDataPartition(y=exercise$classe,p=0.7,list=F)
training<-exercise[inTrain,]
testing<-exercise[-inTrain,]
inTrainsmall<-createDataPartition(y=exercise$classe,p=0.3,list=F)
trainingsmall<-exercise[inTrainsmall,]
## apply random forest method
modFit<-train(classe ~. , 
              data= training, 
              method="rf" ,
              trControl=trainControl(method="cv",number=5),
              prox=TRUE,
              allowParallel=TRUE)
print(modFit)
print(modFit$finalModel)
pred<-predict(modFit$finalModel, newdata=testing)
testing<-complete.cases(testing)
testing$predRight<- pred == testing$classe
table(pred,testing$classe)
pred
testing$classe

## now let's get the real test data
trainurl<-'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
download.file(trainurl,destfile="test.csv", method="curl")
test<-read.csv("test.csv" )
summary(test)
#removing the rownumbers, timestamps, window, and usernames as variables
testing2<-subset(test,select=-c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))
# removing mostly NA data
few_na<-apply(!is.na(exercise),2,sum)>19621
exercise<-exercise[,few_na]

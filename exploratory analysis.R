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
dtraining<-exercise
nzv<-nearZeroVar(dtraining, saveMetrics=TRUE)
nzv<-subset(nzv, nzv==TRUE)     # a subset of 60 columns of the original data
dtraining<-subset(dtraining, select = !names(dtraining) %in% rownames(nzv))
dNA<-function(vector){ return(sum(is.na(vector))) }
NAnum<-sapply(dtraining, dNA)
noNA<-names(NAnum[NAnum / nrow(dtraining) == 0])
dtraining<-subset(dtraining, select = names(dtraining) %in% noNA)


few_na<-apply(!is.na(exercise),2,sum)>19621
exercise<-exercise[,few_na]
#splitting the training data into training and testing data to allow some pre-testing before using the real test set
preProVals <-preProcess(dtraining[,-c(1,55)], method = c("center", "scale"))
preProVals$user_name <- dtraining$user_name
preProVals$classe <- dtraining$classe
toTrain <- createDataPartition(preProVals$classe, p=0.75, list=F) #p = default value.check it!
training<-dtraining[toTrain,]
testing<-dtraining[-toTrain]

inTrain<-createDataPartition(y=dtraining$classe,p=0.7,list=F)
training<-dtraining[inTrain,]
testing<-dtraining[-inTrain,]
inTrainsmall<-createDataPartition(y=exercise$classe,p=0.1,list=F)
trainingsmall<-exercise[inTrainsmall,]
testingbig<-exercise[-inTrainsmall,]

summary(training)
qplot(data=training, accel_dumbbell_z, accel_forearm_z, colour=classe)
## apply random forest method
modFit<-train(classe ~. , data= trainingsmall, method="rf" , trControl=trainControl(method="cv",number=5),prox=TRUE, allowParallel=TRUE)
modFit2<-train(classe ~. , data= training , method="knn", trControl=trainControl(method="cv",number=5),  allowParallel=TRUE)
## so slow! try using knn maybe? adaboost?

print(modFit)
print(modFit$finalModel)
pred<-predict(modFit$finalModel, newdata=testingbig)
testing<-complete.cases(testingbig)
testing$predRight<- pred == testing$classe
table(pred,testing$classe)
pred
testing$classe

## now let's get the real test data
testurl<-'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
download.file(trainurl,destfile="test.csv", method="curl")
test<-read.csv("test.csv" )
summary(test)
#removing the rownumbers, timestamps, window, and usernames as variables
testing2<-subset(test,select=-c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))
# removing mostly NA data
few_na<-apply(!is.na(testing2),2,sum)>19621
testing2<-testing2[,few_na]

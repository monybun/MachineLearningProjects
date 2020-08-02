# MachineLearningProjects
install.packages("kernlab")
install.packages("class")
install.packages("caret")
install.packages("shiny")

#Use spam data in library
rm(list = ls())
library(kernlab)
library(class)
data(spam)
table(spam$type)

spam_c = spam[,c(1:57)]
table(spam$type)
summary(spam_c)

#Scalling data
spam_s=scale(spam_c,center = TRUE, scale = TRUE)
summary(spam_s)

####### 1) randomly select 100 spam and 100 non-spam as training set ######
#Setting classes index by refereing the original dataset
n = 100
index_Y=sample(which(spam$type=="spam"),n)
index_N=sample(which(spam$type=="nonspam"),n)
train = rbind(spam_s[index_Y,], spam_s[index_N,])

# get class factor for training data
train_label= factor(c(rep("spam",n), rep("nonspam",n)))

################# 2) 50 spam and 50 nonspam as the test set ###############
m = 50
test_sample = spam_s[-train,]
index_A=sample(which(spam$type=="spam"),m)
index_B=sample(which(spam$type=="nonspam"),m)
test = rbind(test_sample[index_A,], spam_s[index_B,])

# get class factor for training data
test_label = factor(c(rep("spam",m), rep("nonspam",m)))
                    
# check dimentions of training dataset &test dataset 
dim(train); dim(test)

library(class)

#convert training & test datasets to dataframes
#train_label <- as.data.frame(train)
#test_label <- as.data.frame(test)

# To check whether data contain missing values
anyNA(train); anyNA(test)

################## Use 1NN, 9NN and 25NN to classify the test set ##############
# to use kNN we need caret from library
library(caret)
# train() method for  for various algorithms data training, passing different parameter values for different algorithms
# before train() method, we need trControl controls the computational nuances of the train() method
#trcontrol <- trainControl(method = "repeatedcv", number = 10, repeats = 3) #repeated cross-validation method
set.seed(1234)  #make the random sampling reproducible
#knn_fit <- train(make ~., data = train, method = "knn",
                 #trControl=trcontrol, 
                 #preProcess = c("center", "scale"), # train and test have different lengths, we need to standardize data using caret’s preProcess() method
                 #tuneLength = 10)
knn_pred1=knn(train, test, cl=train_label, k = 1, prob = TRUE)
knn_pred9=knn(train, test, cl=train_label, k = 9, prob = TRUE)
knn_pred25=knn(train, test, cl=train_label, k = 25, prob = TRUE)

#presenting results
knn_pred1
knn_pred9
knn_pred25

# to see all probability
set.seed(1234)
knn3_pred=knn3Train(train, test, cl, k = 3, prob=TRUE)
attributes(knn3_pred)

################## draw classification boundary for kNN ##############
# scatter plot of the training data
yes=spam_s[spam$type=="spam",]
no=spam_s[spam$type=="nonspam",]
plot(yes$capitalAve,yes$capitalTotal,pch=1,col="deepskyblue1",
     xlab="Balance",ylab="Income",cex.lab=1.5,cex.axis=1.5,font.lab=2)
points(yes$balance,yes$income,pch=3,col="red")
legend(2.8,2.5, legend=c("No", "Yes"),
       col=c("deepskyblue1", "red"),pch=c(1,3),cex=1,text.font=4)

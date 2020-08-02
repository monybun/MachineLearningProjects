#########################################################################################################
############################## SMM636 Machine Learning (PRD2 A 2019/20) #################################
##############################            Group Coursework 4            #################################
##############################         Deadline April 17th 2020         #################################
##############################                 GROUP 1                  #################################
#########################################################################################################


####################################################################################
###################################   Question 2   #################################
####################################################################################
#install.packages("CRAN")
library(ISLR)
library(nnet)
library(caret)
library(class)
library(DMwR)
library(mlbench)
library(ROCR)
library(pROC)
library(tree)
library(rpart)
library(rpart.plot)
library(rattle)
credit=read.csv("credit_fraud.csv", header = TRUE)
attach(credit)
anyNA(credit)
####################################################################################
###############  Data Balance Checking and Data Processing  ########################
####################################################################################

print(table(credit$Class))

# 1 for fraudulent transactions and 0 otherwise

credit$Time = as.numeric(credit$Time)
credit$Class = as.factor(credit$Class)

# Balance Data
Credit= SMOTE(Class~.,credit,perc.over = 3700, k = 2, perc.under =103)

# Rename Class: "fine" = "0", "fraudulent"= "1"
levels(Credit$Class) <- c("fine", "fraudulent")
print(table(Credit$Class))

# Find Train and Test sets
trainIndex = createDataPartition(Credit$Class, p = 0.7, list = FALSE,times=1)
train = Credit[trainIndex,-31]; train.Label = Credit[trainIndex,31]
test = Credit[-trainIndex,-31]; test.Label = Credit[-trainIndex,31]

####################################################################################
#######################  Neural Network for Classification  ########################
####################################################################################
print("Method 1: Neutral Network Classification")

# Fit a Neural Network with 10 Hidden Units
train.label=class.ind(train.Label)
set.seed(1112)
nnModel1=nnet(train, train.label, size = 10, rang=0.01, maxit = 500, entropy=TRUE)
nnpred1=predict(nnModel1,test)
pred1=rep(c("fine"),length(test.Label))
pred1[nnpred1[,2]>0.5]="fraudulent"
Acc=mean(pred1 == test.Label)
print(Acc)

# Plot Accuracies Using Different Initial Values
r=c(0.005,0.01,0.02,0.03,0.04,0.05)
acc=vector("numeric",length(r))
set.seed(1234)
for(i in 1:length(r)){
  nnModel1=nnet(train, train.label, size = 10, rang=r[i], maxit = 500, entropy=TRUE)
  pred1_prob=predict(nnModel1,test)  #output layers: probabilities
  pred1=rep(c("fine"),length(test.Label))
  pred1[pred1_prob[,2]>0.5]="fraudulent"
  acc[i]=mean(pred1 == test.Label)
}
print(acc)
plot(r,acc,lty=1,lwd=2,type="b",main="Neutal Network",xlab="Initial Values",ylab="Accuracy",col = "purple")

# Plot of Accuracy Using Different Number of Hidden Units
a=c(10,15,20,25,30)
acc2=vector("numeric",length(a)) 
set.seed(2)
for(ii in 1:length(a)){
  nnModel1=nnet(train, train.label, size = a[ii], rang=0.01, maxit = 500,entropy=TRUE)
  pred1_prob=predict(nnModel1,test)
  pred1=rep(c("fine"),length(test.Label))
  pred1[pred1_prob[,2]>0.5]="fraudulent"
  acc2[ii]=mean(pred1 == test.Label)
}
print(acc2)
plot(a,acc2,lty=1,lwd=2,type="b",main="Neural Network",xlab="Hidden Units",ylab="Accuracy",col = "purple")

####################################################################################
#######################  kNN and LDA Classification  ###############################
####################################################################################
print("Method 2: kNN Classification")
set.seed(1656)
## Set Train Control:5-Fold Repeat 3 Times Cross-Validation
fitControl <- trainControl(method = "repeatedcv",number = 5,repeats = 3,classProbs = TRUE, summaryFunction = twoClassSummary)

#### kNN Classification
knnFit <- train(train, train.Label, method = "knn",trControl = fitControl,metric = "ROC", tuneLength=5)
print(knnFit)
knnPred = predict(knnFit, test)
print(confusionMatrix(knnPred,test.Label))
knnProbs <- predict(knnFit, test, type = "prob") 
print(head(knnProbs))
knnROC <- roc(predictor = knnProbs$fraudulent,response = test.Label,levels=rev(levels(test.Label)))
AUC_knn = knnROC$auc
print(AUC_knn)

#### LDA Classification
print("Method 3: LDA Classification")
ldaFit <- train(train, train.Label, method = "lda", trControl = trainControl(method = "none"))
print(ldaFit$finalModel)
ldaPred = predict(ldaFit,test)
print(confusionMatrix(ldaPred,test.Label))
ldaProbs <- predict(ldaFit, test, type = "prob")
print(head(ldaProbs))
ldaROC <- roc(predictor = ldaProbs$fraudulent, response = test.Label, levels = rev(levels(test.Label)))
AUC_lda = ldaROC$auc
print(AUC_lda)

#### Plot ROC curves for kNN and LDA
plot(knnROC,col = 'purple',type = "l",lwd=2,cex.lab=1,cex.axis=1, font.lab=1)
plot(ldaROC,add = TRUE,col = 'hotpink',lwd=2,cex.lab=1,cex.axis=1, font.lab=1)
legend("bottomright", legend = c("kNN", "LDA"),col = c("purple", "hotpink"), lty = c(1,1), cex = 1.5, text.font = 2)

####################################################################################
#######################  Tree Based Classification  ################################
####################################################################################
print("Method 4: Decision Tree Classification")
set.seed(99)
#### Decision Tree
tree.rpart <- train(train, train.Label, method = "rpart", metric = "ROC", trControl = fitControl,tuneLength=5)
# Presenting model
print(tree.rpart)
Tree.prune = prune.rpart(tree.rpart,cp=TRUE)
print(tree.rpart$finalModel)
fancyRpartPlot(Tree.prune$finalModel)

#### Random Forest
print("Method 5: Random Forest Classification")
rfFit <- train(train, train.Label, method="rf", metric="ROC", trControl=fitControl, tuneLength=5)
# Presenting results and selected model 
print(rfFit)
print(rfFit$finalModel)

#### Tree vs. RF
print("Decision Tree vs. Random Forest")
## Accuracy
 # Tree Accuracy
 treePred=predict(Tree.prune,test,type="raw")
 print(table(treePred,test.Label))
 Acc.tree=mean(treePred == test.Label)
 print(Acc.tree)
 # RF Accuracy
 rfPred=predict(rfFit,test,type="raw")
 print(table(rfPred,test.Label))
 Acc.rf=mean(rfPred == test.Label)
 print(Acc.rf)

## Variable Importance in RF
print(varImp(rfFit))
plot(varImp(rfFit))
## Plot ROC for Tree and RF
Treebest=predict(Tree.prune$finalModel,test,type="prob")
RFbest=predict(rfFit$finalModel,test,type="prob")

Treebest.ROCR=prediction(Treebest[,2],test.Label)  #Treebest[,1] is "fine"
RFbest.ROCR=prediction(RFbest[,2],test.Label)      #RFbest[,1] is "fine"

Treebest.rocplot=performance(Treebest.ROCR,"tpr","fpr")
RFbest.rocplot=performance(RFbest.ROCR,"tpr","fpr")

plot(Treebest.rocplot,col = 'purple',lwd=2,cex.lab=1,cex.axis=1, font.lab=1)
plot(RFbest.rocplot,add = TRUE,col = 'hotpink',lwd=2,cex.lab=1,cex.axis=1, font.lab=1)
legend("bottomright", legend = c("Best Tree Model(8 t-nodes)", " Best Random Forest Model "),
       col = c("purple", "hotpink"), lty = c(1,1), cex = 0.7, text.font = 1)
x=seq(0,1,0.01)
y=x
lines(x,y,lwd =2,lty=3,col="gray")
 #Compare AUC
 Treebest.AUC = roc(predictor=Treebest[,1],response=test.Label,levels=rev(levels(test.Label)))
 RFbest.AUC = roc(predictor=RFbest[,1],response=test.Label,levels=rev(levels(test.Label)))
 print(Treebest.AUC$auc)
 print(RFbest.AUC$auc)

####################################################################################
###################################  SVM  ##########################################
####################################################################################
print("Method 6: SVM Classification")
set.seed(188)
#### SVC
SVC = train(train, train.Label, method = "svmLinear",metric = "ROC",trControl = fitControl,tuneGrid = expand.grid(C = c(0.1,1,5,10,15,20)),tuneLength=5)
print(SVC)
## SVC Test Accuracy
SVC.Pred = predict(SVC,test)
Acc.SVC = mean(SVC.Pred == test.Label)
print(Acc.SVC)

#### SVM
SVM = train(train, train.Label, method = "svmRadial",metric = "ROC",trControl = fitControl,tuneGrid = expand.grid(sigma = 1/17,C = c(0.1,1,5,10,15,20)),tuneLength=5)
print(SVM)
## SVM Test Accuracy
SVM.Pred = predict(SVM,test)
Acc.SVM = mean(SVM.Pred == test.Label)
print(Acc.SVM)

#### Polynomial Kernel SVM
PolySVM = train(train, train.Label, method = "svmPoly",metric = "ROC",trControl = fitControl,tuneGrid = expand.grid(degree = 2,scale = 3,C = c(0.1,1,5,10,15,20)),tuneLength=5)
print(PolySVM)
## Ploynomial Kernel SVM Test Accuracy
PloySVM.Pred = predict(PolySVM,test)
Acc.PolySVM = mean(PloySVM.Pred == test.Label)
print(Acc.PolySVM)

#### Plot ROC Curves
SVC.Pred.Prob = predict(SVC, test, type = "prob")
SVM.Pred.Prob = predict(SVM, test, type = "prob")
PolySVM.Pred.Prob = predict(PolySVM, test, type = "prob")

SVC.ROC = roc(predictor = SVC.Pred.Prob$fraudulent,response = test.Label,levels = rev(levels(test.Label)))
SVM.ROC = roc(predictor = SVM.Pred.Prob$fraudulent,response = test.Label,levels = rev(levels(test.Label)))
PolySVM.ROC = roc(predictor = PolySVM.Pred.Prob$fraudulent,response = test.Label,levels = rev(levels(test.Label)))

plot.roc(SVC.ROC, main = "SVC,SVM & PolySVM ROC Plot",col = "darkblue",identity = TRUE,identity.lty = "dashed",identity.lwd = 2)
plot.roc(SVM.ROC,add = TRUE, col = "hotpink",identity.lwd = 3)
plot.roc(PolySVM.ROC,add = TRUE, col = "green",identity.lwd = 1.5)
legend("bottomright",legend = c("SVC", "SVM", "PolySVM"),col = c("darkblue", "hotpink", "green"),lty = c(1,1),cex = 1, text.font = 1)

## Compare AUC Values
SVC.AUC = SVC.ROC$auc
SVM.AUC = SVM.ROC$auc
PolySVM.AUC = PolySVM.ROC$auc
print(c(SVC.AUC,SVM.AUC,PolySVM.AUC))


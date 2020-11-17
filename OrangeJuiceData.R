
######################################################################################
#################################  Loading Packages ##################################
#install.packages("e1071")
#install.packages("ISLR")
#install.packages("CRAN")
#instal.packages("plotROC")
library(e1071)
library(ISLR)
library(caret)
library(ROCR)
library(kernlab)
library(caret)
library(class)
library(MASS)
library(pROC)
library(DMwR)
library(tree)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
#library(randomForest)
#### Load OJ data from the ISLR package
data(OJ)

#################################    Prepare Data    #################################
set.seed(1112)
#### Check N/A
anyNA(OJ)
#### Convert non-numerical values into numericals
OJ$Store7 = as.numeric(OJ$Store7)
#### Check data balance
print(table(OJ$Purchase))
#### Balance dataset using SMOTE
OJ_Bal= SMOTE(Purchase~., OJ,perc.over = 200, k = 5, perc.under = 150)
print(table(OJ_Bal$Purchase))
#### Check N/A
anyNA(OJ_Bal)

#### Obtain training and test sets
trainIndex = createDataPartition(OJ_Bal$Purchase, p = 0.7, list = FALSE)
train = OJ_Bal[trainIndex,]
test = OJ_Bal[-trainIndex,]

######################################################################################
######################  (1) Support(Soft) Margin Classifier  #########################

#### Tune cost to (0.01,0.1,1,10)
set.seed(1001)
fitcontrol = trainControl(method = "cv",number = 5,summaryFunction = twoClassSummary,classProbs = TRUE)
SVC.tune=train(Purchase ~.,data = train,method = "svmLinear",metric = "ROC",tuneGrid = expand.grid(C = c(0.01,0.1,1,10)),
               preProcess = c("center", "scale"),trControl = fitcontrol)
print(SVC.tune$finalModel)
print(SVC.tune)

#### Predict the test instances using the selected model
pred.SVC=predict(SVC.tune,test)
print(table(pred.SVC, test[,1]))

#### Calculate error rate
Error.SVC = mean(pred.SVC!=test[,1])
print(Error.SVC)

######################################################################################
#############################  (2) Radial Kernel SVM  ################################

#### Define model's default value of gamma
model=svm(train[,-1],train[,1])
print(model$gamma)
Gamma.Default = model$gamma

#### Tune using radial kernel and default gamma
set.seed(100)
SVM.tune = train(Purchase ~.,data = train,method = "svmRadial",metric = "ROC",preProcess = c("center", "scale"),
                 tuneGrid = expand.grid(sigma = 1/17,C = c(0.01,0.1,1,10) ),trControl = fitcontrol)
print(SVM.tune$finalModel)
print(SVM.tune)

#### Predict the training instances using the selected model
pred.SVM=predict(SVM.tune,test)
print(table(pred.SVM, test[,1]))

#### Calculate error rate
Error.SVM = mean(pred.SVM!=test[,1])
print(Error.SVM)

######################################################################################
############################# (3) Polynomial Kernel SVM ##############################

#### Tune using polynomial kernel
set.seed(996)
PolySVM.tune = train(Purchase ~.,data = train,method = "svmPoly",metric = "ROC",preProcess = c("center", "scale"),
                 tuneGrid = expand.grid(degree=2,scale=1,C = c(0.01,0.1,1,10) ),trControl = fitcontrol)
print(PolySVM.tune$finalModel)
print(PolySVM.tune)

#### Predict the training instances using the best model
pred.PolySVM=predict(PolySVM.tune,test,decision.values = TRUE)
print(table(pred.PolySVM, test[,1]))

#### Calculate error rate
Error.PolySVM = mean(pred.PolySVM!=test[,1])
print(Error.PolySVM)

######################################################################################
#############################    (4) Plot ROC Curves    ##############################

#### Predict the test instances using the best models
pred.SVC.test=predict(SVC.tune, test,type="prob")
pred.SVM.test=predict(SVM.tune, test,type="prob")
pred.PolySVM.test=predict(PolySVM.tune, test,type="prob")
SVC.ROCR=prediction(pred.SVC.test[,2],test[,1])
SVM.ROCR=prediction(pred.SVM.test[,2],test[,1])
PolySVM.ROCR=prediction(pred.PolySVM.test[,2],test[,1])

#### Generate ROC curves
SVC.rocplot=performance(SVC.ROCR,"tpr","fpr")
SVM.rocplot=performance(SVM.ROCR,"tpr","fpr")
PolySVM.rocplot=performance(PolySVM.ROCR,"tpr","fpr")

#### Plot
xlab='False positive rate (1-specificity)'
ylab='True positive rate (sensitivity)'
plot(SVC.rocplot,col="hotpink",lwd=2,cex.lab=1,cex.axis=1,xlab=xlab,ylab=ylab,main="Test Set ROC")
plot(SVM.rocplot,col="blue",add=TRUE,lwd=2,cex.lab=1,cex.axis=1)
plot(PolySVM.rocplot,col="orange",add=TRUE,lwd=2,cex.lab=1,cex.axis=1)
legend('bottomright',legend = c('SVC','RadialSVM','PolySVM'),col = c('hotpink', 'blue','orange'),lty = c(1,1),cex = 1,text.font = 2)
# Adding diaganal line
x=seq(0,1,0.01)
y=x
lines(x,y,lwd =2,lty=2)

#### Calculate AUC for each curve
SVC.AUC = roc(predictor=pred.SVC.test[,2],response=test[,1],levels=rev(levels(test[,1])))
SVM.AUC = roc(predictor=pred.SVM.test[,2],response=test[,1],levels=rev(levels(test[,1])))
PolySVM.AUC = roc(predictor=pred.PolySVM.test[,2],response=test[,1],levels=rev(levels(test[,1])))
print(SVC.AUC$auc)
print(SVM.AUC$auc)
print(PolySVM.AUC$auc)

######################################################################################
##############################    (5) Decision Tree    ###############################

#### Fit tree to training set with CV selection in caret
# Repeate 10-fold CV 3 times to tune the parameter cp
fitcontrol=trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(190) 
train.rpart=train(train[,-1], train[,1], method = "rpart", preProcess = c("center", "scale"), trControl = fitcontrol,tuneLength=5)
# Presenting model
print(train.rpart)

#### Prune and plot selected model
set.seed(768)
Tree.prune = prune.rpart(train.rpart,cp=TRUE)
fancyRpartPlot(Tree.prune$finalModel)

#### Compute pruned test error rate 
print(Tree.prune)
pred.prune=predict(Tree.prune,test[,-1],type="raw")
print(table(pred.prune,test[,1]))
error.test.pruned=mean(pred.prune!=test[,1])
print(error.test.pruned)

######################################################################################
##############################    (6) Random Forest    ###############################

#### Use caret to fit RF to find optimal model
# repeate 10-fold CV 3 times to tune the parameter mtry
RFfitControl=trainControl( method = "repeatedcv", number = 10,repeats = 3)
# Train with pecify mtyr value
mtryGrid=expand.grid(mtry=c(1,2,3,4,5,6))
set.seed(89) 
train.RF=train(train[,-1], train[,1],method="rf",metric="Accuracy", preProcess = c("center", "scale"), tuneGrid=mtryGrid, trControl=RFfitControl)
# Presenting results and selected model 
print(train.RF)
print(train.RF$finalModel)

#### Compute test error rates using random forest
# with mtry from 1 to 6
mtry=c(1,2,3,4,5,6)
for (enter.mtry in mtry){
  train.rf=randomForest(Purchase~.,data=train,mtry=enter.mtry,importance=TRUE,ntree=100)
  pred.rf=predict(train.rf,newdata=test[,-1])
  error.n.mtry=mean(pred.rf!=test[,1])
  # present each test error rate
  print(error.n.mtry)
}

#### Plot variable importance with selected model(best test error)
print(varImp(train.RF))
plot(varImp(train.RF))

######################################################################################
########################## (7) Combined ROC for DT and RF  ###########################

#### Plot ROC curves for selected models in Decision Tree and Random Forest
pred.Treebest=predict(Tree.prune$finalModel,test[,-1],type="prob")
pred.RFbest=predict(train.RF$finalModel,test[,-1],type="prob")
Treebest.ROCR=prediction(pred.Treebest[,2],test[,1])
RFbest.ROCR=prediction(pred.RFbest[,2],test[,1])
# Generate ROC curves
Treebest.rocplot=performance(Treebest.ROCR,"tpr","fpr")
RFbest.rocplot=performance(RFbest.ROCR,"tpr","fpr")
# Plot ROC curves
plot(Treebest.rocplot,col = 'purple',lwd=2,cex.lab=1,cex.axis=1, font.lab=1,xlab=xlab,ylab=ylab)
plot(RFbest.rocplot,add = TRUE,col = 'darkblue',lwd=2,cex.lab=1,cex.axis=1, font.lab=1)
legend("bottomright", legend = c("Best Tree Model(8 t-nodes)", " Best Random Forest Model "),
       col = c("purple", "darkblue"), lty = c(0.65,0.65), cex = 0.8, text.font = 1)
lines(x,y,lwd =2,lty=2)

#### Calculate auc for each curve
Treebest.AUC = roc(predictor=pred.Treebest[,2],response=test[,1],levels=rev(levels(test[,1])))
RFbest.AUC = roc(predictor=pred.RFbest[,2],response=test[,1],levels=rev(levels(test[,1])))
print(Treebest.AUC$auc)
print(RFbest.AUC$auc)


#####################################################################################################################
#############################       Load the Pima Indians Diabetes Data      ########################################
##################################################################################################################### 

library(mlbench)
library(caret)
library(class)
library(MASS)
library(ROCR)
library(pROC)
library(DMwR)
library(rpart)
library(rpart.plot)
library(rattle)
library(e1071)

data("PimaIndiansDiabetes")
str(PimaIndiansDiabetes)
table(PimaIndiansDiabetes$diabetes) # The table shows that the data is imbalanced

#####################################################################################################################
###################################################### PART i) ######################################################
#####################################################################################################################

# [1] Randomly split the data 20 times, allocating 70% to the training set
set.seed(165)
N = 20
AUC_knn = vector("numeric", N)
AUC_lda = vector("numeric", N)

Index = createDataPartition(PimaIndiansDiabetes$diabetes, p = 0.70, list = FALSE, times = N)

# [2] Create train and test sets
for (i in 1:N){
  train_feature = PimaIndiansDiabetes[Index[,i], -9]
  train_label = PimaIndiansDiabetes$diabetes[Index[,i]]
  
  test_feature = PimaIndiansDiabetes[-Index[,i], -9]
  test_label = PimaIndiansDiabetes$diabetes[-Index[,i]]
  
# [3] Set up train control
  fitControl <- trainControl(method = "repeatedcv",
                             number = 5,
                             repeats = 5,
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary,
                             sampling = "smote"
                             )
  
# [4] Training process for kNN. Evaluate for k = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
  knnFit <- train(train_feature, train_label, method = "knn",
                    trControl = fitControl,
                    metric = "ROC",
                    preProcess = c("center", "scale"),
                    tuneGrid = expand.grid(k = seq(3, 21, by =2))
                  )
  
  knnPred = predict(knnFit, test_feature)
  
# [5] Training process for LDA.
  ldaFit <- train(train_feature, train_label, method = "lda",
                  trControl = trainControl(sampling = "smote")
                  )
  
  ldaPred = predict(ldaFit,test_feature) 
  
# [6] Record 20 AUC values for kNN using the optimal k-value
  knnProbs <- predict(knnFit, test_feature, type = "prob") 
  head(knnProbs)
  knnAUC <- roc(predictor = knnProbs$neg, 
                   response = test_label,
                   levels=rev(levels(test_label))
                )
  
  AUC_knn[i] = knnAUC$auc
  
# [7] Record 20 AUC values for LDA
  ldaProbs <- predict(ldaFit, test_feature, type = "prob")
  head(ldaProbs)
  ldaAUC <- roc(predictor = ldaProbs$neg,
                   response = test_label,
                   levels = rev(levels(test_label))
                )
  
  AUC_lda[i] = ldaAUC$auc
  
}

print(AUC_knn)
print(AUC_lda)

# Plot the 20 AUC vectors of kNN and LDA
plot(AUC_lda,
     type = "l",
     ylab = "AUC",
     ylim = c(0.7,1),
     main = "AUC: kNN vs. LDA",
     col = "hotpink"
)

lines(AUC_knn, col = c("purple"))

legend("topright", legend = c("kNN", "LDA"),
       col = c("purple", "hotpink"),  lty = c(1,1), cex = 1, text.font = 2
)

# kNN confusion matrix
knnConfMatrix = confusionMatrix(knnPred, test_label)
print(knnConfMatrix)

# LDA confusion matrix
ldaConfMatrix = confusionMatrix(ldaPred, test_label)
print(ldaConfMatrix)

#####################################################################################################################
###################################################### PART ii) #####################################################
#####################################################################################################################

# [1] Retrieve the first random split data
train_feature_1 = PimaIndiansDiabetes[Index[,1], -9]
train_label_1 = PimaIndiansDiabetes$diabetes[Index[,1]]

test_feature_1 = PimaIndiansDiabetes[-Index[,1], -9]
test_label_1 = PimaIndiansDiabetes$diabetes[-Index[,1]]

# [2] Obtrain the area of the ROC curve for kNN
knnProbs_1 <- predict(knnFit, test_feature_1, type = "prob") 
head(knnProbs_1)
knnROC_1 <- roc(predictor = knnProbs_1$neg, 
               response = test_label_1,
               levels=rev(levels(test_label_1))
               )
AUC_knn_1 = knnROC_1$auc
print(AUC_knn_1)

# [3] Obtain the area of the ROC curve for LDA
ldaProbs_1 <- predict(ldaFit, test_feature_1, type = "prob")
head(ldaProbs_1)
ldaROC_1 <- roc(predictor = ldaProbs_1$neg,
               response = test_label_1,
               levels = rev(levels(test_label_1))
               )
AUC_lda_1 = ldaROC_1$auc
print(AUC_lda_1)

# [4] Plot the ROC curve for kNN and LDA on one plot for the first random split data
plot(knnROC_1,
     main = "ROC Curve",
     col = "purple"
     )

lines(ldaROC_1, col = "hotpink")

legend("bottomright", legend = c("kNN", "LDA"),
       col = c("purple", "hotpink"), lty = c(1,1), cex = 1, text.font = 2
       )

#####################################################################################################################
###################################################### PART iii) ####################################################
#####################################################################################################################

# [1] Two boxplots for the 20 AUC values of kNN and LDA
boxplot(AUC_knn, AUC_lda, 
        main = "AUC: kNN vs. LDA",
        ylab = "Classification accuracy",
        xlab = "Classification method",
        names = c("kNN", "LDA"),
        col = c("purple", "hotpink")
        )

# [2] Calculate time for running kNN and LDA
RunningTime = c("knnFit", "knnPred", "knnProbs", "knnAUC",
                "ldaFit", "ldaPred", "ldaProbs", "ldaAUC")
RunningTime.factor = factor(RunningTime)
RunningTime.No = as.numeric(RunningTime.factor)

RunningTime_Func = vector("numeric",8)
for (ii in RunningTime.No){
  minute <- function(ii) { Sys.sleep(60) }
  start_time <- Sys.time()
  minute()
  end_time <- Sys.time()
  RunningTime_Func[ii] = end_time - start_time
}
print(RunningTime_Func)

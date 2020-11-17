#################################################################################
################################ Loading Packages ###############################
#install.packages("kernlab")
#install.packages("class")
#install.packages("caret")
#install.packages("shiny")
#installed.packages("shinythemes")
library(shiny)
library(mlbench)
library(caret)
library(tree)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(gbm)
library(shinythemes)
data("PimaIndiansDiabetes")

#################################################################################
################### Design Users Interface Page of Shiny App  ###################
ui <- fluidPage(
  
#### Making the page "paper-like" ####
  theme = shinythemes::shinytheme("paper"),
  
  h3("Group 1"),
  titlePanel("Tree Based Classification Methods"),
  
  sidebarLayout(
    
##### Disign sidebar panel ####
    sidebarPanel(
      
      # Using checkboxes to interact with users, giving them choices for variables inputs
      checkboxGroupInput(inputId = "variables",
                         label = "Select variables",
                         choices = c(colnames(PimaIndiansDiabetes[,1:8])), # Excluding "diabetes" column
                         selected = c(colnames(PimaIndiansDiabetes[,1:2]))), # Default selections when operating the App
      width = 2),
    
#### Disign main panel ####
    mainPanel(
       tabsetPanel(type = "tabs",
                  tabPanel("Decision Tree",
                           h2("Classification Tree"),
                           h6("A decision tree is a supervised learning predictive model that uses a set of binary rules to calculate a target value.
                           The grapgh below is a classification tree with qualitative, categorical target variable. The algorithm of the decision 
                           tree recursively partitions the data, which means repeatedly partitioning the data into multiple subspaces, such that the 
                           outcome in each final subspace is as homogenous as possible."),
                           plotOutput("PIDTree.prune"),
                           br(),
                           p("The above classification tree is a graphical representation of the data partitioning of the data set “PimaIndiansDiabetes”
                           using training data. The purpose is to predict whether a patient will test positive or negative to diabetes. The predictions
                           are based on the most commonly occurred class of training observations in the region to which the observation belongs to. 
                           The edges of the graph represent the decision rule, the nodes on the graph represent the classes, which are either “positive”
                           (blue) or “negative” (orange), and the darker the color is, the higher proportion of patients that will test positive/negative
                           compared to the other class. The nodes also include the proportion of positives or negatives in each class in the sub-dataset,
                           as well as the weight of the sub-dataset in the whole dataset.")
                           ),
                  tabPanel("Random Forest",
                           h2("Random Forest"),
                           h6("Random Forest is an approach to improve predictions resulting from a decision tree by decorrelating the trees. In a Random Forest process, a number
                           of trees are built on bootstrapped training samples. The decision trees are then built, and each time a split in a tree is
                           considered, a random sample of “m” predictors is chosen as split candidates from the full set of “p” predictors. The split
                           is allowed to use only one of those “m” predictors. Random Forest is an effective method of to estimate missing data and 
                           maintain accuracy when a large proportion of the data is missing."),
                           plotOutput("PID.rf"),
                           br(),
                           p("The Random Forest model is a graphical representation of the importance of the variables, where the ranking is based on the number of times the is 
                             used in the tree, “Mean decrease accuracy” and “Mean decrease Gini”. "),
                           plotOutput("MeanDecrease"),
                           p("Decision Trees are plotted based on binary splits. There are two criterions for binary splits—Classification Accuracy and Gini Index.
                           MeanDecreaseAccuracy plot focuses on illustrating importance of variables to classification accuracy, whereas the MeanDecreaseGini plot stresses the importance of 
                           variables to the “node purity” of decision trees. The MeanDecreaseAccuracy states the proportion of observations that are incorrectly classified by removing respective features.
                           Gini Index is used to evaluate the quality of a particular split, and it is sensitive to node purity.
                           The value of Gini is [0,1], and 0 stands for the highest node purity. 
                           It measures the average gain of purity by splits of a given variable.If a variable is useful, it tends to split mixed labeled nodes into pure single class nodes. 
                           The MeanDecreaseGini tells us the average (mean) of a variable’s total decrease in node impurity, weighted by the proportion of samples reaching that node in each
                           individual decision tree in the random forest.")
                           ),
                  tabPanel("Boosting",
                           h2("Boosting"),
                           h6("Boosting is another approach to improve predictions resulting from a decision tree. In a Boosting process, the tree is grown
                           sequentially, meaning that each tree is grown using information from a previously grown tree. Each tree is therefore fitted on
                           a modified version of the original dataset. Boosting is useful for large datasets where the large original tree is expected to
                           be very complex."),
                           plotOutput("PID.boosting"),
                           p("The Boosting model generates 5000 trees and has the shrinkage parameter set as 0.01 and the total number of splits (interaction depth)
                           is 2. The feature importance is plotted above. The feature importance graph shows the most important variables in a ranked order (from 
                           most important at the top to least important at the bottom).")
                           ),
                  tabPanel("Accuracy",
                           h2("Accuracy Levels"),
                           h6("Accuracy Level of Pruned Decision Tree:"),
                           textOutput("acc.prune"),
                           br(),
                           h6("Accuracy Level of Random Forest:"),
                           textOutput("acc.rf"),
                           br(),
                           h6("Accuracy level of Boosting:"),
                           textOutput("acc.boost"),
                           br(),
                           plotOutput("AccBoxPlot"),
                           br(),
                           h6("The overall prediction accuracy increases as 
                              we introduce more variables for all the three classifications. 
                              From the plot, MeanDecreaseAccuracy, in Random Forest Tab. 
                              Variables, Glucose, mass and age, have the most significant 
                              influence on accuracy. If we excluded these three variables 
                              at the beginning, the arruracy for the three classifications 
                              is all below 70%, especially the accuracy for the decision tree 
                              has the lowest 62%. "),
                           h6("As we include age and mass, accuracy of random forest and 
                              boosting increase to around 70%, but the accuracy of decision 
                              tree is still below 65%. The most significant variable, glucose, 
                              brings the accuracy to 75% under boosting, which is the highest 
                              accuracy under the condition that all variables included."),
                           h6("Generally speaking, decision trees perform better than the 
                              other two at the very beginning. As we introduced more variables, 
                              random forest becomes outperformed. However, boosting 
                              stands out lastly. ")
                           )
                  ),
       width = 10
    )
  )
)

#################################################################################
######################## Design Server Page of Shiny App ########################
server <- function(input, output) {

#### Using reactive intput to simplify data extracting for each output
    dataInput <- reactive({
    set.seed(125)  
    var.col = c(match(input$variables, names(PimaIndiansDiabetes)))
    train.index = createDataPartition(PimaIndiansDiabetes[,ncol(PimaIndiansDiabetes)], p = 0.70, list = FALSE)
    train = PimaIndiansDiabetes[train.index, c(var.col, 9)]
    test = PimaIndiansDiabetes[-train.index, c(var.col, 9)]
  
    })
    
##################### OUTPUT 1: Classification tree ###########################
  output$PIDTree.prune <- renderPlot({dataInput()
    
    # Build decision tree
    PIDTree.rpart = rpart(diabetes ~ ., dataInput())
    PIDTree.prune = prune.rpart(PIDTree.rpart, cp = NULL)
    fancyRpartPlot(PIDTree.prune,palettes = "OrRd")
  })
  
################# OUTPUT 2: Variable importance measure(1) ###################
####   Random Forest-Variable importance graph
  output$PID.rf <- renderPlot({dataInput()
    
    fitControl = trainControl(
      method = "repeatedcv",
      number = 5,
      repeats = 3)
    
    set.seed(125)
    Fit.rf = train(diabetes ~ .,
                   data = dataInput(),
                   method = "rf",
                   metric = "Accuracy",
                   trControl = fitControl,
                   tuneLength = 5)
    PID.rf = varImp(Fit.rf)
    plot(PID.rf, main = "Variable Importance Measure", ylab = "Variables")
    
  })
  
################# OUTPUT 3: Variable importance measure(2): #################
####   Random Forest-Mean decrease graphs
  output$MeanDecrease <- renderPlot({dataInput()
    
    pima.bag = randomForest(diabetes~.,
                            data = dataInput(),
                            mtry = 2,
                            importance = TRUE,
                            ntree = 5000)
    varImpPlot(pima.bag)
    
  })
  
########################## OUTPUT 4: Boosting  #############################
  output$PID.boosting <- renderPlot({
    
    set.seed(125)
    var.col = c(match(input$variables, names(PimaIndiansDiabetes)))
    train.index = createDataPartition(PimaIndiansDiabetes[,ncol(PimaIndiansDiabetes)], p = 0.70, list = FALSE)
    train = PimaIndiansDiabetes[train.index, c(var.col, 9)]
    test = PimaIndiansDiabetes[-train.index, c(var.col, 9)]
    
#### Boosting
    train[,ncol(train)] = ifelse(train$diabetes == "neg", 0, 1)
    test[,ncol(test)]=ifelse(test$diabetes == "neg", 0, 1)
    
    diabetes.boost=gbm(diabetes ~ .,data = train,
                       distribution = "bernoulli", 
                       n.trees = 5000,
                       interaction.depth = 2,
                       shrinkage = 0.01)
    summary(diabetes.boost)
  })    
  
################## OUTPUT 5: Accuracy Calculations ##################
#### 5.1 Accuracy for pruned decision tree
output$acc.prune <- renderText({
  
  set.seed(125)
  var.col = c(match(input$variables, names(PimaIndiansDiabetes)))
  train.index = createDataPartition(PimaIndiansDiabetes[,ncol(PimaIndiansDiabetes)], p = 0.70, list = FALSE)
  train = PimaIndiansDiabetes[train.index, c(var.col, 9)]
  test = PimaIndiansDiabetes[-train.index, c(var.col, 9)]
    
  PIDTree.rpart = rpart(diabetes ~ ., train)
  PIDTree.prune = prune.rpart(PIDTree.rpart, cp = NULL)
  pred.prune = predict(PIDTree.prune,test[,-ncol(test)],type = "class")
  acc.prune = mean(pred.prune==test[,ncol(test)])
})

#### 5.2 Accuracy for random forest
output$acc.rf <- renderText({
  
   set.seed(125)
   var.col = c(match(input$variables, names(PimaIndiansDiabetes)))
   train.index = createDataPartition(PimaIndiansDiabetes[,ncol(PimaIndiansDiabetes)], p = 0.70, list = FALSE)
   train = PimaIndiansDiabetes[train.index, c(var.col, 9)]
   test = PimaIndiansDiabetes[-train.index, c(var.col, 9)]
   
    pima.bag = randomForest(diabetes~.,
                              data = train,
                              mtry = 2,
                              importance = TRUE,
                              ntree = 5000)
    
    pred.bag = predict(pima.bag,test[,-ncol(test)])
    acc.rf = mean(pred.bag==test[,ncol(test)])
})    

#### 5.3 Accuracy for boosting
output$acc.boost <- renderText({
  
    set.seed(125)
    var.col = c(match(input$variables, names(PimaIndiansDiabetes)))
    train.index = createDataPartition(PimaIndiansDiabetes[,ncol(PimaIndiansDiabetes)], p = 0.70, list = FALSE)
    train = PimaIndiansDiabetes[train.index, c(var.col, 9)]
    test = PimaIndiansDiabetes[-train.index, c(var.col, 9)]
    train[,ncol(train)] = ifelse(train$diabetes == "neg", 0, 1)
    test[,ncol(test)]=ifelse(test$diabetes == "neg", 0, 1)
    
    diabetes.boost=gbm(diabetes ~ .,data = train,
                         distribution = "bernoulli", 
                         n.trees = 5000,
                         interaction.depth = 2,
                         shrinkage = 0.01)
    
    pred.boost = predict(diabetes.boost,
                         test[,-ncol(test)],
                         n.trees = 5000,
                         type = "response")
    
    pred.boost=ifelse(pred.boost<0.5,0,1)
    acc.boost = mean(pred.boost==test[,ncol(test)])
})

#### 5.4 Accuracy bar plot
output$AccBoxPlot <- renderPlot({input$variables
  
  set.seed(125)
  var.col = c(match(input$variables, names(PimaIndiansDiabetes)))
  train.index = createDataPartition(PimaIndiansDiabetes[,ncol(PimaIndiansDiabetes)], p = 0.70, list = FALSE)
  train = PimaIndiansDiabetes[train.index, c(var.col, 9)]
  test = PimaIndiansDiabetes[-train.index, c(var.col, 9)]
  
  PIDTree.rpart = rpart(diabetes ~ ., train)
  PIDTree.prune = prune.rpart(PIDTree.rpart, cp = NULL)
  pred.prune = predict(PIDTree.prune,test[,-ncol(test)],type = "class")
  acc.prune = mean(pred.prune==test[,ncol(test)])
  
  pima.bag = randomForest(diabetes~.,
                          data = train,
                          mtry = 2,
                          importance = TRUE,
                          ntree = 5000)
  pred.bag = predict(pima.bag,test[,-ncol(test)])
  acc.rf = mean(pred.bag==test[,ncol(test)])
  
  train[,ncol(train)] = ifelse(train$diabetes == "neg", 0, 1)
  test[,ncol(test)]=ifelse(test$diabetes == "neg", 0, 1)
  
  diabetes.boost=gbm(diabetes ~ .,data = train,
                     distribution = "bernoulli", 
                     n.trees = 5000,
                     interaction.depth = 2,
                     shrinkage = 0.01)
  
  pred.boost = predict(diabetes.boost,
                       test[,-ncol(test)],
                       n.trees = 5000,
                       type = "response")
  
  pred.boost=ifelse(pred.boost<0.5,0,1)
  acc.boost = mean(pred.boost==test[,ncol(test)])
  
  counts = c(acc.prune, acc.rf, acc.boost)
  barplot(counts,
          names.arg = c("Pruned Tree", "Random Fores", "Boosting"),
          col = c("#FFCC99", "#FF9966", "#FF9999"),
          main = "Accruacy Comparison",
          ylab = "Accuracy Level",
          xpd = FALSE,
          ylim = c(0.6,0.8)
          )

  })
  
}

shinyApp(ui = ui, server = server)

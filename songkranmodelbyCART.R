setwd("~/R-Project/ProjectDT")
Sys.setenv('JAVA_HOME'="C:/Program Files/Java/jdk-15.0.2/") 
require("devtools")
library(caret)
library(rJava)
library(RWeka)
library(ROCR)
library(partykit)
library(rattle)
library(RColorBrewer)
library(rpart)
library(rpart.plot)
library(C50)

#IMPORT DATASET
df <- read.csv("songkran_cat_final.csv",header= T, stringsAsFactors = T, na.strings="")

#DROP USELESS COLUMNS
df <- df[,-c(1)]

#CHECK DATASET
head(df)

# Data Partitioning
set.seed(221)
trainIndex <- createDataPartition(df[,length(df)],p = 0.7,list = FALSE, times = 1)
train <- df[trainIndex,, drop = FALSE]
test <- df[-trainIndex,, drop = FALSE]

# Model Training "CART" using Parameter method = "rpart"
set.seed(1234)
flds <- createFolds(train$Status, k = 10, list = TRUE, returnTrain = FALSE) #10-fold
modelCART <- caret::train(Status~., train, method= 'rpart', tuneLength = 5,
                          trControl = trainControl(method="cv",indexOut=flds,classProbs = TRUE),na.action=na.pass)
# Plot Tree of New Model
fancyRpartPlot(modelCART$finalModel,cex=0.7)

# Result of Decision Tree model using Algorithm CART (rpart) | Gini Index
resultCART <- predict(modelCART, test)
test$Status <- as.factor(test$Status)
cfCART <- confusionMatrix(resultCART, test$Status, positive = "Dead")
cfCART

#Manually Overall Data
cat('Overall accuracy is', cfCART$overall['Accuracy'])
cat('Sensitivity is',cfCART$byClass['Sensitivity'])
cat('Specificity is',cfCART$byClass['Specificity']) #**
cat('Precision is',cfCART$byClass['Precision'])
cat('F1-score is',cfCART$byClass['F1'])
cat('G-MEAN is',sqrt(cfCART$byClass['Sensitivity']*cfCART$byClass['Specificity']))

# AUC Values for model CART
resultsCARTAUC <- predict(modelCART, test, type = 'prob')[,2]
resultsCARTAUC <-as.numeric(resultsCARTAUC) #factor -> numeric
test$Status <- as.numeric(test$Status)
predCART <- ROCR::prediction(resultsCARTAUC, test$Status)
aucCART <- ROCR::performance(predCART, measure = "auc")
cat("AUC is", unlist(aucCART@y.values),"\n")

# Plot ROC FOR model CART
perfCART <- performance(predCART, 'tpr', 'fpr')
plot(perfCART, lty=3, cex = 2,main="ROC of CART model")

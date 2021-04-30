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

# Model Training "C4.5" using Parameter method = "J48"
set.seed(1234)
flds <- createFolds(train$Status, k = 10, list = TRUE, returnTrain = FALSE) #10-fold
modelC45 <- caret::train(Status~., train, method= 'J48', tuneLength = 5,
                          trControl = trainControl(method="cv",indexOut=flds,classProbs = TRUE),na.action=na.pass)
# Plot Tree of New Model
fancyRpartPlot(modelC45$finalModel,cex=0.75)

# Result of Decision Tree model using Algorithm C4.5 (J48) | Entropy, Information Gain
resultC45 <- predict(modelC45, test)
test$Status <- as.factor(test$Status)
cf45 <- confusionMatrix(resultC45, test$Status, positive = "Dead")
cf45

#ALL DATA
cat('Overall accuracy is', cf45$overall['Accuracy'])
cat('Sensitivity is',cf45$byClass['Sensitivity'])
cat('Specificity is',cf45$byClass['Specificity']) #**
cat('Precision is',cf45$byClass['Precision'])
cat('F1-score is',cf45$byClass['F1'])
cat('G-MEAN is',sqrt(cf45$byClass['Sensitivity']*cf45$byClass['Specificity']))

# AUC Values for model C4.5
resultsC45AUC <- predict(modelC45, test, type = 'prob')[,2]
resultsC45AUC <-as.numeric(resultsC45AUC) #factor -> numeric
test$Survived <- as.numeric(test$Status)
predC45 <- ROCR::prediction(resultsC45AUC, test$Status)
aucC45 <- ROCR::performance(predC45, measure = "auc")
cat("AUC is", unlist(aucC45@y.values),"\n")

# Plot ROC FOR model C4.5
perfC45 <- performance(predC45, 'tpr', 'fpr')
plot(perfC45, lty=3, cex = 2,main="ROC of C4.5 model")

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
df <- read.csv("songkran_cat_finalv2.csv",header= T, stringsAsFactors = T, na.strings="")

#DROP USELESS COLUMNS
#df <- df[,-c(1)] # Drop useless columns
#CHECK DATASET
head(df)
sum(is.na(dt))
# Data Partitioning
set.seed(221)
trainIndex <- createDataPartition(df[,length(df)],p = 0.7,list = FALSE, times = 1)
train <- df[trainIndex,, drop = FALSE]
test <- df[-trainIndex,, drop = FALSE]

# Model Training "CART" using rpart
flds <- createFolds(train$Status, k = 10, list = TRUE, returnTrain = FALSE) #10-fold
modelCART <- caret::train(Status~., train, method= 'rpart', tuneLength = 5,
                          trControl = trainControl(method="cv",indexOut=flds,classProbs = TRUE),na.action=na.pass)

# Plot Tree of New Model
fancyRpartPlot(modelCART$finalModel,cex=0.5)

# Result of modelCART
resultCART <- predict(modelCART, test)
test$Status <- as.factor(test$Status)
cfCART <- confusionMatrix(resultCART, test$Status, positive = "Dead")
cfCART
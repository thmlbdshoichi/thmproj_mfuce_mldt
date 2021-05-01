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
#df <- df[,-c(1)]

# Data Partitioning
set.seed(221)
trainIndex <- createDataPartition(df[,length(df)],p = 0.7,list = FALSE, times = 1)
train <- df[trainIndex,, drop = FALSE]
test <- df[-trainIndex,, drop = FALSE]

ctrl = C5.0Control(subset = TRUE,
                   noGlobalPruning = FALSE,
                   CF = 0.10,
                   minCases = 200,
                   sample = 0.7,
                   seed = 20,
                   earlyStopping = TRUE,
                   label = "final-model")

modelCARTC50 <- C5.0(x = train[,-c(13)],y = train$Status,control = ctrl)

summary(modelCARTC50)

# Result of Decision Tree model using Algorithm C5.0 (C5.0) | Information Gain?
resultC50 <- predict(modelCARTC50, test)
test$Status <- as.factor(test$Status)
cfC50 <- confusionMatrix(resultC50, test$Status, positive = "Dead")
cfC50

#

C50::plot(modelCARTC50)
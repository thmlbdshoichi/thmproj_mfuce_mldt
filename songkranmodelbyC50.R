setwd("~/R-Project/ProjectDT")
Sys.setenv('JAVA_HOME'="C:/Program Files/Java/jdk-15.0.2/") 
library(caret)
library(rJava)
library(RWeka)
library(ROCR)
library(partykit)
library(rattle)
library(rpart.plot)
require("devtools")
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

# Model Training "C50"
set.seed(4321)

modeltreeC50 = C5.0(Status ~ ., data = train, trials = 10)

# Prediction and Model Performance Testing
predictmodeltreeC50 <- predict(modeltreeC50, test)
overallmodeltreeC50 <- confusionMatrix(predictmodeltreeC50, test$Status)
overallmodeltreeC50

#Plot Tree C5.0
plot(modeltreeC50)
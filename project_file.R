###################
###################
#
# SYST 468 Project 
#
# 
#
#
#

# install packages
install.packages('reshape')
install.packages('ggplot2')
install.packages('mlbench')
install.packages('tidyverse')
install.packages("AppliedPredictiveModeling")
install.packages("caret")
install.packages("e1071")
install.packages('corrplot')
install.packages('caTools')
install.packages('MASS')
install.packages('pls')
install.packages('moments')
install.packages('elasticnet')
install.packages('forcats')

# libraries
library(reshape)
library(mlbench)
library(ggplot2)
library(tidyverse)
library(AppliedPredictiveModeling)
library(caret)
library(e1071)
library(corrplot)
library(caTools)
library(MASS)
library(pls)
library(moments)
library(elasticnet)
library(forcats)

#set working directory 
setwd("C:/Users/Murat/Desktop/project_468")
source("bin_cat_function.r")

#read dataset
train <- read.csv("train.csv")
test <- read.csv("test.csv")

#split into numerical and catergorical
train_num <- select_if(train, is.numeric)
train_cat <- select_if(train, negate(is.numeric))



sales_price <- data.frame(train_num[,38])
train_num <- train_num[,1:37]
names(sales_price)[1] = "sales_price"
train_cat <- as.tibble(train_cat)
#train_cat <- data.frame(cbind(train_cat, sales_price))

#remove null values
train_num[is.na(train_num)] <- 0
train_cat <- 
  mutate_if(train_cat, is.factor, fct_explicit_na, na_level = "BLANK")
train_cat <- data.frame(train_cat)
#train_cat <- data.frame(cbind(train_cat, sales_price))

#function for binary categorical predictor
bintrans_cat <- bin.cat(train_cat)


#############################################################


y <- aov(sales_price ~., train_cat)
z <- summary(y)
#zz <- data.frame(z[[1]]$`Pr(>F)`)

train_cat_int <- as.integer(train_cat)


#skewness reduction transformation
skewness(train_num)
train_pp <- preProcess(train_num, method= "BoxCox")
train_tran <- data.frame(predict(train_pp, train_num))
skewness(train_tran)


#additional preprocessing




#create training and test set
datPart <- createDataPartition(sales_price$sales_price, p=.6, list = FALSE)

# train/test split numerical
numTrain <- train_num[datPart,]
numTest <- train_num[-datPart,]

#train/test split categorical
catTrain <- train_cat[datPart,]
catTest <- train_cat[-datPart,]

#train/test sales price
saleTrain <- data.frame(sales_price[datPart])
saleTest <- data.frame(sales_price[-datPart])

train_set <- cbind(numTrain, saleTrain)
names(train_set)[38] = "sales_price"



#ordinary regression fit
lm_FitAll <- lm(sales_price ~., data = train_set)
summary(lm_FitAll)
xyplot(saleTrain$sales_price.datPart. ~ predict(lm_FitAll))

lm_all <- predict(lm_FitAll, numTest)
defaultSummary(data.frame(obs = saleTest$sales_price..datPart., pred =lm_all))
xyplot(saleTest$sales_price..datPart. ~ lm_all)



#plot predictor regressions in grid
featurePlot(train_tran, sales_price$sales_price, between = list(x = 1, y= 1), type = c("g", "p", "smooth"))
featurePlot(train_cat[,1:43], sales_price$sales_price, between = list(x = 1, y= 1), type = c("g", "p", "smooth"))


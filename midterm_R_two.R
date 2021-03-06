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
install.packages("dataPreparation")


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
library(readxl)
library(dataPreparation)
library(data.table)
library(reshape)




#read dataset

train <- read_csv("train.csv")
train_scaled <- read_excel("~/DOCUMENTS-20190130T164303Z-001/DOCUMENTS/College/College_Eigth_Semester/Syst 468/Project/train-scaled.xlsx")
#set working directory
setwd("C:/Users/wrayo/Documents/DOCUMENTS-20190130T164303Z-001/DOCUMENTS/College/College_Eigth_Semester/Syst 468/Project/Murat")
source("bin_cat_function.r")

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


#y <- aov(sales_price ~., train_cat)
#z <- summary(y)
#zz <- data.frame(z[[1]]$`Pr(>F)`)

#train_cat_int <- as.integer(train_cat)


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
#featurePlot(train_num, sales_price$sales_price, between = list(x = 1, y= 1), type = c("g", "p", "smooth"))
#featurePlot(train_cat[,1:43], sales_price$sales_price, between = list(x = 1, y= 1), type = c("g", "p", "smooth"


#********************************************************  Data Set 2

View(train_scaled)

#****** Remove Duplicates
filter_train_scaled = fastFilterVariables(train_scaled)
filter_train_scaled$MSZoning...4 = NULL
filter_train_scaled$Alley...10 = NULL
filter_train_scaled$LotShape...12 = NULL
filter_train_scaled$LandSlope...17 = NULL
filter_train_scaled$ExterQual...34 = NULL
filter_train_scaled$ExterCond...36 = NULL
filter_train_scaled$BsmtQual...39 = NULL
filter_train_scaled$BsmtCond...41  = NULL
filter_train_scaled$BsmtExposure...43 = NULL
filter_train_scaled$BsmtFinType1...45 = NULL
filter_train_scaled$BsmtFinType2...48 = NULL
filter_train_scaled$HeatingQC...54 = NULL
filter_train_scaled$CentralAir...56 = NULL
filter_train_scaled$KitchenQual...69 = NULL
filter_train_scaled$FireplaceQu...74 = NULL
filter_train_scaled$GarageFinish...78 = NULL
filter_train_scaled$GarageQual...82 = NULL
filter_train_scaled$GarageCond...84 = NULL
filter_train_scaled$PavedDrive...86 = NULL
filter_train_scaled$PoolQC...94= NULL

#***** R-Generated Scale (Remaining Values)
filter_train_scaled$LandContour =as.integer(as.factor(filter_train_scaled$LandContour))
filter_train_scaled$Utilities =as.integer(as.factor(filter_train_scaled$Utilities))
filter_train_scaled$LotConfig =as.integer(as.factor(filter_train_scaled$LotConfig))
filter_train_scaled$Neighborhood =as.integer(as.factor(filter_train_scaled$Neighborhood))
filter_train_scaled$Condition1 =as.integer(as.factor(filter_train_scaled$Condition1))
filter_train_scaled$Condition2 =as.integer(as.factor(filter_train_scaled$Condition2))
filter_train_scaled$BldgType =as.integer(as.factor(filter_train_scaled$BldgType))
filter_train_scaled$HouseStyle =as.integer(as.factor(filter_train_scaled$HouseStyle))
filter_train_scaled$RoofStyle =as.integer(as.factor(filter_train_scaled$RoofStyle))
filter_train_scaled$RoofMatl =as.integer(as.factor(filter_train_scaled$RoofMatl))
filter_train_scaled$Exterior1st =as.integer(as.factor(filter_train_scaled$Exterior1st))
filter_train_scaled$Exterior2nd =as.integer(as.factor(filter_train_scaled$Exterior2nd))
filter_train_scaled$MasVnrType =as.integer(as.factor(filter_train_scaled$MasVnrType))
filter_train_scaled$Foundation =as.integer(as.factor(filter_train_scaled$Foundation))
filter_train_scaled$Heating =as.integer(as.factor(filter_train_scaled$Heating))
filter_train_scaled$Electrical =as.integer(as.factor(filter_train_scaled$Electrical))
filter_train_scaled$Functional = as.integer(as.factor(filter_train_scaled$Functional))
filter_train_scaled$GarageType = as.integer(as.factor(filter_train_scaled$GarageType))
filter_train_scaled$GarageYrBlt =as.integer(as.factor(filter_train_scaled$GarageYrBlt))
filter_train_scaled$Fence = as.integer(as.factor(filter_train_scaled$Fence))
filter_train_scaled$MiscFeature= as.integer(as.factor(filter_train_scaled$MiscFeature))
filter_train_scaled$SaleType = as.integer(as.factor(filter_train_scaled$SaleType))
filter_train_scaled$SaleCondition = as.integer(as.factor(filter_train_scaled$SaleCondition))

#******************** Data Visualiz

                                                                                             
                                                                                             
                                                                                             
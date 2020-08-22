#SYST 468-001 Predicting Future House Prices
#Students: Murat Gokturk
#          O'Ryan Lattin
#
#





# packages and libraries --------------------------------------------------------

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
install.packages("dplyr")
install.packages('ellipse')
install.packages('ggrepel')
install.packages('nnet')
install.packages('neuralnet')
install.packages('randomForest')
install.packages('kernlab')
install.packages("xlsx")
install.packages("DMwR")

library(caret)
library(dataPreparation)
library(data.table)
library(AppliedPredictiveModeling)
library(reshape)
library(mlbench)
library(ggplot2)
library(tidyverse)
library(e1071)
library(corrplot)
library(caTools)
library(MASS)
library(pls)
library(moments)
library(elasticnet)
library(forcats)
library(readxl)
library(readr)
library(dplyr)
library(corrplot)
library(ellipse)
library(ggrepel)
library(elasticnet)
library(glmnet)
library(nnet)
library(neuralnet)
library(randomForest)
library(kernlab)
library(xlsx)
library(DMwR)

# directory set/sourcing ---------------------------------------------------
setwd("C:/Users/wrayo/Documents/DOCUMENTS-20190130T164303Z-001/DOCUMENTS/College/College_Eigth_Semester/Syst 468/Project/Murat")
source("bin_cat_function.r")
source("cat.transf.R")



#---------------------------- Data Visualization & Organization --------------------------

#read in datasets
trainX <- read.csv("train.csv")
test <- read.csv("test.csv")

#split into numerical and catergorical
train_num <- select_if(trainX, is.numeric)
train_cat <- select_if(trainX, negate(is.numeric))

test_num <- select_if(test, is.numeric)
test_cat <- select_if(test, negate(is.numeric))

sales_price <- data.frame(train_num[,38])
train_num <- train_num[,1:37]
names(sales_price)[1] = "sales_price"

train_cat <- cbind.data.frame(train_cat, train_num$MSSubClass)
train_num <- dplyr::select(train_num, -c(Id, MSSubClass))
names(train_cat)[44] = "MSSubClass"

test_cat <- cbind.data.frame(test_cat, test_num$MSSubClass)
test_num <- dplyr::select(test_num, -c(Id, MSSubClass))
names(test_cat)[44] = "MSSubClass"

train_cat <- train_cat %>% data.frame
train_cat[,44] <- train_cat[,44] %>% as.integer  %>%
  as.factor
train_cat <- as_tibble(train_cat)

train_year <-dplyr::select(train_num %>% as_tibble, YearBuilt, YearRemodAdd,
                           GarageYrBlt, YrSold) %>%
  data.frame
train_num <- dplyr::select(train_num %>% as_tibble, -c( YearBuilt, YearRemodAdd,
                                                        GarageYrBlt, YrSold) ) %>%
  data.frame

train_cat <- cbind(train_cat, train_year) 
s <-length(train_cat)
id <- 1:s
train_cat <- mutate_each(train_cat, funs(as.factor), id)


#test set repeat
test_cat <- test_cat %>% data.frame
test_cat[,44] <- test_cat[,44] %>% as.integer  %>%
  as.factor
test_cat <- as_tibble(test_cat)

test_year <-dplyr::select(test_num %>% as_tibble, YearBuilt, YearRemodAdd,
                           GarageYrBlt, YrSold) %>%
  data.frame
test_num <- dplyr::select(test_num %>% as_tibble, -c( YearBuilt, YearRemodAdd,
                                                        GarageYrBlt, YrSold) ) %>%
  data.frame

test_cat <- cbind(test_cat, test_year) 
s <-length(test_cat)
id <- 1:s
test_cat <- mutate_each(test_cat, funs(as.factor), id)


#remove null values

train_num[is.na(train_num)] <- 0

train_cat <- 
  mutate_if(train_cat, is.factor, fct_explicit_na, na_level = "BLANK")


test_num[is.na(test_num)] <- 0

test_cat <- 
  mutate_if(test_cat, is.factor, fct_explicit_na, na_level = "BLANK")


#-------------------- Categorical Preprocessing -------------------

#function for binary categorical predictor conversion
train_cat <- cbind(train_cat, sales_price$sales_price)
train_cat <- train_cat %>% data.frame
names(train_cat)[49] = "sales_price"

train_cat <- cat.trans(train_cat)

test_cat <- dplyr::select(test_cat, colnames(train_cat))

fact_cat <- train_cat 
f <- length(fact_cat)
id <- 1:f
fact_cat <- mutate_each(fact_cat, funs(as.factor), f)


bin_cat <- bin.cat(train_cat)
bin_catTest <- bin.cat(test_cat)


# ---------------------- Correlation ---------------------
# Identifying and removing correlated Predictors
num_corr <- cor(train_num)
corr_melt <- melt(num_corr, varnames = c("x", "y"))
corr_melt <- corr_melt[order(corr_melt$value),]
names(corr_melt)[3]= "correlation"
highcorr <- findCorrelation(num_corr, .8)
train_num = train_num[, -highcorr]

test_num <- dplyr::select(test_num, colnames(train_num)) 

ggplot(corr_melt, aes(x=x, y=y)) +
  geom_tile(aes(fill=correlation), color = 'black')+
  scale_fill_gradient2(low="red", mid="white",
                       high = "blue",
                       guide = guide_colorbar(ticks=FALSE,
                                              barheight = 10),
                       limits=c(-1,1)) +
  theme_minimal()+
  labs(x = NULL, 
       y = NULL,
       title = "Correlation Plot") +
  theme(axis.text.x = element_text(angle = 90))


#------------------------- BOXCOX Transformation --------------------
#Check Data Skewness
skewGrid <- skewness(train_num)
skewGrid
par(mfrow=c(1,2))
#Before BoxCox Transformation Data Visuals
boxplot(train_num$LotArea)
boxplot(train_num$TotRmsAbvGrd)

#BoxCox Transformation
train_pp <- preProcess(train_num, method= "BoxCox", center =TRUE, scale =TRUE)
train_tran <- data.frame(predict(train_pp, train_num))

test_pp <- preProcess(test_num, method= "BoxCox", center =TRUE, scale =TRUE)
test_tran <- data.frame(predict(test_pp, test_num))


par(mfrow=c(1,2)) 
#After BoxCox Transformation Data Visuals
boxplot(train_tran$LotArea)
boxplot(train_tran$TotRmsAbvGrd)
par(mfrow=c(1,1)) 
train_num <- train_tran
test_num <- test_tran

train_num <- cbind(train_num, sales_price) %>% data.frame
train_num <- scale(train_num, center = TRUE, 
                   scale = TRUE) %>% data.frame


#------------------------------------- PCA --------------------------------------
num_pca <- prcomp(train_num, center = TRUE,scale.=TRUE)

test_pca <- prcomp(test_num, center = TRUE,scale.=TRUE)

percentVariancePCA <- num_pca$sd^2/sum(num_pca$sd^2)*100
e <- length(train_num) - 1
ctrl <- trainControl(method = "cv", number = 10)


pcrTune <- train(x = train_num[,1:e], y = train_num$sales_price,
                 method ="pcr",
                 tuneGrid =expand.grid(ncomp =1:e),
                 trControl =ctrl)
pcrTune

pca_pred <- num_pca$x
pca_num <- data.frame(pca_pred[,1:25])
pca_rot <- num_pca$rotation %>% data.frame

testpca_pred <- test_pca$x
testpca_num <- data.frame(testpca_pred[,1:25])


#PCA cumulative variance plot
plot(percentVariancePCA, xlab="Components", 
     ylab="Percentage of Total Variance",
     type="l", main="Total variance vs. PCA components")

#PCA component tuning plot RMSE
plot(pcrTune, type = 'l',
     main="RMSE vs. PCA components")

#PCA scatter plot PC1 vs PC2
pca_plot <- ggplot(data = pca_rot) +
  geom_point(aes(x=PC1, y = PC2),
             size = 2, color = 'blue') +
  geom_hline(yintercept = 0)+
  geom_vline(xintercept = 0)+
  labs (
    x = 'Principle Component 1',
    y = 'Principle Component 2',
    title = 'Principle Component 1 vs. Principle Component 2'
  ) +
  xlim(-.5, .3) +
  ylim(-.5,.5) 

pca_plot

pca_plot2 <- pca_plot + 
  geom_segment(aes(x = 0, xend = PC1,
                   y = 0, yend = PC2))+
  geom_text_repel(aes(x = PC1, y = PC2,
                      label = rownames(pca_rot)),
                  point.padding = 0.2,
                  segment.color = 'grey50', 
                  size = 3, title = "Principle Component 1 vs. Principle Component 2")

pca_plot2


#-------------------------------Data Partioning --------------------------------------


#binary categorical predictors
dat_num<- train_num


#binary categorical predictors
dat_bin <- cbind(bin_cat,train_num)
dat_pca <- cbind(scale(pca_num, center = TRUE,
                       scale = TRUE), bin_cat, 
                 scale(sales_price, 
                       center = TRUE, 
                       scale = TRUE))


dat_fact <- cbind(fact_cat,train_num)


#final full kaggle training set for final model
datTest <- cbind(bin_catTest, test_num)
datTestPCA <- cbind(bin_catTest, testpca_num)


#create training and test set
datPart <- createDataPartition(dat_bin$sales_price, p=.80, list = FALSE)

# train/test split binary categories
numTrain <- dat_num[datPart,]
numTest <- dat_num[-datPart,]
binTrain <- dat_bin[datPart,]
binTest <- dat_bin[-datPart,]
PCATrain <- dat_pca[datPart,]
PCATest <- dat_pca[-datPart,]
factTrain <- dat_fact
testSet <- datTest
testSetPCA <- datTestPCA

n <- length(numTrain)-1
b <- length(binTrain)-1
p <- length(PCATrain)-1


# ------------------------Regression Models ----------------------------------------

#ordinary linear regression only numerical predictors
lm_num <- lm(sales_price ~., data = numTrain )
#head(lm_num)
lm_nump <- predict(lm_num, numTest[,1:n])

defaultSummary(data.frame(obs = numTest$sales_price, pred =lm_nump))
plot(numTest$sales_price ~ lm_nump, 
     xlab = "predicted",
     ylab = "observed",
     main = "Numerical Predictors Only")
# xlim = c(-5,5))
abline(0,1, col = "red")



#ordinary regression all predictors
lm_all <- lm(sales_price ~., data = binTrain )
summary(lm_all)

lm_allp <- predict(lm_all, binTest[,1:b])

defaultSummary(data.frame(obs = binTest$sales_price, pred =lm_allp))
plot(binTest$sales_price ~ lm_allp, 
     xlab = "predicted",
     ylab = "observed", 
     main = "Full Model")
# xlim = c(-5,5))
abline(0,1, col = "red")
 

#ordinary regression PCA
set.seed(456)
lm_pca <- lm(sales_price ~., data = PCATrain)
summary(lm_pca)
lm_pcap <- predict(lm_pca, PCATest[,1:p])

defaultSummary(data.frame(obs = PCATest$sales_price, pred =lm_pcap))
plot(PCATest$sales_price ~ lm_pcap, 
     xlab = "predicted",
     ylab = "observed", 
     main = "PCA Model")
abline(0,1, col = "red")

PCAVarIMp = as.data.frame(varImp(lm_pca))


#pls regression
set.seed(323)
plsTune <- train(binTrain[,1:b], binTrain$sales_price,
                 method = "pls",
                 tuneLength = b-25,
                 trControl = ctrl)

plsTune

#PCA component tuning plot RMSE
plot(plsTune, type = 'l',
     main="RMSE vs. PLS components")

plsFit <- plsr(sales_price ~., data = binTrain, 
               ncomp = 118, scale = FALSE)
summary(plsFit)

plsp <- predict(plsFit,binTest[,1:b], ncomp = 118)
plspValue = data.frame(obs= binTest$sales_price,pred = plsp)
plspresults = caret::postResample(binTest$sales_price, plsp)
plspresults
plot(binTest$sales_price ~ plsp, 
     xlab = "predicted",
     ylab = "observed",
     main= "PLS Model")
abline(0,1, col = "red")
plsVarIMp = as.data.frame(varImp(plsFit))

#Penalized Regression Models

#Elastic-Net Regression
#numerical (only)
set.seed(476)
elastic = enet(x =as.matrix(numTrain[,1:n]), y = numTrain$sales_price, lambda = 0.001) 
elasticPred = predict(elastic, newx = as.matrix(numTest[,1:n]), s = 1, mode = "fraction", type = "fit")
elasticValue = data.frame(obs =numTest$sales_price, pred = elasticPred$fit)
defaultSummary(elasticValue)
plot(x = elasticPred$fit, y = numTest$sales_price, xlab = "predicted", ylab = "observed", main = "Elastic Net Model", abline(0,1, col="red"))



#Ridge Regression
#numerical (only)
alpha0 = cv.glmnet(x = as.matrix.cast_df(numTrain[,1:n]), y= numTrain$sales_price, alpha= 0, family="gaussian", type.measure = "mse")
alpha0Pred = predict(alpha0, s=alpha0$lambda.1se, newx = as.matrix(numTest[,1:n]))
alpha0presults = caret::postResample(numTest$sales_price, alpha0Pred)
alpha0presults
plot(x = alpha0Pred, y = numTest$sales_price, xlab = "predicted", ylab = "observed",main = "Ridge Model", abline(0,1, col="red"))

#Combined Binary
binalpha0 = cv.glmnet(x =as.matrix.cast_df(binTrain[,1:b]), y =binTrain$sales_price, alpha=0, 
                      family ="gaussian", type.measure = "mse",
                      normalize = FALSE)
binalpha0Pred = predict(binalpha0, s=binalpha0$lambda.1se, newx = as.matrix(binTest[,1:b]))
binalpha0Pred = predict(binalpha0, s=binalpha0$lambda.1se, newx = as.matrix(binTest[,1:b]))
binalpha0results = caret::postResample(binTest$sales_price, binalpha0Pred)
binalpha0results

plot(x = binalpha0Pred, y = binTest$sales_price, xlab = "predicted", ylab = "observed", abline(0,1, col="red"), main = "Ridge Model")

#Lasso Regression
#numerical (only)
alpha1 = cv.glmnet(x = as.matrix.cast_df(numTrain[,1:n]), y= numTrain$sales_price, alpha= 1, family="gaussian", type.measure = "mse")
alpha1Pred = predict(alpha1, s=alpha1$lambda.1se, newx = as.matrix(numTest[,1:n]))
alpha1presults = caret::postResample(numTest$sales_price, alpha1Pred)
alpha1presults
plot(x = alpha1Pred, y = numTest$sales_price, xlab = "predicted", ylab = "observed", main = "Lasso Model", abline(0,1, col="red"))

#Combined Binary
binalpha1 = cv.glmnet(x = as.matrix.cast_df(binTrain[,1:b]), y= binTrain$sales_price, 
                      alpha= 1, family="gaussian", type.measure = "mse",
                      normalize = FALSE)
binalpha1Pred = predict(binalpha1, s=binalpha1$lambda.1se, newx = as.matrix(binTest[,1:b]))
binalpha1results = caret::postResample(binTest$sales_price, binalpha1Pred)
binalpha1results
plot(x = binalpha1Pred, y = binTest$sales_price, xlab = "predicted", ylab = "observed", abline(0,1, col="red"))


#table(binTest$sales_price, nnetPredict)


#random forest models
#rfGrid <- expand.grid(.mtry = c(1:(b/3)))

#rfTune <- train( x = binTrain[,1:b],
#                 y = binTrain$sales_price,
#                 method = "rf",
#                 tuneGrid = rfGrid,
#                 trControl = ctrl)



rfFit <- randomForest(binTrain[,1:b], binTrain$sales_price, importance = TRUE)
rfFit

rfPred <- predict(rfFit, newdata = binTest[,1:b])
rfPr <- postResample(pred = rfPred, obs = binTest$sales_price)
rfPr

plot(binTest$sales_price ~ rfPred, 
     xlab = "predicted",
     ylab = "observed",
     main= "Random Forest Model")
abline(0,1, col = "red")
#important predictors
varImpPlot(rfFit, type = 2)
RFvarRank = as.data.frame(importance(rfFit))



# Kaggle Models -----------------------------------------------------------
binKaggle <- dplyr::select(dat_bin, -c(starts_with('BLANK')))
testSetF <- dplyr::select(testSet, -c(starts_with('BLANK')))

binKaggleF <- intersect(colnames(binKaggle),
                        colnames(testSetF))


binKaggleT <- dplyr::select(dat_bin, c(binKaggleF))
testSetT <- dplyr::select(testSet, c(binKaggleF))

binKaggleT <- cbind(binKaggleT,sales_price)
bl <- length(binKaggleT)-1

binKaggle2 <- dplyr::select(bin_cat, -c(starts_with('BLANK')))
testSetF2 <- dplyr::select(bin_catTest, -c(starts_with('BLANK')))

binKaggleF2 <- intersect(colnames(binKaggle2),
                        colnames(testSetF2))

binT <- dplyr::select(testSetF2, c(binKaggleF2))
binK <- dplyr::select(binKaggle2,c(binKaggleF2 ))


testSetPCAF <- cbind(binT, testpca_num) %>% data.frame

trainSetPCAF <- cbind(binK, pca_num) %>% data.frame

trainSetPCAF <- cbind(trainSetPCAF, sales_price)
pl <- length(trainSetPCAF)-1



#LM
Kagglelm_all <- lm(sales_price ~., data = binKaggleT )
Kagglelm_allp <- predict(Kagglelm_all, testSetT)
write.csv(Kagglelm_allp, file = "Lm_Predict")


#PCA
Kagglelm_pca <- lm(trainSetPCAF$sales_price ~., data = trainSetPCAF[,1:pl])
Kagglelm_pcap <- predict(Kagglelm_pca, testSetPCAF)

write.csv(Kagglelm_pcap, file = "PCA_Predict")

#PLS
KagglePLS <- plsr(sales_price ~., data = binKaggleT, 
                            ncomp = 118, scale = FALSE)
KagglePLSp <- predict(KagglePLS, testSetT)
write.csv(KagglePLSp, file = "PLS_Predict")
#RIDGE
KaggleRidge = cv.glmnet(x =as.matrix.cast_df(binKaggleT[,1:bl]), y =binKaggleT$sales_price, alpha=0, 
                        family ="gaussian", type.measure = "mse",
                        normalize = FALSE)
KaggleRidgep = predict(KaggleRidge, s=KaggleRidge$lambda.1se, newx = as.matrix(testSetT))
write.csv(KaggleRidgep, file = "Ridge_Predict")


#RandomForest
rfFit <- randomForest(binKaggleT[,1:bl], 
                      binKaggleT$sales_price, importance = TRUE)
rfFit

rfPred <- predict(rfFit, newdata = testSetT)
write.csv(rfPred, file = "randomForest_Predict")
write


# -------------------------------------------Packages and Libraries --------------------------------------------------------

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
library("elasticnet")
library(glmnet)
library(plotrix)
library(Formula)
library(plotmo)
library(TeachingDemos)
library(earth)

#---------------------------------------- Directory Set/Sourcing ---------------------------------------------------

setwd("C:/Users/wrayo/Documents/DOCUMENTS-20190130T164303Z-001/DOCUMENTS/College/College_Eigth_Semester/Syst 468/Project/Used Files for R")
source("bin_cat_function.r")
source("cat.transf.r")



#---------------------------- Data Visualization & Organization --------------------------

#read in datasets
trainX <- read.csv("train.csv")
test <- read.csv("test.csv")

#split into numerical and catergorical
train_num <- select_if(trainX, is.numeric)
train_cat <- select_if(trainX, negate(is.numeric))

sales_price <- data.frame(train_num[,38])
train_num <- train_num[,1:37]
names(sales_price)[1] = "sales_price"

train_cat <- cbind.data.frame(train_cat, train_num$MSSubClass)
train_num <- dplyr::select(train_num, -c(Id, MSSubClass))
names(train_cat)[44] = "MSSubClass"

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


#remove null values

train_num[is.na(train_num)] <- 0

train_cat <- 
  mutate_if(train_cat, is.factor, fct_explicit_na, na_level = "BLANK")


#-------------------- Categorical Preprocessing -------------------

#function for binary categorical predictor conversion
train_cat <- cbind(train_cat, sales_price$sales_price)
train_cat <- train_cat %>% data.frame
names(train_cat)[49] = "sales_price"

train_cat <- cat.trans(train_cat)
bin_cat <- bin.cat(train_cat)


# ---------------------- Correlation ---------------------
# Identifying and removing correlated
# predictors
num_corr <- cor(train_num)
corr_melt <- melt(num_corr, varnames = c("x", "y"))
corr_melt <- corr_melt[order(corr_melt$value),]
names(corr_melt)[3]= "correlation"
highcorr <- findCorrelation(num_corr, .8)
train_num = train_num[, -highcorr]

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
par(mfrow=c(1,2)) 
#After BoxCox Transformation Data Visuals
boxplot(train_tran$LotArea)
boxplot(train_tran$TotRmsAbvGrd)
par(mfrow=c(1,1)) 
train_num <- train_tran
train_num <- cbind(train_num, sales_price) %>% data.frame
train_num <- scale(train_num, center = TRUE, 
                    scale = TRUE) %>% data.frame


#------------------------------------- PCA --------------------------------------
num_pca <- prcomp(train_num, center = TRUE,scale.=TRUE)
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


#create training and test set
datPart <- createDataPartition(dat_bin$sales_price, p=.85, list = FALSE)

# train/test split binary categories
numTrain <- dat_num[datPart,]
numTest <- dat_num[-datPart,]
binTrain <- dat_bin[datPart,]
binTest <- dat_bin[-datPart,]
PCATrain <- dat_pca[datPart,]
PCATest <- dat_pca[-datPart,]


# ------------------------Regression Models ----------------------------------------

#ordinary linear regression only numerical predictors
lm_num <- lm(sales_price ~., data = numTrain )
#head(lm_num)
o <- length(numTest) - 1
lm_nump <- predict(lm_num, numTest[,1:o])

defaultSummary(data.frame(obs = numTest$sales_price, pred =lm_nump))
plot(numTest$sales_price ~ lm_nump, 
     xlab = "predicted",
     ylab = "observed",
     main = "Numerical Predictors Only")
# xlim = c(-5,5))
abline(0,1, col = "red")



#ordinary regression all predictors
lm_all <- lm(sales_price ~., data = binTrain, )
summary(lm_all)

t <- length(binTrain)-1
lm_allp <- predict(lm_all, binTest[,1:t])

defaultSummary(data.frame(obs = binTest$sales_price, pred =lm_allp))
plot(binTest$sales_price ~ lm_allp, 
     xlab = "predicted",
     ylab = "observed", 
     main = "Full Model")
    # xlim = c(-5,5))
abline(0,1, col = "red")


#ordinary regression PCA
lm_pca <- lm(sales_price ~., data = PCATrain , )
summary(lm_pca)
z <- length(PCATest) -1
lm_pcap <- predict(lm_pca, PCATest[,1:z])

defaultSummary(data.frame(obs = PCATest$sales_price, pred =lm_pcap))
plot(PCATest$sales_price ~ lm_pcap, 
     xlab = "predicted",
     ylab = "observed", 
     main = "PCA Model")
abline(0,1, col = "red")



#pls regression
set.seed(123)
d <- length(numTrain) -1
plsTune <- train(numTrain[,1:d], numTrain$sales_price,
                 method = "pls",
                 tuneLength = d,
                 trControl = ctrl,
                 preProcess = c("center","scale"))

plsTune

#PCA component tuning plot RMSE
plot(plsTune, type = 'l',
     main="RMSE vs. PLS components")

plsFit <- plsr(sales_price ~., data = numTrain, 
               ncomp = 6)
summary(plsFit)

plsp <- predict(plsFit,numTest[,1:d], ncomp = 6)
plspValue = data.frame(obs= numTest$sales_price,pred = plsp)
plspresults = caret::postResample(numTest$sales_price, plsp)
plspresults
plot(numTest$sales_price ~ plsp, 
     xlab = "predicted",
     ylab = "observed",
     main= "PLS Model")
abline(0,1, col = "red")


#----------------------------------Penalized Regression Models-----------------------------------

#Elastic-Net Regression
#numerical (only)
set.seed(476)
elastic = enet(x =as.matrix(numTrain[,1:d]), y = numTrain$sales_price, lambda = 0.001) 
elasticPred = predict(elastic, newx = as.matrix(numTest[,1:d]), s = 1, mode = "fraction", type = "fit")
elasticValue = data.frame(obs =numTest$sales_price, pred = elasticPred$fit)
defaultSummary(elasticValue)
plot(x = elasticPred$fit, y = numTest$sales_price, xlab = "predicted", ylab = "observed", main = "Elastic Net Model", abline(0,1, col="red"))
  
#Combined 
#binelastic = enet(x =as.matrix(factTrain[,1:j]), y = as.matrix(binTrain$sales_price), lambda = 0.001) 
#binelasticPred = predict(binelastic, newx = as.matrix(factTrain[,1:j]), s = 1, mode = "fraction", type = "fit")
#binelasticValue = data.frame(obs =factTrain$sales_price, pred = binelasticPred$fit)
#defaultSummary(binelasticValue)
#plot(x = binelasticPred$fit, y = factTrain$sales_price, xlab = "predicted", ylab = "observed", abline(0,1, col="red"))


#Ridge Regression
#numerical (only)
alpha0 = cv.glmnet(x = as.matrix.cast_df(numTrain[,1:d]), y= numTrain$sales_price, alpha= 0, family="gaussian", type.measure = "mse")
alpha0Pred = predict(alpha0, s=alpha0$lambda.1se, newx = as.matrix(numTest[,1:d]))
alpha0presults = caret::postResample(numTest$sales_price, alpha0Pred)
alpha0presults
plot(x = alpha0Pred, y = numTest$sales_price, xlab = "predicted", ylab = "observed",main = "Ridge Model", abline(0,1, col="red"))
  
#Combined Binary

#binalpha0 = cv.glmnet(x =as.matrix.cast_df(binTrain[,1:315]), y =binTrain$sales_price, alpha=0, family ="gaussian", type.measure = "mse")
#binalpha0Pred = predict(alpha0, s=alpha0$lambda.1se, newx = as.matrix(binTest[,315]))
#mean((binTest$sales_price - alpha0.predicted)^2)
#plot(x = alpha0Pred, y = binTest$sales_price, xlab = "predicted", ylab = "observed", abline(0,1, col="red"))
  
#Lasso Regression
#numerical (only)
a <- length(numTrain) - 1
alpha1 = cv.glmnet(x = as.matrix.cast_df(numTrain[,1:a]), y= numTrain$sales_price, alpha= 1, family="gaussian", type.measure = "mse")
alpha1Pred = predict(alpha1, s=alpha1$lambda.1se, newx = as.matrix(numTest[,1:a]))
alpha1presults = caret::postResample(numTest$sales_price, alpha1Pred)
alpha1presults
plot(x = alpha1Pred, y = numTest$sales_price, xlab = "predicted", ylab = "observed", main = "Lasso Model", abline(0,1, col="red"))
  
#Combined Binary
#binalpha1 = cv.glmnet(x = as.matrix.cast_df(binTrain[,1:315]), y= binTrain$sales_price, alpha= 1, family="gaussian", type.measure = "mse")
#binalpha1Pred = predict(alpha1, s=alpha1$lambda.1se, newx = as.matrix(binTest[,1:315]))
#binmean((binTest$sales_price - alpha1.predicted)^2)
#binplot(x = alpha1Pred, y = binTest$sales_price, xlab = "predicted", ylab = "observed", abline(0,1, col="red"))
  
#KNN Regression
#Numerical Only
knnModel_num = train(x = numTrain[,1:32], y = numTrain$sales_price, method = "knn", preProcess = c("center","scale"), tuneLength = 10)
knnModel
knnPredict = predict(knnModel, newdata = numTest[,1:32])
knnValue = data.frame(obs = numTest$sales_price, pred = knnPredict)
defaultSummary(knnValue)
plot(x = knnPredict, y = numTest$sales_price, xlab = "predicted", ylab = "observed", main = "KNN Model", abline(0,1, col="red"))


#NN Regression
#Numerical Only
set.seed(321)
nnGrid = expand.grid(.decay = c(0,0.01, 0.1), .size = 1:10)
nnetModel = train(x= numTrain[,1:32], y =numTrain$sales_price, method = "nnet", preProcess = c("center","scale"), linout = TRUE, MaxNWts = (10 * (ncol(numTrain[,1:32]+1) +10 +1)), maxit = 500, tuneGrid = nnGrid)
nnetPredict = predict(nnetModel, newdata = numTest[,1:32])
nnResults  =  caret::postResample(numTest$sales_price, nnetPredict)
nnResults
plot(x = nnetPredict, y = numTest$sales_price, xlab = "predicted", ylab = "observed", main = "NN Model", abline(0,1, col="red"))
#MARS Regression
#Numerical Only
marsGrid = expand.grid(.degree = 1:2, .nprune =2:38)
varImp(marsModel)
marsPredict = predict(marsModel, newdata = numTest[,1:32])
marsresults = caret::postResample(numTest$sales_price, marsPredict)
marsresults
plot(x = marsPredict, y = numTest$sales_price, xlab = "predicted", ylab = "observed", main = "MARS Model", abline(0,1, col="red"))

#Svm
#Numerical Only
set.seed(444)
svRModel = train(x=numTrain[,1:32], y = numTrain$sales_price, method = "svmRadial", preProcess = c("center","scale"), tuneLength = 20)
svRPredict = predict(svRModel, newdata = numTest[,1:32])
svrValue = data.frame(obs =numTest$sales_price, pred = svRPredict)
defaultSummary(svrValue)
plot(x = svRPredict, y = numTest$sales_price, xlab = "predicted", ylab = "observed", main = "SvR Model", abline(0,1, col="red"))

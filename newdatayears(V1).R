
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


# directory set/sourcing ---------------------------------------------------

setwd("C:/Users/wrayo/Documents/DOCUMENTS-20190130T164303Z-001/DOCUMENTS/College/College_Eigth_Semester/Syst 468/Project/Used Files for R")
source("bin_cat_function.r")



# Data Organizing ---------------------------------------------------------

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
train_num <- dplyr::select(train_num %>% as_tibble, -c(Id, MSSubClass))
names(train_cat)[44] = "MSSubClass"

train_cat <- train_cat %>% data.frame
train_cat[,44] <- train_cat[,44] %>% as.integer  %>%
  as.factor
train_cat <- as_tibble(train_cat)


#remove null values

train_num[is.na(train_num)] <- 0

train_cat <- 
  mutate_if(train_cat, is.factor, fct_explicit_na, na_level = "BLANK")



# Categorical Preprocessing -----------------------------------------------

#function for binary categorical predictor conversion
train_cat <- train_cat %>% data.frame
bin_cat <- bin.cat(train_cat)



# Numerical Preprocessing -------------------------------------------------

#centering and scaling numeric data
train_numS <- scale(train_num, center = TRUE, scale = TRUE) %>%
  data.frame
sales_priceS <- scale(sales_price, center = TRUE, scale = TRUE) %>%
  data.frame


# Identifying and removing correlated
# predictors
num_corr <- cor(train_numS)
corr_melt <- melt(num_corr, varnames = c("x", "y"))
corr_melt <- corr_melt[order(corr_melt$value),]
names(corr_melt)[3]= "correlation"
highcorr <- findCorrelation(num_corr, .8)
train_numS = train_numS[, -highcorr]

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


# BOXCOX transformation for removing skewness
skewGrid <- skewness(train_numS)
skewGrid
par(mfrow=c(1,3))
#Before BoxCox Transformation Data Visuals
boxplot(train_numS$LotArea)
boxplot(train_numS$YearRemodAdd)
boxplot(train_numS$TotRmsAbvGrd)

train_pp <- preProcess(train_numS, method= "BoxCox", center =TRUE, scale =TRUE)
train_tran <- data.frame(predict(train_pp, train_numS))
boxplot(train_tran$LotArea)
boxplot(train_tran$YearRemodAdd)
boxplot(train_tran$TotRmsAbvGrd)
par(mfrow=c(1,1)) 
train_numS <- train_tran
train_numS <- scale(train_numS, center = TRUE, 
                    scale = TRUE) %>% data.frame

# PCA analysis
num_pca <- prcomp(train_numS, center = TRUE,scale.=TRUE)
percentVariancePCA <- num_pca$sd^2/sum(num_pca$sd^2)*100

ctrl <- trainControl(method = "cv", number = 10)
pcrTune <- train(x = train_numS, y = sales_priceS$sales_price,
                 method ="pcr",
                 tuneGrid =expand.grid(ncomp =1:30),
                 trControl =ctrl)

pca_pred <- num_pca$x
pca_num <- data.frame(pca_pred[,1:28])
pca_rot <- num_pca$rotation %>% data.frame



# Visualization -----------------------------------------------------------

#boxcox transformation
bc_plot <- scale(train_num[,8:11], center = TRUE,
                 scale = TRUE) %>% data.frame

bc_melt <- melt(bc_plot) %>% data.frame

ggplot(bc_melt) +
  geom_histogram(aes(x = value, y = ..density..),
                 color = 'black', fill = 'cyan', 
                 bins = 40) +
  facet_grid(vars(variable), 
             scales = 'free' ) + 
  geom_density(aes(x = value))+
  geom_vline(aes(xintercept = 0), 
             color = 'red', linetype = 'dashed')+
  xlim(-3,6)+
  labs(x = 'predictor value (scales)',
       y = 'density',
       title = 'Numerical Predictors before BoxCox')


bc_plot2 <- train_numS[,8:11]%>% data.frame
bc_melt2 <- melt(bc_plot2) %>% data.frame


ggplot(bc_melt2) +
  geom_histogram(aes(x = value, y = ..density..),
                 color = 'black', fill = 'cyan', 
                 bins = 40) +
  facet_grid(vars(variable), 
             scales = 'free' ) + 
  geom_density(aes(x = value))+
  geom_vline(aes(xintercept = 0), 
             color = 'red', linetype = 'dashed')+
  xlim(-3,6)+
  labs(x = 'predictor value (scales)',
       y = 'density',
       title = 'Numerical Predictors after BoxCox')


#correlation plot 
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

highcorr <- findCorrelation(num_corr, .8)
train_numS = train_num[, -highcorr]

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
                  size = 3)

pca_plot2


# data partitioning -------------------------------------------------------

#binary categorical predictors
dat_num<- cbind(scale(train_numS), 
                 scale(sales_price, 
                       center = TRUE, 
                       scale = TRUE)) %>% data.frame


#binary categorical predictors
dat_bin <- cbind(scale(train_numS), bin_cat, 
                 scale(sales_price, 
                       center = TRUE, 
                       scale = TRUE))
dat_pca <- cbind(scale(pca_num, center = TRUE,
                       scale = TRUE), bin_cat, 
                     scale(sales_price, 
                           center = TRUE, 
                           scale = TRUE))

#create training and test set
datPart <- createDataPartition(dat_bin$sales_price, p=.7, list = FALSE)

# train/test split binary categories
numTrain <- dat_num[datPart,]
numTest <- dat_num[-datPart,]
binTrain <- dat_bin[datPart,]
binTest <- dat_bin[-datPart,]
PCATrain <- dat_pca[datPart,]
PCATest <- dat_pca[-datPart,]


# regression models -------------------------------------------------------


#ordinary linear regression only numerical predictors
lm_num <- lm(sales_price ~., data = numTrain )
summary(lm_num)

lm_nump <- predict(lm_num, numTest)

defaultSummary(data.frame(obs = binTest$sales_price, pred =lm_nump))
plot(numTest$sales_price ~ lm_nump, 
     xlab = "predicted",
     ylab = "observed")
# xlim = c(-5,5))
abline(0,1, col = "red")



#ordinary regression all predictors
lm_all <- lm(sales_price ~., data = binTrain, )
summary(lm_all)

lm_allp <- predict(lm_all, binTest[,1:316])

defaultSummary(data.frame(obs = binTest$sales_price, pred =lm_allp))
plot(binTest$sales_price ~ lm_allp, 
     xlab = "predicted",
     ylab = "observed")
    # xlim = c(-5,5))
abline(0,1, col = "red")


#ordinary regression PCA

lm_pca <- lm(sales_price ~., data = PCATrain , )
summary(lm_pca)

lm_pcap <- predict(lm_pca, PCATest[,1:311])

defaultSummary(data.frame(obs = PCATest$sales_price, pred =lm_pcap))
plot(PCATest$sales_price ~ lm_pcap, 
     xlab = "predicted",
     ylab = "observed")
abline(0,1, col = "red")



#pls regression
set.seed(123)
plsTune <- train(numTrain[,1:32], numTrain$sales_price,
                 method = "pls",
                 tuneLength = 30,
                 trControl = ctrl,
                 preProcess = c("center","scale"))

plsTune

#PCA component tuning plot RMSE
plot(plsTune, type = 'l',
     main="RMSE vs. PLS components")

plsFit <- plsr(sales_price ~., data = numTrain, 
               ncomp = 7)
summary(plsFit)

plsp <- predict(plsFit,numTest[,1:32], ncomp = 7)

defaultSummary(data.frame(obs = numTest$sales_price, pred =plsp))
plot(numTest$sales_price ~ plsp, 
     xlab = "predicted",
     ylab = "observed")
abline(0,1, col = "red")















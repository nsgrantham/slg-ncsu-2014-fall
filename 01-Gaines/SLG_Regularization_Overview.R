#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
### NCSU Statistical Learning Group - Regularized Regression Overview ###
###                NCAA Data Example - Brian R. Gaines                ###
###       Data from Mangold, Bean, Adams (2003) via Dennis Boos       ###
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#### Initial Stuff ####
# clean up
rm(list=ls())
# load required packages
library(glmnet)
library(arm)
library(knitr) # needed for the kable function
# pick random seed
seed = 2014



#### Organize Data ####
# load dataset
ncaaData <- 
  read.table("http://www4.stat.ncsu.edu/~boos/var.select/ncaa.data2.txt",
             header=TRUE, quote="\"")

# define response and predictors
y = ncaaData$y
# X = scale(ncaaData[,-20])  # not needed, glmnet auto standardizes predictors
X = as.matrix(ncaaData[,-20])  # X needs to be a matrix, not a data frame

# randomly split data into training (2/3) and test (1/3) sets
set.seed(seed)
train = sample(1 : nrow(X), round((1/2) * nrow(X)))
test = -train

# subset training data
yTrain = y[train]
XTrain = X[train, ]
XTrainOLS = cbind(rep(1,nrow(XTrain)), XTrain)

# subset test data
yTest = y[test]
XTest = X[test, ]



#### Model Estimation ####
# estimate models
fitOLS = lm(yTrain ~ XTrain)  # Ordinary Least Squares
# glmnet automatically standardizes the predictors
fitRidge = glmnet(XTrain, yTrain, alpha = 0)  # Ridge Regression
fitLasso = glmnet(XTrain, yTrain, alpha = 1)  # The Lasso

### Plot Solution Paths ###
# Lasso
plot(fitLasso,xvar="lambda", label="TRUE")
# add label to upper x-axis
mtext("# of Nonzero (Active) Coefficients", side=3, line=2.5)

# Ridge
plot(fitRidge,xvar="lambda", label="TRUE")
# add label to upper x-axis
mtext("# of Nonzero (Active) Coefficients", side=3, line=2.5)



#### Model Selection ####
### 10-fold cross validation ###
# Lasso
set.seed(seed)  # set seed 
# (10-fold) cross validation for the Lasso
cvLasso = cv.glmnet(XTrain, yTrain, alpha = 1)
plot(cvLasso)
mtext("# of Nonzero (Active) Coefficients", side=3, line=2.5)

# Ridge Regression
set.seed(seed)  # set seed 
# (10-fold) cross validation for Ridge Regression
cvRidge = cv.glmnet(XTrain, yTrain, alpha = 0)
plot(cvRidge)
mtext("# of Nonzero (Active) Coefficients", side=3, line=2.5)


### Extract Coefficients ###
# OLS coefficient estimates
betaHatOLS = fitOLS$coefficients
# Lasso coefficient estimates 
betaHatLasso = as.double(coef(fitLasso, s = cvLasso$lambda.1se))  # s is lambda
# Ridge  coefficient estimates 
betaHatRidge = as.double(coef(fitRidge, s = cvRidge$lambda.1se))


### Test Set MSE ###
# calculate predicted values

# For some reason, the predict function was not working correctly for OLS
# One suggestion was to include the test response (yTrain) in the test set...
# data.frame with the same name as the original response, but this didn't work
# predOLS =  predict(fitOLS, 
#                      newdata = as.data.frame(cbind(XTest, yTrain = yTest)))

XTestOLS = cbind(rep(1, nrow(XTest)), XTest) # add intercept to test data
predOLS = XTestOLS%*%betaHatOLS 
predLasso = predict(fitLasso, s = cvLasso$lambda.1se, newx = XTest)
predRidge = predict(fitRidge, s = cvRidge$lambda.1se, newx = XTest)

# calculate test set MSE
testMSEOLS = mean((predOLS - yTest)^2)
testMSELasso = mean((predLasso - yTest)^2)
testMSERidge = mean((predRidge - yTest)^2)


### Plot Regression Coefficients ###
# create variable names for plotting 
varNames = c("top10", "act25", "oncampus", "ft_grad", "size", "tateach",
             "bbindex", "tuition", "board", "attend", "full_sal", "sf_ratio",
             "white", "ast_sal", "pop", "phd", "accept", "l_pct", "outstate",
             "intercept")

# Graph regression coefficients
coefplot(betaHatOLS[2:20], sd = rep(0,19), cex.pts = 1.5, 
         main = "Regression Coefficient Estimates", varnames = varNames)
coefplot(betaHatLasso[2:20], sd = rep(0,19), add = TRUE, col.pts = "red",
         cex.pts = 1.5)
coefplot(betaHatRidge[2:20], sd = rep(0,19), add = TRUE, col.pts = "blue",
         cex.pts = 1.5)
legend(2.7,18.5, c("OLS", "Lasso", "Ridge"), col = c("black", "red", "blue"), 
       pch = c(19, 19 ,19), bty = "n")

MSETable = data.frame(OLS=testMSEOLS, Lasso=testMSELasso, Ridge=testMSERidge)
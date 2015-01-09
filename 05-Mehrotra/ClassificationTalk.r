##############################################################################
#### Suchit Mehrotra 
#### Bayesian Classification Methods
##############################################################################

##############################################################################
######################## Bayesian Regression Example #########################
##############################################################################
rm(list = ls())

require(LearnBayes)
require(ISLR)
library(MASS)
require(klaR)
require(xtable)
require(arm)
setwd("C:/Users/Suchit/Desktop/Classification-Talk")

set.seed(94981)

# Using the college data from ISL
boston.fit <- lm(medv ~ crim + indus + chas + rm, data = Boston, y = TRUE, 
    x = TRUE)

# the blinreg function in the learn bayes package runs samples from the 
# posterior distribution of the regression parameters
theta.sample <- blinreg(boston.fit$y, boston.fit$x, 100000)

par(mfrow = c(2,2))
hist(theta.sample$beta[,2], main = "Crime", xlab = expression(beta[1]), 
    breaks = 100)
hist(theta.sample$beta[,3], main = "Industry", xlab = expression(beta[2]), 
     breaks = 100)
hist(theta.sample$beta[,4], main = "Charles River", xlab = expression(beta[3]), 
     breaks = 100)
hist(theta.sample$beta[,5], main = "Rooms", xlab = expression(beta[4]), 
     breaks = 100)
par(mfrow = c(1,1))
hist(theta.sample$sigma, main = "Std. Deviation", 
    xlab = expression(sigma), breaks = 100)


lm.ci <- confint(boston.fit)
bayes.ci <- apply(theta.sample$beta, 2, quantile, c(0.025, 0.975))
xtable(lm.ci)
xtable(bayes.ci)

x1 <- apply(boston.fit$x, 2, quantile, 0.25)
x2 <- apply(boston.fit$x, 2, quantile, 0.75)
X1 <- rbind(x1, x2)

mean.draws <- blinregexpected(X1, theta.sample)
pred.draws <- blinregpred(X1, theta.sample)


par(mfrow = c(1,2))
plot(density(mean.draws[,1]), main = "Mean Response at 25th Percentile", 
    xlab= "", col = "red", xlim = mean(pred.draws) + c(-3, 3) * sd(pred.draws))
lines(density(pred.draws[,1]), col = "blue")

###############################################################################
######################### Naive Bayes ########################################
###############################################################################

# Simple Example with Iris data set
pdf("iris.pdf")
pairs(iris[1:4],main="Iris Data", 
      pch=21, bg=c("red","green3","blue")[unclass(iris$Species)])
dev.off();
# the classes are spearable therefore leads to a high prediction accuracy

iris.nb <- NaiveBayes(Species ~ ., data = iris)
xtable(table(predict(iris.nb)$class, iris$Species))
mean((predict(iris.nb)$class == iris$Species) * 1)


par(mfrow = c(2,2))
plot(iris.nb, legend = FALSE, main = "")


##### Using the breast-cancer data set
cancer <- read.csv("breast-cancer-wisconsin.csv")
colnames(cancer) <- c("id", "thickness", "unif.size", "unif.shape", "adhesion", 
    "cell.size", "bare.nuclei", "bland.chromatin", "normal.nucleoli", "mitoses",
    "malignant")
cancer$malignant <- ifelse(cancer$malignant == 4, 1, 0)

# graph for logistic regression before using NaiveBayes
library(gplot) 
ggplot(cancer, aes(x = thickness, y = malignant)) + 
    stat_smooth(method = "bayesglm", family = "binomial", se = FALSE) + 
    geom_point(size = 3)

cancer$malignant <- as.factor(cancer$malignant)

# running NaiveBayes with non-parametric estimation
cancer.nb <- NaiveBayes(malignant ~. ,data = cancer, usekernel = TRUE)
tab <- table(predict(cancer.nb)$class, cancer$malignant)


cancer.nb1 <- NaiveBayes(malignant ~. , data = cancer)
tab1 <- table(predict(cancer.nb1)$class, cancer$malignant)


#assigning prior information to the Naive Bayes model
priors <- rbind(c(0.1, 0.9), c(0.25, 0.75), c(0.5, 0.5), c(0.9, 0.1))

#first prior
cancer.p1 <- NaiveBayes(malignant ~ . , usekernel = TRUE, prior = priors[1,], 
    data = cancer)
tab.p1 <- table(predict(cancer.p1)$class, cancer$malignant)
xtable(tab.p1)

#second prior
cancer.p2 <- NaiveBayes(malignant ~ . , usekernel = TRUE, prior = priors[2,], 
    data = cancer)
tab.p2 <- table(predict(cancer.p2)$class, cancer$malignant)
xtable(tab.p2)

#third prior
cancer.p3 <- NaiveBayes(malignant ~ . , usekernel = TRUE, prior = priors[3,], 
    data = cancer)
tab.p3 <- table(predict(cancer.p3)$class, cancer$malignant)
xtable(tab.p3)

#fourth prior
cancer.p4 <- NaiveBayes(malignant ~ . , usekernel = TRUE, prior = priors[4,], 
    data = cancer)
tab.p4 <- table(predict(cancer.p4)$class, cancer$malignant)
xtable(tab.p4)

###############################################################################
# sepsis data. data and code from AMS 205B taught by David Draper at UCSC
# http://classes.soe.ucsc.edu/ams205b/Winter14/

sepsis.anc.age.i2t <- matrix( scan( "sepsis-anc-age-i2t.txt" ),
                              66846, 4, byrow = T )

sepsis <- sepsis.anc.age.i2t[ , 1 ]

anc <- sepsis.anc.age.i2t[ , 2 ]

age <- sepsis.anc.age.i2t[ , 3 ]

i2t <- sepsis.anc.age.i2t[ , 4 ]

par( mfrow = c( 1, 3 ) )

plot( anc[ age <= 1 ], i2t[ age <= 1 ], type = 'n',
      xlab = 'ANC', ylab = 'I2T', main = 'Age <= 1' )

points( anc[ ( age <= 1 ) & ( sepsis == 0 ) ],
        i2t[ ( age <= 1 ) & ( sepsis == 0 ) ], col = 'black', pch = '.' )

points( anc[ ( age <= 1 ) & ( sepsis == 1 ) ],
        i2t[ ( age <= 1 ) & ( sepsis == 1 ) ], col = 'red', lwd = 2 )

plot( anc[ ( 1 < age ) & ( age <= 4 ) ], 
      i2t[ ( 1 < age ) & ( age <= 4 ) ], type = 'n',
      xlab = 'ANC', ylab = 'I2T', main = '1 < Age <= 4' )

points( anc[ ( 1 < age ) & ( age <= 4 ) & ( sepsis == 0 ) ],
        i2t[ ( 1 < age ) & ( age <= 4 ) & ( sepsis == 0 ) ], 
        col = 'black', pch = '.' )

points( anc[ ( 1 < age ) & ( age <= 4 ) & ( sepsis == 1 ) ],
        i2t[ ( 1 < age ) & ( age <= 4 ) & ( sepsis == 1 ) ], 
        col = 'red', lwd = 2 )

plot( anc[ age > 4 ], i2t[ age > 4 ], type = 'n',
      xlab = 'ANC', ylab = 'I2T', main = '4 < Age' )

points( anc[ ( age > 4 ) & ( sepsis == 0 ) ],
        i2t[ ( age > 4 ) & ( sepsis == 0 ) ], col = 'black', pch = '.' )

points( anc[ ( age > 4 ) & ( sepsis == 1 ) ],
        i2t[ ( age > 4 ) & ( sepsis == 1 ) ], col = 'red', lwd = 2 )

par( mfrow = c( 1, 1 ) )

url <- "http://classes.soe.ucsc.edu/ams205b/Winter13/sepsis-anc-age-i2t.txt"
sepsis.data <- read.table(url, col.names = c("sepsis", "anc", "age", "i2t"))
sepsis.data[,1] <- as.factor(sepsis.data[,1])

# trying out both parametric and non-parametric methods 
sepsis.nb1 <- NaiveBayes(sepsis ~ ., data = sepsis.data, usekernel = TRUE)
sepsis.tab1 <- table(predict(sepsis.nb1)$class, sepsis.data$sepsis)
sum(diag(sepsis.tab1))/sum(sepsis.tab1)

sepsis.nb2 <- NaiveBayes(sepsis ~ ., data = sepsis.data)
sepsis.tab2 <- table(predict(sepsis.nb2)$class, sepsis.data$sepsis)
sum(diag(sepsis.tab2))/sum(sepsis.tab2)


###############################################################################
############################### Logistic Regression ###########################
###############################################################################

# logistic regression on cancer data
cancer.glm <- glm(malignant ~ . , data = cancer, family = "binomial")
glm.probs <- predict(cancer.glm, type = "response")
glm.pred <- rep(0, 698)
glm.pred[glm.probs > .5] <- 1
cancer.tab1 <- table(glm.pred, cancer$malignant)

cancer.bayesglm <- bayesglm(malignant ~ . , data = cancer, family = binomial
    (link = "logit"))

xtable(tab)
xtable(tab1)

thick.0 <- cancer$thickness[cancer$malignant == 0]

plot(cancer.nb, legendplot = FALSE, main = "")
hist(thick.0)


# generating data for logistic regression plots 
# this is the example from Andrew Gelmans book Data Analysis Using Regression 
# and Multilevel/Hierarchical Models

x <- rnorm(60, mean =1, sd = 2)
y <- ifelse(x<2,0,1)
sep.df <- data.frame(x, y)

## Fit the model

fit.0 <- glm (y ~ x, family=binomial(link="logit"))

# plot with glm
ggplot(sep.df, aes(x = x, y = y)) + 
    stat_smooth(method = "glm", family = "binomial", se = FALSE) + 
    geom_point(size = 3)

# plot with bayes glm
ggplot(sep.df, aes(x = x, y = y)) + 
    stat_smooth(method = "bayesglm", family = "binomial", se = FALSE) + 
    geom_point(size = 3)

fit.bayes <- bayesglm(y ~ x, family = binomial(link = "logit"))

################################################################################
########################## Sepsis Bayes GLM ####################################
################################################################################

sepsis.bayes <- bayesglm(sepsis ~ . , data = sepsis.data, 
    family = binomial(link = "logit"))
sepsis.probs <- predict(sepsis.bayes, type = "response")

sepsis.pred <- rep(0, 66846)
sepsis.pred[sepsis.probs > 0.1] <- 1
sepsis.tab.1 <- table(sepsis.pred, sepsis.data$sepsis)
sum(diag(sepsis.tab.1))/sum(sepsis.tab.1)
xtable(sepsis.tab.1)

sepsis.pred2 <- rep(0, 66846)
sepsis.pred2[sepsis.probs > 0.05] <- 1
sepsis.tab.05 <- table(sepsis.pred2, sepsis.data$sepsis)
sum(diag(sepsis.tab.05))/sum(sepsis.tab.05)
xtable(sepsis.tab.05)

sepsis.pred3 <- rep(0, 66846)
sepsis.pred3[sepsis.probs > 0.025] <- 1
sepsis.tab.025 <- table(sepsis.pred3, sepsis.data$sepsis)
sum(diag(sepsis.tab.025))/sum(sepsis.tab.025)
xtable(sepsis.tab.025)

sepsis.pred4 <- rep(0, 66846)
sepsis.pred4[sepsis.probs > 0.01] <- 1
sepsis.tab.01 <- table(sepsis.pred4, sepsis.data$sepsis)
sum(diag(sepsis.tab.01))/sum(sepsis.tab.01)
xtable(sepsis.tab.01)













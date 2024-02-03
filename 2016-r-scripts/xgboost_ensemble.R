train <- read.csv("preds.csv", header=F)

require(caret)
require(corrplot)
require(Rtsne)
require(xgboost)
require(stats)
require(knitr)
require(ggplot2)

outcome.org = as.factor(train[, "V26"])
outcome = outcome.org 
levels(outcome)

# convert character levels to numeric
num.class = length(levels(outcome))
levels(outcome) = 1:num.class
head(outcome)

# remove outcome from train
train$V26 = NULL

# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

# convert data to matrix
train.matrix = as.matrix(train)
mode(train.matrix) = "numeric"

y = as.matrix(as.integer(outcome)-1)

# set random seed, for reproducibility 
set.seed(1234)
# k-fold cross validation, with timing
nround.cv = 200
system.time( bst.cv <- xgb.cv(param=param, data=train.matrix, label=y, 
                              nfold=4, nrounds=nround.cv, prediction=TRUE, verbose=FALSE) )
tail(bst.cv$dt) 

# index of minimum merror
min.merror.idx = which.min(bst.cv$dt[, test.merror.mean]) 
min.merror.idx 

# minimum merror
bst.cv$dt[min.merror.idx,]

# get CV's prediction decoding
pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
pred.cv = max.col(pred.cv, "last")
# confusion matrix
confusionMatrix(factor(y+1), factor(pred.cv))

library(irr)
ratings=data.frame(labels=factor(y+1), res=factor(pred.cv))
kappa2(ratings, weight ="squared")

system.time( bst <- xgboost(param=param, data=train.matrix, label=y, 
                            nrounds=min.merror.idx, verbose=0) )

# xgboost predict test data using the trained model
pred <- predict(bst, test.matrix)  
head(pred, 10)  

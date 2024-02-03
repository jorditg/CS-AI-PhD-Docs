data <- read.csv("dump.csv")
head(data)

data$Class <- as.factor(data$Class)
class_proportions <- summary(data$Class) / length(data$Class)

samples <- nrow(data)/2
train <- data[1:samples,]
test <- data[(samples+1):nrow(data),]
require(randomForest)
set.seed(17)

model.rf <- randomForest(Class~., data=train, mtry=15, importance=T, 
                         do.trace=100, ntree=500)

predict = predict(model.rf, type="response", newdata=test)

# Kappa with info only of own eye
library(irr)
ratings=data.frame(labels=test$Class, res=predict)
kappa2(ratings, weight ="squared")

require(xgboost)
require(data.matrix)
require(ggplot2)
tlabels <- as.numeric(train[,26])-1 # labels shoulb be in [0, num_class-1]
xgbtrain <- xgb.DMatrix(data=data.matrix(train[,-c(26)]), label=tlabels)
xgbtest <- xgb.DMatrix(data=data.matrix(test[,-c(26)]))
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = 5,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 6,   # number of threads to be used 
              "max_depth" = 15,    # maximum depth of tree 
              "eta" = 0.01,    # step size shrinkage 
              "subsample" = 0.8,    # part of data instances to grow tree 
              "colsample_bytree" = 0.9  # subsample ratio of columns when constructing each tree 
              
)

xgb_cv <- xgb.cv(params = params,
                  data = xgbtrain,
                  label = tlabels,
                  nrounds = 100, 
                  nfold = 5,            # number of folds in K-fold
                  prediction = TRUE,    # return the prediction using the final model 
                  showsd = TRUE,        # standard deviation of loss across folds
                  stratified = TRUE,    # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print.every.n = 1, 
                  early.stop.round = 10
)

# fit the model with the arbitrary parameters specified above
xgb <- xgboost(data = xgbtrain,
                label = tlabels,
                params = params,
                nrounds = 100,       # max number of trees to build
                verbose = TRUE,                                         
                print.every.n = 1,
                early.stop.round = 10 # stop if no improvement within 10 trees
)

predict = predict(xgb, newdata=xgbtest)
predict <- matrix(predict, nrow = nrow(test), byrow = TRUE) # reshape
predict <- max.col(predict)
# Kappa with info only of own eye
library(irr)
ratings=data.frame(labels=test$Class, res=predict)
kappa2(ratings, weight ="squared")

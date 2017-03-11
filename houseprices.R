

#alternative way of loading and installing libraries quickely ----
load.libraries <- c("Hmisc", "Amelia", "mice", "lattice", "missForest", 
                    "mlr", "dplyr", "VIM", "Boruta", "xgboost", "irace",
                    "splitstackshape", "gbm", "ggvis", "RWeka", "brnn",
                    "bartMachine", "tgp", "kohonen", "tgp", "bst", "party", "crs",
                    "Cubist", "glmnet", "earth", "elmNN", "evtree", "extraTrees",
                    "FNN", "mlr", "frbs", "mboost", "kernlab", "gbm", "stats",
                    "mboost", "glmnet", "GPfit", "h2o", "DiceKriging", "kernlab",
                    "laGP", "LiblineaR", "mda", "nnet", "nodeHarvest", "pls",
                    "penalized", "randomForest", "ranger", "rknn", "rpart",
                    "RRF", "rsm", "e1071", "flare", "randomForestSRC")

install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs)
sapply(load.libraries, require, character = TRUE)


#load used functions ----
antilog<-function(lx,base) 
{ 
  lbx<-lx/log(exp(1),base=base) 
  result<-exp(lbx) 
  result 
} 

#read files ----
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)

describe(train)

train$Alley[is.na(train$Alley)] <- "None"
train$BsmtQual[is.na(train$BsmtQual)] <- "None"
train$BsmtCond[is.na(train$BsmtCond)] <- "None"
train$BsmtExposure[is.na(train$BsmtExposure)] <- "None"
train$BsmtFinType1[is.na(train$BsmtFinType1)] <- "None"
train$BsmtFinType2[is.na(train$BsmtFinType2)] <- "None"
train$FireplaceQu[is.na(train$FireplaceQu)] <- "None"
train$GarageType[is.na(train$GarageType)] <- "None"
train$GarageYrBlt[is.na(train$GarageYrBlt)] <- "None"
train$GarageFinish[is.na(train$GarageFinish)] <- "None"
train$GarageQual[is.na(train$GarageQual)] <- "None"
train$GarageCond[is.na(train$GarageCond)] <- "None"
train$PoolQC[is.na(train$PoolQC)] <- "None"
train$Fence[is.na(train$Fence)] <- "None"
train$MiscFeature[is.na(train$MiscFeature)] <- "None"


test$Alley[is.na(test$Alley)] <- "None"
test$BsmtQual[is.na(test$BsmtQual)] <- "None"
test$BsmtCond[is.na(test$BsmtCond)] <- "None"
test$BsmtExposure[is.na(test$BsmtExposure)] <- "None"
test$BsmtFinType1[is.na(test$BsmtFinType1)] <- "None"
test$BsmtFinType2[is.na(test$BsmtFinType2)] <- "None"
test$FireplaceQu[is.na(test$FireplaceQu)] <- "None"
test$GarageType[is.na(test$GarageType)] <- "None"
test$GarageYrBlt[is.na(test$GarageYrBlt)] <- "None"
test$GarageFinish[is.na(test$GarageFinish)] <- "None"
test$GarageQual[is.na(test$GarageQual)] <- "None"
test$GarageCond[is.na(test$GarageCond)] <- "None"
test$PoolQC[is.na(test$PoolQC)] <- "None"
test$Fence[is.na(test$Fence)] <- "None"
test$MiscFeature[is.na(test$MiscFeature)] <- "None"

hist(train$SalePrice)

# train$Id <- NULL
# test$Id <- NULL


all <- rbind(train[,-81], test)
# all$Id <- NULL

md.pattern(all)
aggr_plot <- aggr(all,
                  col=c('navyblue','red'),
                  numbers=TRUE,
                  sortVars=TRUE,
                  labels=names(data),
                  cex.axis=.7,
                  gap=3,
                  ylab=c("Histogram of missing data","Pattern"))

tempData <- mice(all, m=5, maxit=50, meth='pmm', seed=5)
summary(tempData)


xyplot(tempData,Ozone ~ Wind+Temp+Solar.R,pch=18,cex=1)
densityplot(tempData)
stripplot(tempData, pch = 20, cex = 1.2)

completedData <- complete(tempData,1)

train.response <- as.data.frame(train[,c(1,81)])

completed.train <- inner_join(completedData, train.response, by = "Id")
completed.test <- completedData %>% dplyr::filter(!Id %in% completed.train$Id)

indx <- sapply(completed.train, is.character)
completed.train[indx] <- lapply(completed.train[indx], function(x) as.factor(x))

indx <- sapply(completed.test, is.character)
completed.test[indx] <- lapply(completed.test[indx], function(x) as.factor(x))

completed.train$SalePrice <- log(completed.train$SalePrice)
hist(completed.train$SalePrice)

#train and test for first submission ----
regr.task <- makeRegrTask(id = "train1",
                          data = as.data.frame(completed.train[, !colnames(completed.train) %in% c("Id")]),
                          target = "SalePrice")
regr.lrn <- makeLearner("regr.gbm",
                        par.vals = list(n.trees = 2000, interaction.depth = 5))
mod <- train(regr.lrn, regr.task)
pred <- predict(mod,  newdata = as.data.frame(completed.test[, !colnames(completed.test) %in% c("Id")]))

pred.antilog <- antilog(pred$data$response)

sample_submission <- list()
sample_submission$Id <- completed.test$Id
sample_submission$SalePrice <- pred.antilog
sample_submission <- data.frame(sample_submission)

write.csv(sample_submission,
          paste(getwd(), "sample_submission_1.csv", sep = "/"),
          row.names = FALSE)


#feature importance run ----
set.seed(5)
data.boruta <- completed.train[complete.cases(completed.train), ]
boruta.train <- Boruta(SalePrice ~ .,
                       data = as.data.frame(data.boruta[, !colnames(data.boruta) %in% c("Id")]),
                       doTrace = 2)
mar.default = c(14,1,1,1)
par(mar = mar.default + c(0, 4, 0, 0))
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels), at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)


# train important features only ----
completed.train.temp <- completed.train %>% dplyr::select(-PoolArea,
                                                          -PoolQC,
                                                          -LotConfig,
                                                          -MoSold,
                                                          -X3SsnPorch,
                                                          -MiscVal,
                                                          -LowQualFinSF,
                                                          -YrSold,
                                                          -MiscFeature,
                                                          -LandSlope,
                                                          -Street,
                                                          -RoofMatl,
                                                          -Heating,
                                                          -BsmtFinSF2,
                                                          -SaleType,
                                                          -BsmtCond,
                                                          -BsmtHalfBath,
                                                          -ExterCond,
                                                          -Fence,
                                                          -EnclosedPorch,
                                                          -Alley,
                                                          -SaleCondition,
                                                          -Utilities)

completed.test.temp <- completed.test %>% dplyr::select(-PoolArea,
                                                        -PoolQC,
                                                        -LotConfig,
                                                        -MoSold,
                                                        -X3SsnPorch,
                                                        -MiscVal,
                                                        -LowQualFinSF,
                                                        -YrSold,
                                                        -MiscFeature,
                                                        -LandSlope,
                                                        -Street,
                                                        -RoofMatl,
                                                        -Heating,
                                                        -BsmtFinSF2,
                                                        -SaleType,
                                                        -BsmtCond,
                                                        -BsmtHalfBath,
                                                        -ExterCond,
                                                        -Fence,
                                                        -EnclosedPorch,
                                                        -Alley,
                                                        -SaleCondition,
                                                        -Utilities)
#train and test for second submission ----
regr.task <- makeRegrTask(id = "train1",
                          data = as.data.frame(completed.train.temp[, !colnames(completed.train.temp) %in% c("Id")]),
                          target = "SalePrice")

regr.lrn <- makeLearner("regr.gbm",
                        par.vals = list(n.trees = 2000, interaction.depth = 5))

mod <- train(regr.lrn, regr.task)
pred <- predict(mod,  newdata = as.data.frame(completed.test.temp[, !colnames(completed.test.temp) %in% c("Id")]))

pred.antilog <- antilog(pred$data$response)

sample_submission <- list()
sample_submission$Id <- completed.test.temp$Id
sample_submission$SalePrice <- pred.antilog
sample_submission <- data.frame(sample_submission)

write.csv(sample_submission,
          paste(getwd(), "sample_submission_2.csv", sep = "/"),
          row.names = FALSE)


#train and test for third submission ----

completed.train.temp_2 <- completed.train[complete.cases(completed.train), ]
completed.test.temp_2 <- completed.test[complete.cases(completed.test), ]

control <- trainControl(method="repeatedcv", number=10, repeats=3)
set.seed(5)
model <- caret::train(SalePrice ~.,
                      data = as.data.frame(completed.train.temp_2[, !colnames(completed.train.temp_2) %in% c("Id")]),
                      method = "brnn",
                      metric="RMSE",
                      tuneGrid = NULL,
                      trControl = control)

predict <- predict(model,
                   as.data.frame(completed.test.temp_2[, !colnames(completed.test.temp_2) %in% c("Id")]))
pred.antilog <- antilog(predict)

sample_submission <- list()
sample_submission$Id <- completed.test$Id
sample_submission$SalePrice <- pred.antilog
sample_submission <- data.frame(sample_submission)

write.csv(sample_submission,
          paste(getwd(), "sample_submission_3.csv", sep = "/"),
          row.names = FALSE)


#split train set to train and test ----
set.seed(5)
response <- completed.train[, "SalePrice"]
train_set_new <- createDataPartition(response, p = 0.4,list = FALSE)

sample_test <- completed.train[train_set_new, ]
sample_train <- completed.train[-train_set_new, ]

sample_test <- sample_test[complete.cases(sample_test),]
sample_train <- sample_train[complete.cases(sample_train),]

sample_test <- sample_test %>% dplyr::select(-Condition1,
                                             -Condition2,
                                             -RoofStyle,
                                             -RoofMatl,
                                             -Heating,
                                             -HeatingQC,
                                             -GarageYrBlt,
                                             -GarageQual,
                                             -GarageCond)


sample_train <- sample_train %>% dplyr::select(-Condition1,
                                             -Condition2,
                                             -RoofStyle,
                                             -RoofMatl,
                                             -Heating,
                                             -HeatingQC,
                                             -GarageYrBlt,
                                             -GarageQual,
                                             -GarageCond)


regr.task <- makeRegrTask(id = "train2",
                          data = as.data.frame(sample_train[, !colnames(sample_train) %in% c("Id")]),
                          target = "SalePrice")

regr.lrn <- makeLearner("regr.gbm",
                        par.vals = list(n.trees = 2000, interaction.depth = 5))

regr.lrn <- makeLearner("regr.randomForestSRC")

mod <- train(regr.lrn, regr.task)
pred <- predict(mod,
                newdata = data.frame(sample_test[, !colnames(sample_test) %in% c("Id", "SalePrice")], stringsAsFactors = TRUE))

pred.antilog <- antilog(pred$data$response)

actual_preds <- data.frame(cbind(actual = antilog(sample_test$SalePrice), predicted = pred.antilog))
MAPE <- mean(abs((actual_preds$predicted - actual_preds$actual)) / actual_preds$actual)
100 * (1 - MAPE)

#Tuning across whole model spaces with ModelMultiplexer ----

base.learners = list(
  # makeLearner("regr.bartMachine"),
  # makeLearner("regr.bcart"),
  # makeLearner("regr.bdk"),
  # makeLearner("regr.bgp"),
  # makeLearner("regr.bgpllm"),
  # makeLearner("regr.blm"),
  # makeLearner("regr.brnn"),
  # makeLearner("regr.bst"),
  # makeLearner("regr.btgp"),
  # makeLearner("regr.btgpllm"),
  # makeLearner("regr.btlm"),
  # makeLearner("regr.cforest"),
  # # makeLearner("regr.crs"),
  # makeLearner("regr.ctree"),
  # makeLearner("regr.cubist"),
  # makeLearner("regr.cvglmnet"),
  # makeLearner("regr.earth"),
  # makeLearner("regr.elmNN"),
  # makeLearner("regr.evtree"),
  makeLearner("regr.extraTrees"),
  # makeLearner("regr.fnn"),
  # makeLearner("regr.featureless"),
  # makeLearner("regr.frbs"),
  # makeLearner("regr.gamboost"),
  # makeLearner("regr.gausspr"),
  makeLearner("regr.gbm")
  # makeLearner("regr.glm"),
  # makeLearner("regr.glmboost"),
  # makeLearner("regr.glmnet"),
  # makeLearner("regr.GPfit"),
  # makeLearner("regr.h2o.deeplearning"),
  # makeLearner("regr.h2o.glm"),
  # makeLearner("regr.h2o.randomForest"),
  # makeLearner("regr.IBk"),
  # # makeLearner("regr.kknn"),
  # makeLearner("regr.km"),
  # makeLearner("regr.ksvm"),
  # makeLearner("regr.laGP"),
  # makeLearner("regr.LiblineaRL2L1SVR"),
  # makeLearner("regr.LiblineaRL2L2SVR"),
  # makeLearner("regr.lm"),
  # makeLearner("regr.mars"),
  # makeLearner("regr.mob"),
  # makeLearner("regr.nnet"),
  # makeLearner("regr.nodeHarvest"),
  # makeLearner("regr.pcr"),
  # makeLearner("regr.penalized.fusedlasso"),
  # makeLearner("regr.penalized.lasso"),
  # makeLearner("regr.penalized.ridge"),
  # makeLearner("regr.plsr"),
  # makeLearner("regr.randomForest"),
  # makeLearner("regr.randomForestSRC"),
  # makeLearner("regr.ranger"),
  # makeLearner("regr.rknn"),
  # makeLearner("regr.rpart"),
  # makeLearner("regr.RRF")
  # makeLearner("regr.rsm"),
  # makeLearner("regr.rvm"),
  # makeLearner("regr.slim"),
  # makeLearner("regr.svm"),
  # makeLearner("regr.xgboost"),
  # makeLearner("regr.xyf"),
  # makeLearner("regr.ctree"),
  # makeLearner("regr.blackboost")
)



indx <- sapply(sample_train, is.factor)
sample_train[indx] <- lapply(sample_train[indx], function(x) NULL)

indx <- sapply(sample_test, is.factor)
sample_test[indx] <- lapply(sample_test[indx], function(x) NULL)

sample_train$PoolArea <- NULL
sample_test$PoolArea <- NULL

regr.task <- makeRegrTask(id = "train5",
                          data = data.frame(sample_train[, !colnames(sample_train) %in% c("Id")]),
                          target = "SalePrice")

lrn = makeModelMultiplexer(base.learners)

ps = makeModelMultiplexerParamSet(lrn,
                                  makeIntegerParam("ntree", lower = 1L, upper = 2000L),
                                  makeIntegerParam("ntree", lower = 1L, upper = 2000L))

rdesc = makeResampleDesc("CV", iters = 2L)
ctrl = makeTuneControlIrace(maxExperiments = 200L)
res = tuneParams(lrn, regr.task, rdesc, par.set = ps, control = ctrl, show.info = FALSE)

#Iterated F-Racing for mixed spaces and dependencies ----


ps = makeParamSet(
  makeNumericParam("C", lower = -12, upper = 12, trafo = function(x) 2^x),
  makeDiscreteParam("kernel", values = c("vanilladot", "polydot", "rbfdot")),
  makeNumericParam("sigma", lower = -12, upper = 12, trafo = function(x) 2^x,
                   requires = quote(kernel == "rbfdot")),
  makeIntegerParam("degree", lower = 2L, upper = 5L,
                   requires = quote(kernel == "polydot"))
)
ctrl = makeTuneControlIrace(maxExperiments = 200L)
rdesc = makeResampleDesc("Holdout")
res = tuneParams("classif.ksvm", iris.task, rdesc, par.set = ps, control = ctrl, show.info = FALSE)
print(head(as.data.frame(res$opt.path)))














#fourth submission ----
pred <- predict(mod,
                newdata = data.frame(completed.test[, !colnames(completed.test) %in% c("Id")], stringsAsFactors = TRUE))

pred.antilog <- antilog(pred$data$response)

sample_submission <- list()
sample_submission$Id <- completed.test$Id
sample_submission$SalePrice <- pred.antilog
sample_submission <- data.frame(sample_submission)

write.csv(sample_submission,
          paste(getwd(), "sample_submission_4.csv", sep = "/"),
          row.names = FALSE)











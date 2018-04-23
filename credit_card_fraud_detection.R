# set pwd
setwd('/Users/alexchen/achen/Waterloo/Academic/4A/afm423/Project/')

# library
library(caret)
library(corrplot)
library(ellipse)
library(e1071)
library(psych)
library(randomForest)
library(tibble)
library(knitr)

# read csv
creditcard <- read.csv('creditcard.csv')

# set seed
set.seed(123456789)

# undersampling
creditcard$Class <- as.factor(creditcard$Class)
class_0_idx = which(creditcard$Class == 0) # legitimate
class_1_idx = which(creditcard$Class == 1) # fraudulent
nsamp = length(class_1_idx) # 492 obs

class_0_idx_under_sample = sample(class_0_idx, nsamp)
creditcard_undersample = creditcard[c(class_0_idx_under_sample,class_1_idx), ]
creditcard_undersample = creditcard_undersample[, -1]  # remove time column for training

# split train and test data
train_index = sample(1:nrow(creditcard_undersample), size = round(0.5 * nrow(creditcard_undersample)))
creditcard_trn = creditcard_undersample[train_index, ]
creditcard_tst = creditcard_undersample[-train_index, ]

# split X and Y into a factor and a matrix
trn_X = as.matrix(creditcard_trn[, -1])
trn_y = as.factor(creditcard_trn$Class)

# Get the test classification accuracy 
get_class_acc = function(model, data, response) {
  mean(data[,response] == predict(model,data))
}

# Get the false negative rate
get_false_negative_rate = function(model, data, response) {
  cm = confusionMatrix(predict(model,data),data[,response])
  fn = cm$table[1,2]
  tp = cm$table[1,1]
  fn/(tp+fn)
}

# Random Forest
cc_rf_tuned = train(Class~., data=creditcard_trn, 
                    trControl=trainControl(method="oob"),
                    method = "rf", tuneGrid =expand.grid(mtry=1:10))

cc_rf_random_tuned_tunelength10 = train(Class~., data=creditcard_trn, 
                                        trControl=trainControl(method="oob"),
                                        method = "rf", tuneLength = 10)

cc_rf = randomForest(Class~., 
                     data=creditcard_trn,
                     mtry = 5, 
                     importance = TRUE, 
                     ntrees = 500)

cc_rf_tst_pred = predict(cc_rf, newdata = creditcard_tst)

cc_rf_cv = train(Class~., data=creditcard_trn, trControl=trainControl(method="cv",number=10), method="rf", tuneLength=10)

# making list of models 
model_list = list(cc_rf_tuned, cc_rf_random_tuned_tunelength10, cc_rf_cv)

# Get Test Accuracy
test_rmse = sapply(model_list, get_class_acc, data = creditcard_tst, response = "Class")
fnr = sapply(model_list, get_false_negative_rate, data = creditcard_tst, response = "Class")
results = data.frame(
  titles = c("cc_rf_tuned",
             "cc_rf_random_tuned_tunelength10",
             "cc_rf_cv"),
  test_rmse,
  fnr
)

colnames(results) = c("model", "test accuracy","false negative rate")
knitr::kable(results, escape = FALSE, booktabs = TRUE)



# SVM

# Setting up the train control for SVM
trnCtrl = trainControl(method="cv",number = 10)

# SVM with linear kernel before preprocessing
svm_linear = train(Class ~.,
                   data = creditcard_trn,
                   method = "svmLinear",
                   trControl = trnCtrl,
                   tuneGrid = expand.grid(C = c(2 ^ (-2:5))))

# SVM with linear kernel after preprocessing
svm_linear_preprocess = train(Class ~ .,
                              data = creditcard_trn,
                              method = "svmLinear",
                              trControl=trnCtrl,
                              preProcess = c("center","scale"),
                              tuneGrid = expand.grid(C = c(2 ^ (-2:5))))

# SVM with radial kernel before preprocessing
svm_radial = train(Class ~ .,
                   data = creditcard_trn,
                   method = "svmRadial",
                   trControl=trnCtrl,
                   tuneLength = 10)

# SVM with a radial kernel after preprocessing
svm_radial_preprocess = train(Class ~ .,
                              data = creditcard_trn,
                              method = "svmRadial",
                              trControl = trnCtrl,
                              preProcess = c("center", "scale"),
                              tuneLength = 10)

# SVM with a polynomial kernel before preprocessing
svm_poly = train(Class ~ .,
                 data = creditcard_trn,
                 method = "svmPoly",
                 trControl = trnCtrl,
                 tuneLength = 5)

# SVM with a polynomial kernel after preprocessing
svm_poly_preprocess = train(Class ~ .,
                            data = creditcard_trn,
                            method = "svmPoly",
                            trControl = trnCtrl,
                            preProcess = c("center", "scale"),
                            tuneLength = 5)

# SVM with weighted costs (emphasize on fraudulent transactions)
costs = table(creditcard_trn$Class)
# Legitimate transactions
costs[1] = 1
# Fraudulent transactions
costs[2] = 1e10
svm_weighted_cost = svm(Class ~ .,
                        data = creditcard_trn,
                        type = 'C-classification',
                        kernel = 'polynomial',
                        class.weights=costs,
                        degree = 1,
                        scale=FALSE,
                        cross = 10)

# Creating a list of trained models
model_list = list(svm_linear,
                  svm_linear_preprocess,
                  svm_radial,
                  svm_radial_preprocess,
                  svm_poly,
                  svm_poly_preprocess,
                  svm_weighted_cost)

# Getting test accuracy
test_rmse = sapply(model_list, get_class_acc, data = creditcard_tst, response = "Class")
fnr = sapply(model_list, get_false_negative_rate, data = creditcard_tst, response = "Class")
results = data.frame(
  titles = c("svm linear",
             "svm linear (preprocess)",
             "svm radial",
             "svm radial (preprocess)",
             "svm poly",
             "svm poly (preprocess)",
             "svm poly weighted cost"),
  test_rmse,
  fnr
)
colnames(results) = c("model", "test accuracy","false negative rate")
knitr::kable(results, escape = FALSE, booktabs = TRUE)



# GBM

# Setting up the tune grid for the GBM model
gbm_grid = expand.grid(interaction.depth = 1:5,
                       n.trees = (1:30) * 100, 
                       shrinkage = c(0.001, 0.1, 0.5),     
                       n.minobsinnode = 10) 

# GBM Model
creditcard_gbm_mod = train(   
  Class ~ .,   
  data = creditcard_trn,   
  trControl = trainControl(method="cv",number = 5),   
  method = "gbm",   
  tuneGrid = gbm_grid,    
  verbose = FALSE )

# GBM Model with preprocessing
creditcard_gbm_mod_preprocess = train(   
  Class ~ .,   
  data = creditcard_trn,   
  trControl = trainControl(method="cv",number = 5),   
  method = "gbm",   
  tuneGrid = gbm_grid,
  preProcess = c("center","scale"),    
  verbose = FALSE )

# Creating a list of trained models
model_list = list(creditcard_gbm_mod, creditcard_gbm_mod_preprocess)

# Getting test accuracy
test_rmse = sapply(model_list, get_class_acc, data = creditcard_tst, response = "Class")
fnr = sapply(model_list, get_false_negative_rate, data = creditcard_tst, response = "Class")
results = data.frame(
  titles = c("gbm_mod","gbm_mod_preprocess"),
  test_rmse,
  fnr
)

colnames(results) = c("model", "test accuracy","false negative rate")
knitr::kable(results, escape = FALSE, booktabs = TRUE)


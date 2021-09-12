library(tidyverse)

Sys.setenv(JAVA_HOME="D:\\ZZZ Program Files\\Java")
library(rJava)

library(dataMaid)
library(explore)
library(keras)
library(dplyr)
require(foreign)
require(farff)
library(ggplot2)
library(tidyr)
library(tibble)

setwd("C:/Users/Hendrich/Desktop/Professional Internship/Predict_SAR_COVID2_Base-on_Sympton_h2o")

df <- read.csv("WuhanVirus.csv")
makeDataReport(df,
               render = FALSE,
               file = "EDA.Rmd",
               replace = TRUE)

Hmisc::describe(df)

View(df)
df$CHUAN_DOAN = df$CHUAN_DOAN%>%recode_factor(., `SAR-COVID2 atypical symptoms` = "SAR_Covid_2" ,
                                             `Acute respiratory pneumonia- Typhoid Infection` = "viem-Phoi",
                                             `Atypical cold symptoms` = "Cam")

View(df)
library(caret)
set.seed(123)
idTrain = caret::createDataPartition(y = df$CHUAN_DOAN, p = 900/1000 , list = FALSE)

trainset = df[idTrain,]
testset = df[-idTrain,]

View(testset)

library(h2o)
h2o.init(nthreads = -1 , max_mem_size = "1g")

wtrain = as.h2o(trainset)
wtest = as.h2o(testset)
View(wtrain)

reponse = "CHUAN_DOAN"
feature = setdiff(colnames(wtrain), reponse)

rfmod1 = h2o.randomForest(x = feature,
                                y = reponse,
                                training_frame = wtrain , nfolds = 12,
                                fold_assignment = "AUTO",
                                ntrees = 700 , max_depth =50, sample_rate =  0.5,
                                mtries = 4, balance_classes = FALSE,
                                stopping_metric = "logloss",
                                stopping_tolerance = 0.001,
                                stopping_rounds = 3,
                                keep_cross_validation_fold_assignment = TRUE,
                                keep_cross_validation_predictions=TRUE,
                                score_each_iteration = TRUE,
                                seed = 12345
                                )

rfmod2 = h2o.randomForest(x = feature,
                          y = reponse,
                          training_frame = wtrain , nfolds = 8,
                          fold_assignment = "Stratified",
                          ntrees = 700 , max_depth =50, sample_rate =  0.5,
                          mtries = 4, balance_classes = TRUE,
                          stopping_metric = "logloss",
                          stopping_tolerance = 0.001,
                          stopping_rounds = 3,
                          keep_cross_validation_fold_assignment = TRUE,
                          keep_cross_validation_predictions=TRUE,
                          score_each_iteration = TRUE,
                          seed = 12345)


rfmod3 = h2o.randomForest(x = feature,
                          y = reponse,
                          training_frame = wtrain , nfolds = 8,
                          fold_assignment = "Stratified",
                          ntrees = 700 , max_depth =50, sample_rate =  0.5,
                          mtries = 4, balance_classes = FALSE,
                          stopping_metric = "logloss",
                          stopping_tolerance = 0.001,
                          stopping_rounds = 3,
                          keep_cross_validation_fold_assignment = TRUE,
                          keep_cross_validation_predictions=TRUE,
                          score_each_iteration = TRUE,
                          seed = 12345)

rfmod4 = h2o.randomForest(x = feature,
                          y = reponse,
                          training_frame = wtrain , nfolds = 13,
                          fold_assignment = "AUTO",
                          ntrees = 500 , max_depth =50, sample_rate =  0.5,
                          mtries = 4, balance_classes = TRUE,
                          stopping_metric = "logloss",
                          stopping_tolerance = 0.001,
                          stopping_rounds = 3,
                          keep_cross_validation_fold_assignment = F,
                          keep_cross_validation_predictions=F,
                          score_each_iteration = TRUE,
                          seed = 12345)

h2o.performance(rfmod2,wtrain)
h2o.performance(rfmod2, wtest)
h2o.performance(rfmod3, wtest)
h2o.performance(rfmod4, wtest)
h2o.performance(rfmod1, wtest)

library(lime)
model_type.H2OModel <- function(x, ...) "classification"


predict_model.H2OModel <- function(x, newdata , type, ...){
  pred <- h2o.predict(x , as.h2o(newdata))
  return(as.data.frame(pred[,-1]))
}

set.seed(12345)
explainerdf <- lime(trainset[,-12], rfmod3, bin_continuous = FALSE, n_bins = 10, n_permutions = 10000)
library(magrittr)
library(dplyr)
duongtinhdf = subset(testset,CHUAN_DOAN == "SAR_Covid_2")%>%. [,-12]
amtindf = subset(testset, CHUAN_DOAN != "SAR_Covid_2")%>%. [,-12]



library(caret)

caseat <- lime::explain(amtindf[1,], explainer = explainerdf, labels = "Am tinh" , n_features = 11, feature_select = "auto")

case1 <- lime::explain(duongtinhdf[5,], explainer, labels = "SAR_Covid_2", n_features = 11 , 
                       feature_select =  "auto")

case1
plot_features(case1)

casevp <- lime::explain(viemphoidf[3,], explainer, labels = "viem-Phoi", n_features = 11 ,
                        feature_select = "highest_weights")

caseamtinh <- lime::explain(camdf[c(1:4),], explainer, labels = "cam", n_features = 10, feature_select = "auto")
case1
plot_features(caseamtinh)


exdf = case1 %>%as_tibble()
rlang::last_error()
exdf$Class = ifelse(exdf$label_prod >= 0.3,"SAR_Covid_2","Benh_Cam_Trieu_chung_Khong_Dien_Hinh")
exdf%>%
  ggplot(aes(x=reorder(case, feature_weight), y = feature_desc, fill=feature_weight,
             col = feature_weight)) + 
  geom_tile(show.legend = FALSE) + 
  scale_fill_gradient2(low="red4", mid = "white", high = "green4",midpoint = 0.0)+
  scale_color_gradient2(low = "red", mid = "grey" , high = "green" , midpoint = 0.0)+
  theme_bw()+
  theme(axis.text.x = element_blank()) + 
  facet_wrap(~Class, shrink = TRUE, scales = "free")


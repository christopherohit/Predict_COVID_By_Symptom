library(tidyverse)
library(tibble)
library(explore)
library(dataMaid)
library(summarytools)
Sys.setenv(JAVA_HOME="D:\\ZZZ Program Files\\Java")
library(rJava)
setwd("C:/Users/Hendrich/Desktop/Professional Internship/Death")
library(keras)
library(dplyr)
require(foreign)
require(farff)
library(ggplot2)
library(tidyr)
df=read.arff("Death.arff")

makeDataReport(df,
               render = FALSE,
               file = "EDA.Rmd",
               replace = TRUE)

names(df)=c("Level","FVC","FEV1","Zubrod",
            "Pain","Haemoptysis","Sultry","Cough","Weakness",
            "T_grade","DBtype2","MI","PAD","Smoking","Asthma","Age","Survival")
view(df)
df$Survival=df$Survival%>%recode_factor(.,`F` = "Survived", `T` = "Dead")
df$Tiffneau=df$FEV1/df$FVC

Hmisc::describe(df)

library(caret)
set.seed(123)

idTrain=caret::createDataPartition(y=df$Survival,p=410/470,list=FALSE)
trainset=df[idTrain,]
testset=df[-idTrain,]
testset = subset(testset , select = -c(Survival))

library(h2o)
h2o.init(nthreads = -1,max_mem_size ="1g")

wtrain=as.h2o(trainset)
wtest=as.h2o(testset)

response="Survival"
features=setdiff(colnames(wtrain),response)

# Balance Class and Stratifield
rfmod1=h2o.randomForest(x = features,
                        y = response,
                        training_frame = wtrain,nfolds=10,
                        fold_assignment = "Stratified",
                        balance_classes = TRUE,class_sampling_factors=c(1.17,6.66),
                        ntrees = 300, max_depth = 50,
                        stopping_metric = "logloss",
                        stopping_tolerance = 0.01,
                        stopping_rounds = 3,
                        keep_cross_validation_fold_assignment = TRUE,
                        keep_cross_validation_predictions=TRUE,
                        score_each_iteration = TRUE,
                        seed=12345)

#Balance + Not Stratifieds
rfmod2 = h2o.randomForest(x = features,
                          y = response,
                          training_frame = wtrain, nfolds = 10,
                          fold_assignment = "AUTO",class_sampling_factors=c(1.17,6.66),
                          ntrees = 100,max_depth = 50,sample_rate=0.5,mtries=4,
                          balance_classes = TRUE,
                          stopping_metric = "mean_per_class_error",
                          stopping_tolerance = 0.001,
                          stopping_rounds = 3,
                          keep_cross_validation_fold_assignment = F,
                          keep_cross_validation_predictions=F,
                          score_each_iteration = TRUE,
                          seed=12345)

#unBalance + Stratifields

rfmod3 = h2o.randomForest(x = features,
                          y = response,
                          training_frame =  wtrain, nfolds = 10,
                          fold_assignment = "Stratified",
                          ntrees = 100,max_depth = 50,sample_rate=0.5,mtries=4,
                          balance_classes = FALSE,
                          stopping_metric = "mean_per_class_error",
                          stopping_tolerance = 0.001,
                          stopping_rounds = 3,
                          keep_cross_validation_fold_assignment = F,
                          keep_cross_validation_predictions=F,
                          score_each_iteration = TRUE,
                          seed=12345)

#Unbalance + unstratified

rfmod4 = h2o.randomForest(x = features,
                          y = response,
                          training_frame = wtrain,nfolds=10,
                          balance_classes = F,
                          ntrees = 100, max_depth = 50,
                          stopping_metric = "logloss",
                          stopping_tolerance = 0.01,
                          stopping_rounds = 3,
                          keep_cross_validation_fold_assignment = TRUE,
                          keep_cross_validation_predictions=TRUE,
                          score_each_iteration = TRUE,
                          seed=12345)

h2o.performance(rfmod1,wtrain)

h2o.performance(rfmod1,wtest)

h2o.performance(rfmod2 , wtrain)

h2o.performance(rfmod2 , wtest)

h2o.performance(rfmod3 , wtrain)

h2o.performance(rfmod3 , wtest)

h2o.performance(rfmod4 , wtrain)

h2o.performance(rfmod4 , wtest)

library(lime)

# Adapting lime functions to h2O framework

model_type.H2OModel<- function(x, ...) "classification"

predict_model.H2OModel <- function(x, newdata, type, ...) {
  pred <- h2o.predict(x, as.h2o(newdata))
  return(as.data.frame(pred[,-1]))
}

set.seed(12345)
explainer <- lime(trainset[,-17], rfmod1, bin_continuous = FALSE, n_bins = 10, n_permutations = 10000)


library(magrittr)
library(dplyr)
deaddf=subset(testset,Survival!="Survived")%>%.[,-17]

survdf=subset(testset,Survival=="Survived")%>%.[,-17]


require(caret)

caseS1<-lime::explain(survdf[15,], explainer, labels="Survived",
                      n_features =10,feature_select="auto")

plot_features(caseS1)

case2 <- lime::explain(survdf[8,], explainer = explainer, labels = "Survived",
                       n_features = 10, feature_select = "auto")

plot_features(case2)

dead1<-lime::explain(deaddf[c(1:4),], explainer, labels="Dead",
                     n_features =10,feature_select="auto")
plot_features(dead1)

dead2<- lime::explain(deaddf[c(5:8),], explainer, labels = "Dead",
                      n_features = 10 , feature_select = "auto")
plot_features((dead2))

explanation <-lime::explain(testset[,-17], explainer, labels="Dead", n_features = 10,feature_select="auto")

exdf=explanation%>%as_tibble()

exdf$Class=ifelse(exdf$label_prob>=0.5,"Dead","Survived")

exdf%>%
  ggplot(aes(x=reorder(case,feature_weight),y=feature_desc,fill=feature_weight,col=feature_weight))+
  geom_tile(show.legend=F)+
  scale_fill_gradient2(low="red4",mid="white",high="green4",midpoint = 0.0)+
  scale_color_gradient2(low="red",mid="grey",high="green",midpoint = 0.0)+
  theme_bw()+
  theme(axis.text.x=element_blank())+
  facet_wrap(~Class,shrink=T,scale="free")



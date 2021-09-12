#Đầu tiên H2o chỉ thực sự hoạt động trên môi trường Java 8 - 15 nên chúng ta cần
#config một chút

Sys.setenv(JAVA_HOME="D:\\ZZZ Program Files\\Java")
library(rJava)


##
library(tidyverse)

my_theme <- function(base_size = 10, base_family = "sans"){
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      axis.text = element_text(size = 10),
      axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5),
      axis.title = element_text(size = 12),
      panel.grid.major = element_line(color = "grey"),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "#ffffef"),
      strip.background = element_rect(fill = "#ffbb00", color = "black", size =0.5),
      strip.text = element_text(face = "bold", size = 10, color = "black"),
      legend.position = "bottom",
      legend.justification = "center",
      legend.background = element_blank(),
      panel.border = element_rect(color = "grey30", fill = NA, size = 0.5)
    )
}
theme_set(my_theme())

mycolors=c("#f32440","#ffd700","#ff8c00","#c9e101","#c100e6","#39d3d6","#e84412")

require(foreign)
df=read.arff("https://archive.ics.uci.edu/ml/machine-learning-databases/00277/ThoraricSurgery.arff")%>%as_tibble()

names(df)=c("Diagnosis","FVC","FEV1","Zubrod","Pain","Haemoptysis","Dyspnoea","Cough","Weakness","T_grade","DBtype2","MI","PAD","Smoking","Asthma","Age","Survival")

df$Survival=df$Survival%>%recode_factor(.,`F` = "Survived", `T` = "Dead")

df$Tiffneau=df$FEV1/df$FVC

Hmisc::describe(df)

library(gridExtra)

a1=df%>%ggplot(aes(x=Survival,fill=Diagnosis))+geom_bar(position="fill",color="black",alpha=0.8,show.legend = T)+scale_fill_manual(values=mycolors)+coord_flip()+ggtitle("Diagnosis")

a2=df%>%ggplot(aes(x=Diagnosis,y=..count..,fill=Diagnosis))+geom_bar(color="black",alpha=0.8,show.legend =F)+scale_fill_manual(values=mycolors)+coord_flip()+facet_grid(Survival~.)

grid.arrange(a1,a2,ncol=1)

b1=df%>%ggplot(aes(x=Survival,fill=T_grade))+geom_bar(position="fill",color="black",alpha=0.8,show.legend = T)+scale_fill_manual(values=mycolors)+coord_flip()+ggtitle("T_grade")
b2=df%>%ggplot(aes(x=T_grade,y=..count..,fill=T_grade))+geom_bar(color="black",alpha=0.8,show.legend =F)+scale_fill_manual(values=mycolors)+coord_flip()+facet_grid(Survival~.)

grid.arrange(b1,b2,ncol=1)


df%>%gather(Pain:Weakness,DBtype2:Asthma,key="Features",value="Value")%>%ggplot(aes(x=Survival,y=..count..,fill=Value))+geom_bar(alpha=0.8,color="black")+facet_wrap(~Features,ncol=5)+scale_fill_manual(values=mycolors)

df%>%gather(Age,FEV1,FVC,Tiffneau,key="Features",value="Value")%>%ggplot(aes(x=Survival,y=Value,fill=Survival))+geom_boxplot(alpha=0.8,color="black")+coord_flip()+facet_wrap(~Features,ncol=1,scales="free")+scale_fill_manual(values=mycolors)


df%>%gather(Age,FEV1,FVC,Tiffneau,key="Features",value="Value")%>%ggplot(aes(x=Value,fill=Survival))+geom_density(alpha=0.6,color="black")+facet_wrap(~Features,ncol=2,scales="free")+scale_fill_manual(values=mycolors)

library(h2o)

h2o.init(nthreads = -1,max_mem_size ="4g")

library(caret)
set.seed(123)

idTrain=caret::createDataPartition(y=df$Survival,p=369/470,list=FALSE)
trainset=df[idTrain,]
testset=df[-idTrain,]


sp1=df%>%ggplot(aes(x=Survival,fill=Survival))+stat_count(color="black",alpha=0.7,show.legend = F)+scale_fill_manual(values=c("#f32440","#ffd700"))+coord_flip()+ggtitle("Origin")
sp2=trainset%>%ggplot(aes(x=Survival,fill=Survival))+stat_count(color="black",alpha=0.7,show.legend = F)+scale_fill_manual(values=c("#f32440","#ffd700"))+coord_flip()+ggtitle("Train")
sp3=testset%>%ggplot(aes(x=Survival,fill=Survival))+stat_count(color="black",alpha=0.7,show.legend = F)+scale_fill_manual(values=c("#f32440","#ffd700"))+coord_flip()+ggtitle("Test")

grid.arrange(sp1,sp2,sp3,ncol=1)

# Use lib bit64 to run
library(bit64)

wtrain=as.h2o(trainset)
wtest=as.h2o(testset)
response="Survival"
features=setdiff(colnames(wtrain),response)


#RF learner

#Balanced + stratified

rfmod1=h2o.randomForest(x = features,
                        y = response,
                        training_frame = wtrain,nfolds=10,
                        fold_assignment = "Stratified",
                        balance_classes = TRUE,class_sampling_factors=c(1.17,6.66),
                        ntrees = 100, max_depth = 50,
                        stopping_metric = "logloss",
                        stopping_tolerance = 0.01,
                        stopping_rounds = 3,
                        keep_cross_validation_fold_assignment = TRUE,
                        keep_cross_validation_predictions=TRUE,
                        score_each_iteration = TRUE,
                        seed=12345)

#Balanced + Not stratified

rfmod2=h2o.randomForest(x = features,
                        y = response,
                        training_frame = wtrain,nfolds=10,
                        fold_assignment = "AUTO",
                        balance_classes = TRUE,class_sampling_factors=c(1.17,6.66),
                        ntrees = 100, max_depth = 50,
                        stopping_metric = "logloss",
                        stopping_tolerance = 0.01,
                        stopping_rounds = 3,
                        keep_cross_validation_fold_assignment = TRUE,
                        keep_cross_validation_predictions=TRUE,
                        score_each_iteration = TRUE,
                        seed=12345)

#Unbalanced + stratified

rfmod3=h2o.randomForest(x = features,
                        y = response,
                        training_frame = wtrain,nfolds=10,
                        fold_assignment = "Stratified",
                        balance_classes = FALSE,
                        ntrees = 100, max_depth = 50,
                        stopping_metric = "logloss",
                        stopping_tolerance = 0.01,
                        stopping_rounds = 3,
                        keep_cross_validation_fold_assignment = TRUE,
                        keep_cross_validation_predictions=TRUE,
                        score_each_iteration = TRUE,
                        seed=12345)

#Unbalanced + not stratified

rfmod0=h2o.randomForest(x = features,
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

h2o.performance(rfmod0,wtest)

h2o.performance(rfmod1,wtest)

h2o.performance(rfmod2,wtest)

h2o.performance(rfmod3,wtest)

library(mlr)

taskTS=mlr::makeClassifTask(id="Thorac",data=df,target="Survival",positive = "Dead")

learnerH2ORF=makeLearner(id="h2oRF","classif.h2o.randomForest", predict.type = "prob")

mlrRF=train(learner = learnerH2ORF, task=taskTS)


bootmlrPERF=function(h2omodel,data,i){
  d=data[i,]
  predmlr=predict(mlrRF,newdata=d)
  predh2o=predict(get(h2omodel),as.h2o(d))
  predmlr$data$response<-as.vector(predh2o$predict)
  predmlr$data$prob.Dead<-as.vector(predh2o$Dead)
  predmlr$data$prob.Dead<-as.vector(predh2o$Survived)
  mets=list(bac,f1,tpr,tnr,fpr,fnr)
  p=mlr::performance(predmlr,mets)
  BAC=p[[1]]
  F1=p[[2]]
  TPR=p[[3]]
  TNR=p[[4]]
  FPR=p[[5]]
  FNR=p[[6]]
  return=cbind(BAC,F1,TPR,TNR,FPR,FNR)
}


set.seed(123)
library(boot)

perfmod0=boot(statistic=bootmlrPERF,h2omodel="rfmod0",data=df,R=30)%>%.$t%>%as_tibble()%>%mutate(Mode="Unbalanced_Random",iter=as.numeric(rownames(.)))

perfmod1=boot(statistic=bootmlrPERF,h2omodel="rfmod1",data=df,R=30)%>%.$t%>%as_tibble()%>%mutate(Mode="Balanced_Stratified",iter=as.numeric(rownames(.)))

perfmod2=boot(statistic=bootmlrPERF,h2omodel="rfmod2",data=df,R=30)%>%.$t%>%as_tibble()%>%mutate(Mode="Balanced_Random",iter=as.numeric(rownames(.)))

perfmod3=boot(statistic=bootmlrPERF,h2omodel="rfmod3",data=df,R=30)%>%.$t%>%as_tibble()%>%mutate(Mode="Unbalanced_Stratified",iter=as.numeric(rownames(.)))

bootperf=rbind(perfmod0,perfmod1,perfmod2,perfmod3)

names(bootperf)=c("BAC","F1","TPR","TNR","FPR","FNR","Mode","Iteration")

bootperf[,c(1:6)]%>%psych::describeBy(.,bootperf$Mode)


pairwise.wilcox.test(x=bootperf$FNR,g=bootperf$Mode,p.adjust.method="bonferroni",n=6,paired=T)

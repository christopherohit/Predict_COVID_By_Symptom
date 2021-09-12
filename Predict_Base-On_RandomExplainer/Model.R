library(tidyverse)
library(tibble)
library(summarytools)
library(readr)
library(explore)
library(dataMaid)

setwd("C:/Users/Hendrich/Desktop/Professional Internship/Predict_Base-On_RandomExplainer")

my_theme <- function(base_size =8, base_family = "Time new roman"){
  theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.major = element_line(color = "purple"),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = NA),
      strip.background = element_rect(fill = "#001d60", color = "#00113a", size =0.5),
      strip.text = element_text(face = "bold", size = 8, color = "white"),
      legend.position = "bottom",
      legend.justification = "center",
      legend.background = element_blank()
    )
}


theme_set(my_theme())


theme_bare <- function(base_size=8,base_family="Time new roman"){theme_bw(base_size = base_size, base_family = base_family)+
    theme(
      axis.line = element_blank(), 
      axis.text.x = element_blank(), 
      axis.text.y = element_blank(),
      axis.ticks = element_blank(), 
      axis.title.x = element_blank(), 
      axis.title.y = element_blank(), 
      legend.position = "bottom", 
      panel.background = element_rect(fill = NA), 
      panel.border = element_blank(), 
      panel.grid.major = element_blank(), 
      panel.grid.minor = element_blank(), 
      plot.margin = unit(c(0,0,0,0), "lines")
    )
}


data_frame <- read.csv("WuhanVirus2.csv")
data_frame = subset(data_frame,select = -c(X))
#EDA

view(dfSummary(data_frame))
explore(data_frame)
view(data_frame)
makeDataReport(data_frame,
               render = FALSE,
               file = "EDA.Rmd",
               replace = TRUE)

names(data_frame) = c("Cough",
                     "Fever",
                     "snivel",
                     "stuffy_nose",
                     "sneezing",
                     "sore_throat",
                     "sultry",
                     "sputum",
                     "vomit",
                     "diarrhea",
                     "distress",
                     "Class"
)

data_frame$Class = data_frame$Class %>% recode_factor(., `SAR-COVID2 atypical symptoms` = "SAR_Covid_2" ,
                                                      `Acute respiratory pneumonia- Typhoid Infection` = "viem_Phoi",
                                                      `Atypical cold symptoms` = "Cam")

library(caret)

set.seed(1234)
idtest = caret::createDataPartition(y = data_frame$Class,p = 800/1000,list = FALSE)

trainset = data_frame[idtest,]
testset = data_frame[-idtest,]

library(randomForest)

trainset$Class = factor(trainset$Class)

rfmod = randomForest(Class~.,
                     data=trainset,
                     ntree=700,
                     mtry = sqrt(11),
                     replace=TRUE,
                     localImp=TRUE
)
rfmod

testpred = predict(rfmod, testset)
testset$Class = as.factor(testset$Class)

confusionMatrix(factor(testpred),
                reference = testset$Class,
               # positive = "SAR_Covid_2",
                mode = "everything")

plot(rfmod,
     main = "Duong_Cong_Hoc_Tap_RF")
legend("topright", 
       c("error for 'Am Tinh'",
                     "Average misclassification error",
                     "error for 'Duong Tinh'"),
       lty = c(1,1,1),
       col = c("green" , "black" , "red"))
varImpPlot(rfmod)


varImp(rfmod)%>%as.data.frame()%>%
  mutate(Feature=rownames(.))%>%
  gather("Am Tinh","Duong Tinh",key="Label",value="Importance")%>%
  mutate(Importance=100*Importance/max(.$Importance))%>%
  ggplot(aes(x=reorder(Feature,Importance),
             y=Importance,
             fill=Importance,
             color=Importance))+
  geom_segment(aes(x=reorder(Feature,Importance),
                   xend=Feature,
                   y=0,
                   yend=Importance),
               size=1,
               show.legend = F)+
  geom_point(size=2,show.legend = F)+
  scale_x_discrete("Features")+
  coord_flip()+
  scale_fill_gradient(low="blue",high="red")+
  scale_color_gradient(low="blue",high="red")+
  geom_text(aes(label = round(Importance,1)),
            vjust=-0.5,size=3)+
  facet_wrap(~Label,ncol=2,scales="free")+
  theme_bw(base_family = "mono",8)

library(randomForestExplainer)

# display 7 var important
rfmod %>%
  important_variables(k = 7, ties_action = "draw")

rfmod %>% measure_importance(mean_sample = "relevant_trees") %>%
  as_tibble() ->vimdf


vimdf %>% head()
vimdf %>% gather(mean_min_depth: p_value,
                 key = "Metric",
                 value = "Value") %>%
  ggplot(aes(x = reorder(variable,Value),
             y = Value,
             fill = Metric,
             color= Metric)) + 
  geom_segment(aes(x = reorder(variable, Value),
                   xend = variable,
                   y = 0,
                   yend = Value),
               size = 1,
               show.legend = FALSE)+
  geom_point(size = 2, show.legend = FALSE)+
  scale_x_discrete("Features") + 
  coord_flip() + 
  facet_wrap(~Metric, ncol = 4 , scale = "free")+
  theme_bw(base_family = "Arial", 8)

plot_importance_ggpairs(vimdf)
plot_importance_rankings(vimdf)

# Auto Draw

library(GGally)

DrawSyntax <- function(data,mapping){
  p <- ggplot(data = data , mapping = mapping)+
    geom_point(aes(fill = rownames(data),
                   color= rownames(data)),
               shape = 21 , size = 2,
               alpha = 0.8,
               show.legend = FALSE)+
    geom_line(aes(color=rownames(data),
                  group = 1))+
    theme_bw(base_family = "Arial",8)
  p
}

plotfuncmid <- function(data,mapping){
  p <- ggplot(data = data, mapping = mapping)+
    geom_histogram(aes(fill = rownames(data),
                       color = rownames(data)),
                   alpha = 0.5)+
    theme_bw(base_family = "Arial",8)
  p
}

library(ggplot2)

ggpairs(vimdf,columns=2:8,
        lower=list(continuous=DrawSyntax()),
        diag=list(continuous=plotfuncmid))
#///

mindepthdf = rfmod %>% min_depth_distribution() %>% as_tibble()

mindepthdf %>% ggplot(aes(x = minimal_depth,
                          fill = variable,
                          col= variable)) + 
  geom_histogram(alpha=0.5) + 
  facet_wrap(~variable, ncol = 3, scales = "free") + 
  my_theme()


#
rlang::last_error()

rfmod %>% min_depth_distribution() %>%
  plot_min_depth_distribution(mean_sample = "all_trees") +
  scale_fill_brewer(palette = "Spectral", direction = -1)

mindprint = min_depth_interactions(rfmod, mean_sample = "relevant_trees",
                                   uncond_mean_sample = "relevant_trees")

warnings() 
library(viridis)

mindprint %>%
  plot_min_depth_interactions() + 
  scale_fill_viridis("A")
head(mindprint)

library(igraph)
library(ggraph)

gdf = filter(mindprint,
             occurrences >= 550)

graph <- graph_from_data_frame(gdf[,c(1:3,4,6)])


ggraph(graph,
       layout = 'kk',
       circular=F)+
  geom_edge_link(aes(colour = occurrences),
                 show.legend = T,width=1,alpha=0.7)+
  geom_node_label(aes(label = name))+
  coord_fixed()+
  scale_edge_color_gradient(high="#ff003f",low="#0094ff")+
  scale_edge_width_continuous()+
  theme_bare()

# Multiway Importance


plot_multi_way_importance(rfmod,
                          x_measure = "mean_min_depth",
                          y_measure = "times_a_root",
                          size_measure = "no_of_nodes",
                          min_no_of_trees = 100)


plot_multi_way_importance(vimdf, 
                          x_measure = "no_of_nodes", 
                          y_measure = "no_of_nodes", 
                          size_measure = "p_value",
                          min_no_of_trees = 100)


plot_multi_way_importance(vimdf, 
                          x_measure = "accuracy_decrease", 
                          y_measure = "gini_decrease", 
                          size_measure = "p_value",
                          min_no_of_trees = 100)


plot_multi_way_importance(vimdf, 
                          x_measure = "mean_min_depth", 
                          y_measure = "gini_decrease", 
                          size_measure = "p_value",
                          min_no_of_trees = 100)

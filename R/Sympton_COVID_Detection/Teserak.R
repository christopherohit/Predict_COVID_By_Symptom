library(tidyverse)
library(cluster)
library(stats)
library(norm)
library(Metrics)


### Import data

df = read.csv('Cleaned-Data.csv')

### EDA
## view data 
View(df)
## check data is null ?
is.null(df)
## describe data
summary(df)
## info in data type in data
names(df)
str(df)
# list the structure of df
levels(df)
# dimensions of an object
dim(df)
#class of an object (numeric, matrix, data frame, etc)
class(df)

head(df, n = 10)
tail(df, n = 10)

###

df.
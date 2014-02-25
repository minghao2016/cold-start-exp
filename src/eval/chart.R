# Create a chart comparing the algorithms

library("ggplot2")
library("grid")
library(data.table)


setwd('~/Dropbox/cold_start/cold-start-exp/')
result = fread('target/analysis/eval-results.csv')

# helper functions to process the file names
getNum = function(str) {
  l = strsplit(str, "_")[[1]]
  return(as.numeric(l[length(l)]))
} 

removeNum = function(str) {
  l = strsplit(str, "_")[[1]]
  return(paste(l[1:length(l)-1], collapse='_'))
} 

# create the final data frame for the plot
plt = result[, lapply(.SD, mean),by=list(DataSet, Algorithm)]
plt[, Strategy := sapply(DataSet, removeNum)]
plt[, Num := sapply(DataSet, getNum)]

# compute the average number of rated movies 
movie.rated = data.table('DataSet'=unique(plt$DataSet))
path = 'target/data/ml-1m-crossfold/'
for(i in 1:dim(movie.rated)[1]){
  re = glob2rx(paste("*.", 
                     strsplit(t[i], "\\.")[[1]][2], 
                     "_rated.csv", 
                     sep=""))
  fnames = grep(re, list.files(path), value=T)
  num.user = 0 
  num.rated = 0
  for(j in 1:length(fnames)) {
    df = fread(paste(path, fnames[j], sep=''))
    num.user = num.user + length(unique(df$user))
    num.rated = num.rated + sum(df$rated == 'True')
  }
  movie.rated[i, avg.rated := num.rated/num.user]
}

setkey(plt, DataSet)
setkey(movie.rated, DataSet)
plt = plt[movie.rated]


print(ggplot(plt, aes(x=Num, y=avg.rated, color=Strategy)) + 
        geom_line() + geom_point(size=4) +
        labs(x='# movies showed', y='average # movies rated'))

print(ggplot(plt, aes(x=Num, y=RMSE.ByUser, color=Strategy)) + 
        geom_line(aes(linetype=Algorithm)) + geom_point(size=4) +
        labs(x='# movies showed'))

print(ggplot(plt, aes(x=Num, y=MAE.ByUser, color=Strategy)) + 
        geom_line(aes(linetype=Algorithm)) + geom_point(size=4) + 
        labs(x='# movies showed'))

print(ggplot(plt, aes(x=Num, y=nDCG, color=Strategy)) + 
        geom_line(aes(linetype=Algorithm)) + geom_point(size=4) + 
        labs(x='# movies showed'))

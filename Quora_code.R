rm(list = ls())


#load data into R
setwd("D:/Kaam/Edwisor/Data +Others/Asignment/Hemant_Project")

data = read.csv("train.csv", header = T , stringsAsFactors = F, nrow = 1000)

# str(data)


#sentiment Analysis
library(stringr)
library(tm)
library(randomForest)
library('dplyr')
library("caret")
library(ggplot2)
library(ggthemes)
library(wordcloud)
library(wordcloud2)
library(stringdist)
library(slam)
library(class)
library(caret)
library(gridExtra)
library(stringi)
library("randomForest")
library(caret)
library(e1071)
library(clusterSim)
library(xgboost)
#Calculating different string methods without preprocessing
preprocess = function(all_text){
  
  #Converting all words to lower case
  all_text = data.frame(tolower(as.matrix(all_text)) , stringsAsFactors = FALSE)
  
  
  all_text = data.frame(apply(all_text,2 , function(y)
    stri_trans_general(y , "latin-ascii")) , stringsAsFactors = FALSE)
  
  
  all_text = data.frame(apply(all_text,2 , function(y)
    gsub("'ve", " have ", y)),stringsAsFactors = FALSE)
  
  all_text = data.frame(apply(all_text,2 , function(y)
    gsub("'s", " is ", y)),stringsAsFactors = FALSE)
  
  all_text = data.frame(apply(all_text,2 , function(y)
    gsub("can't", " cannot ", y)),stringsAsFactors = FALSE)
  
  all_text = data.frame(apply(all_text,2 , function(y)
    gsub("hadn't", " had not ", y)),stringsAsFactors = FALSE)
  
  all_text = data.frame(apply(all_text,2 , function(y)
    gsub("i'm", " i am ", y)),stringsAsFactors = FALSE)
  
  all_text = data.frame(apply(all_text,2 , function(y)
    gsub("'re", " are ", y)),stringsAsFactors = FALSE)
  
  all_text = data.frame(apply(all_text,2 , function(y)
    gsub("'d", " would ", y)),stringsAsFactors = FALSE)
  
  all_text = data.frame(apply(all_text,2 , function(y)
    gsub("'ll", " will ", y)), stringsAsFactors = FALSE)
  
  
  #Removing punctuation marks
  all_text = data.frame( apply(all_text, 2, function(y)
    gsub("[[:punct:]]", " ", y, perl = T)))
  
  
  
  #Delete extra spaces
  all_text = data.frame(lapply(all_text, function(y) 
    gsub("^ *|(?<= ) | *$", "", y, perl = TRUE)), stringsAsFactors = FALSE)
  
  
  
  return(all_text)
  
}



#Performing different methods on text
ques_matrix = function(all_text){
  
  m = matrix(NA, nrow = nrow(all_text), ncol = 11)
  
  #Length of Question 1
  m[,1] = sapply(all_text[,1], function(x) length(strsplit(x , " ")[[1]]))
  
  #Length of Question 2
  m[,2] = sapply(all_text[,2], function(x) length(strsplit(x , " ")[[1]]))
  
  #Differnece in lenght(positive)
  m[,3] = abs(m[,1]-m[,2])
  
  # Compute Distance with qgram method
  m[,4] = stringdist(all_text[,1],all_text[,2], method = "qgram")
  
  #measuring dissimilarity between strings
  m[,5] = stringdist(all_text[,1],all_text[,2] , method = "jw" , p=0)
  
  #Cosine Method
  m[,6] = stringdist(all_text[,1],all_text[,2] , method = "cosine")
  
  # Compute similarity scores between strings
  m[,7] = stringsim(all_text[,1],all_text[,2])
  
  #Longest Common Substring (LCS)
  m[,8] = stringdist(all_text[,1],all_text[,2] , method = "lcs" )
  
  # Full Damerau-Levenshtein distance.
  m[,9] = stringdist(all_text[,1],all_text[,2] , method = "dl" )
  
  # Jaccard Distance Method
  m[,10] = stringdist(all_text[,1],all_text[,2] , method = "jaccard")
  
  #Optimal String Alignment distance
  m[,11] = stringdist(all_text[,1],all_text[,2] , method = "osa" )
  
  colnames(m) = c("q1length","q2length","diff_length","dist",
                  "jw_meth","cosine_meth","simi","lcs","dl","jaccard","osa")
  
  return(m)
}


#Word Cloud Function
wc = function(text)
{
  
  corpus = Corpus(VectorSource(text))
  
  #Remove Stopwords and our predefined words
  corpus = tm_map(corpus, removeWords, c('i','its','it','us','use','want',
                                         'added','used','using','will','yes','say',
                                         'can','take','one',stopwords('english')))
  
  #remove unnecesary spaces
  corpus = tm_map(corpus, stripWhitespace)
  corpus = tm_map(corpus, removeNumbers)
  
  
  #Word cloud
  w = wordcloud(corpus, max.words = 100, scale=c(5, 1), colors=brewer.pal(8, "Dark2"))
  
  return(w)
  
}

#Plot Functions
line = function(dat,X,Y,title,nameX,nameY){
  l =  ggplot(aes_string(x=X,y=Y), data = dat) +
    geom_line(aes(color = 'red' ), stat='summary',fun.y=median) +
    labs(x = nameX , y = nameY) + ggtitle(title)
  return(l)
}

line2 = function(dat,X,Y,title,nameX,nameY){
  l =  ggplot(aes_string(x=X,y=Y), data = dat) +
    geom_line(aes(color = is_duplicate ), stat='summary',fun.y=median) +
    labs(x = nameX , y = nameY) + ggtitle(title)
  return(l)
}



#Text Minning

text_process = preprocess(data[,4:5])
text_process$is_duplicate = data$is_duplicate


text_process = data.frame(apply(text_process, 2, function(x) gsub("^$|^ $", NA, x)))
sum(is.na(text_process))


#Removing incomplete observations
text_process = text_process[complete.cases(text_process),]
str(text_process)


#Converting variables to their respective classes
text_process$question1 = as.character(text_process$question1)
text_process$question2 = as.character(text_process$question2)

t_matrix = ques_matrix(text_process[,1:2]) #distance matrices

#Combining the performed methods to rest of the data
text_process = cbind(text_process,t_matrix)

# data = text_process
str(text_process)
summary(text_process)




#Observations

  length(unique(data$question1))#unique ibservation in question1
  length(unique(data$question2))#unique ibservation in question2
  table(data$is_duplicate)
  
  summary(text_process[,1:3])
  
  sum(data$is_duplicate==0)/nrow(data)*100 #67.81 % data contains 0
  sum(data$is_duplicate==1)/nrow(data)*100 #37.19 % data contains 1
  
  sum(text_process$q1length>=7 & text_process$q1length<=13)/nrow(text_process)*100
  sum(text_process$q1length<7)/nrow(text_process)*100
  sum(text_process$q1length>13)/nrow(text_process)*100
  
  sum(text_process$q2length>=7 & text_process$q2length<=13)/nrow(text_process)*100
  sum(text_process$q2length<7)/nrow(text_process)*100
  sum(text_process$q2length>13)/nrow(text_process)*100
  
  
  table(text_process$is_duplicate[which(text_process$q2length>20)])
  table(text_process$is_duplicate[which(text_process$diff_length>1)])
  
  
  sum(text_process$is_duplicate==0 & text_process$diff_length>10)/nrow(text_process)*100
  sum(text_process$is_duplicate==1 & text_process$diff_length>10)/nrow(text_process)*100
  
  


#All Plots

  
  library(psych)
  multi.hist(text_process[,6:14], main = NA, dcol = c("blue", "red"),
             dlty = c("solid", "solid"), bcol = "grey95")
  
  h1=qplot(q1length, data = text_process, geom = "histogram", binwidth = 1, main = "Distribution of Q1 Length", color="080807")
  h2=qplot(q2length, data = text_process, geom = "histogram", binwidth = 1, main = "Distribution of Q2 Length",color="080807")
  grid.arrange(h1,h2)
  
 

  # Q1 Length Boxplot
  q1_l=qplot(is_duplicate,q1length, data = text_process, geom = "boxplot",
        main = "Boxplot of q1 length", xlab = "Duplicate", ylab = "Q1 Length")
  
  # Q2 Length Boxplot
  q2_l=qplot(is_duplicate,q2length, data = text_process, geom = "boxplot",
        main = "Boxplot of q2 length", xlab = "Duplicate", ylab = "Q2 Length")
  
  grid.arrange(q1_l,q2_l)
  
  
  
  d1=qplot(q1length, data = text_process, geom = "density", fill = is_duplicate)
  d2=qplot(q2length, data = text_process, geom = "density", fill = is_duplicate)
  grid.arrange(d1,d2)
  
  b1=qplot(q1length, data = text_process, geom = "bar", fill = is_duplicate)
  b2=qplot(q2length, data = text_process, geom = "bar", fill = is_duplicate)
  grid.arrange(b1,b2)
  
  qplot(diff_length, data = text_process, geom = "bar", fill = is_duplicate)
  
  
  
  
  l1=line(text_process,"q1length","dist","Q1 Length VS Distance","Q1 Length","Distance")
  l2=line(text_process,"q2length","dist","Q2 Length VS Distance","Q2 Length","Distance")
  grid.arrange(l1,l2)
  
  l3=line(text_process,"q1length","q2length","Q1 Length VS Q2 Length","Q1 Length","Q2 Length")
  l4=line2(text_process,"q1length","q2length","Q1 Length VS Q2 Length","Q1 Length","Q2 Length")
  grid.arrange(l3,l4)
  
  l5=line(text_process,"q1length","simi","Q1 Length VS Similarity","Q1 Length","Similarity")
  l6=line(text_process,"q2length","simi","Q2 Length VS Similarity","Q2 Length","Similarity")
  grid.arrange(l5,l6)
  
  
  l7=line(text_process,"diff_length","dist","Difference in Length VS Distance","Length Difference","Distance")
  l8=line2(text_process,"diff_length","dist","Difference in Length VS Distance","Length Difference","Distance")
  grid.arrange(l7,l8)
  
  
  l9=line(text_process,"diff_length","simi","Difference in Length VS String Similarity","Length Difference","String Similarity")
  l10=line2(text_process,"diff_length","simi","Difference in Length VS String Similarity","Length Difference","String Similarity")
  grid.arrange(l9,l10)
  
  
  rm(b1,b2,d1,d2,h1,h2,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10)
  
rm


#word clouds

q1_wc = wc(text_process[,1]) #for question1
q2_wc = wc(text_process[,2]) #for question2
  
  #Model Building
  
  var1=text_process
  
  var1 = data.Normalization(var1[,4:14], type = "n4", normalization = "column")
  var1$is_duplicate = text_process$is_duplicate
  
  train = var1[][sample(nrow(var1), 70000,replace = F),]
  test = var1[][!(1:nrow(var1)) %in% as.numeric(row.names(train)), ]
  
  
  
  fit_classify = randomForest(is_duplicate ~ ., train, importance = TRUE, ntree = 300)
  fit_classify 
  pred = predict(fit_classify, test[,-12])
  xtab = table(observed = test[,12], predicted = pred)
  confusionMatrix (xtab)
  
  
  #importing and Minning for test data set
  
  data1 = read.csv("test.csv", header = T , stringsAsFactors = F, nrow = 100000)
  
  
  text_process = preprocess(data1[,2:3])
 
   text_process = data.frame(apply(text_process, 2, function(x) gsub("^$|^ $", NA, x)))
  sum(is.na(text_process))
  
  
  
  #Removing incomplete observations
  text_process = text_process[complete.cases(text_process),]
  str(text_process)
  
  
  #Converting variables to their respective classes
  text_process$question1 = as.character(text_process$question1)
  text_process$question2 = as.character(text_process$question2)
  
  t_matrix = ques_matrix(text_process[,1:2]) #distance matrices
  
  #Combining the performed methods to rest of the data
  text_process = cbind(text_process,t_matrix)
  
  
  
  
  #Model Deployment
 
  var2 = text_process
  
  var2 = data.Normalization(var2[,3:13], type = "n4", normalization = "column")

  pred2 = predict(fit_classify, var2)

  output = cbind(text_process,pred2)
  
  names(output)[14] = "pred_is_duplicate"
  
  output = subset(output, select = c(question1, question2,pred_is_duplicate))
  write.csv(output, file = "pred_output.csv")
  
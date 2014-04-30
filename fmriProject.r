
#####################
### LOAD PACKAGES ###
#####################


library(R.matlab)     #for reading in the data
library(MASS)         #for LDA
library(e1071)        #for SVM, naive Bayes
#library(knn)         #Not available for R version >= 3.1.0
library(kknn)         
library(randomForest)

########################
###  DATA PROCESSING ###
########################

loadData <- function(wd){
  #read in data 
  setwd(wd)
  file <- "fmri1.mat"
  #file <- readline(prompt = "Please specify which file to read in: ")
  data <- readMat(file)
  data <- as.data.frame(data[[1]])
  return(data)
}

addClassLabels <- function(data){
  
  # Need to create a class label to the datasets.
  # Jack's code (lines 247-266) and thesis (P.22) indicate for that for dataset A,
  # patients 1-62 (62 total) are controls, 63-116 (54 total) have schizophrenia, and 117-164 (62 total) have bipolar disorder.
  class_labels <- as.factor(c(rep(0,62), rep(1,54), rep(2,48)))
  data <- cbind(class_labels, data)
  names(data)[1] <- "class.labels"
  return(data)
  
}

cutData <- function(data){
  
  data <- data[, 1:11]
  return(data)
  
}

#The function below returns the indices of the testing set.
partDataInd <- function(df, percentage) {
  
  train_size <- floor(percentage * nrow(df))
  train_ind <- sample((nrow(df)), size = train_size)
  return(train_ind)
  
}





##########################
### FEATURE SELECTION  ###
##########################

runLDA <- function(data){
  
  class_labels <- data$class.labels
  lda.model <- lda(class.labels ~ ., data)
  data.lda <- as.matrix(data[,2:ncol(data)])%*%lda.model$scaling  
  data.lda <- cbind("class.labels" = class_labels, as.data.frame(data.lda))
  return(data.lda)
  
}

####################
### CLASSIFIERS  ###
####################


#Radial Basis function (RBF) SVM
runRBFSVM <- function(data, p){
  
  test.ind <- partDataInd(data, p)
  rbsvm.model <- svm(class.labels ~ ., data = data, subset=test.ind, type = "C")
  rbsvm.model.pred <- predict(rbsvm.model, newdata=data[-test.ind,])
  rbsvm.table <- table(rbsvm.model.pred, data$class.labels[-test.ind])
  rbsvm.metrics <- calcMetrics(rbsvm.table)
  return(rbsvm.metrics)
  
}

#Linear SVM
runLinSVM <- function(data, p){
  
  test.ind <- partDataInd(data, p)
  linsvm.model <- svm(class.labels ~ ., data = data, subset=test.ind, type = "C")
  linsvm.model.pred <- predict(linsvm.model, newdata=data[-test.ind,])
  linsvm.table <- table(linsvm.model.pred, data$class.labels[-test.ind])
  linsvm.metrics <- calcMetrics(linsvm.table)
  return(linsvm.metrics)
  
}

#k-Nearest Neighbors


runKNN <- function(data, p){
  
  #Note: the knn package is not available for R 3.1.0. So I used the kknn package instead.
  
  # TO DO:
  # 1. decide whether we should try to find optimal k or not
  #class_labels <- data$class.labels 
  #knn.data <- knn(train = data[,-1], test = data[,-1], cl = class_labels, k = 5)
  #knn.data.table <- table(knn.data, class_labels)
  #return(knn.data.table)
  
  test.ind <- partDataInd(data, p)
  knn.model <- kknn(class.labels ~ ., train = data[test.ind,], test=data[-test.ind,], k=7)
  knn.model.pred <- fitted(knn.model)
  knn.table <- table(knn.model.pred, data$class.labels[-test.ind])
  knn.metrics <- calcMetrics(knn.table)
  return(knn.metrics)
  
}

# Random Forest
runRandomForest <- function(data, p){
  
  #TO DO: 
  #1. Change the number of trees used in the random forest classifier in Python (10).
  #2. Decide if we want to use the default number of trees in R (500).
  
  #ntrees <- 10  
  ntrees <- 500  
  test.ind <- partDataInd(data, p)
  rf.model <- randomForest(class.labels ~ ., data = data, subset=test.ind, type = "C")
  rf.model.pred <- predict(rf.model, newdata=data[-test.ind,])
  rf.table <- table(rf.model.pred, data$class.labels[-test.ind])
  rf.metrics <- calcMetrics(rf.table)
  return(rf.metrics)
  
}

#Gaussian Naive Bayes
runGaussianNaiveBayes <- function(data, p){
   
  test.ind <- partDataInd(data, p)
  gnb.model <- naiveBayes(class.labels ~ ., data = data, subset=test.ind)
  gnb.model.pred <- predict(gnb.model, newdata=data[-test.ind,])
  gnb.table <- table(gnb.model.pred, data$class.labels[-test.ind])
  gnb.metrics <- calcMetrics(gnb.table)
  return(gnb.metrics)
  
}

#Multinomial Naive Bayes

#4/19/14
#First attempt at finding an R package for multinomial naive bayes classification yielded no easy results.
#http://stackoverflow.com/questions/8874058/multinomial-naive-bayes-classifier?rq=1
#library(bnlearn)

####################
###   METRICS    ###
####################

#Definitions:

# tp = "true positive"
# tn = "true negative"
# fp = "false positive"
# fn = "false negative"
# "sensitivity" = tp/(tp+fn)
# "specificity" = tn/(fp+tn)
# "precision"   = tp/(tp+fp)
# "negative predictive value (npv)" = tn/(tn+fn)
# "false positive rate (fpr)"       = fp/(fp+tn)
# "false discovery rate (fdr)"      = fp/(fp+tp)
# "false negative rate (fnr)"       = fn/(fn+tp)
# "accuracy (ACC)"            = (tp+tn)/(tp+tn+fp+fn)
# "F1 score"                  =  2*tp/(2*tp + fp + fn)


calcMetrics <- function(confusion.mat){
  
  #This chunk calculates tp, fn, fp, fn for each class.
  v <- rep(0,3)
  confusion.table <- data.frame("tp" = v, "fp" = v, "tn" = v, "fn" = v)
  row.names(confusion.table) <- c("class.0", "class.1", "class.2")
  N <- sum(confusion.mat)
  for (i in 1:3) {
    
    confusion.table[i,1] <- confusion.mat[i,i] 
    confusion.table[i,2] <- sum(confusion.mat[i,]) - confusion.mat[i,i]
    confusion.table[i,3] <- sum(confusion.mat[,i]) - confusion.mat[i,i]
    confusion.table[i,4] <- N -  (confusion.table[i,1] + confusion.table[i,2] + confusion.table[i,3])
    
  }
  
  #This chunk calculates the performance metrics above for each class.
  metric.table <- data.frame("sensitivity" = v, "specificity" = v, "precision" = v, "npv" = v, "fpr" = v, "fdr" = v, "fnr" = v, "accuracy" = v, "f1.score" = v)
  row.names(metric.table) <- c("class.0", "class.1", "class.2")
  
  for (i in 1:3){
    
    metric.table$sensitivity[i] <- confusion.table$tp[i] / (confusion.table$tp[i] + confusion.table$fn[i])
    metric.table$specificity[i] <- confusion.table$tn[i] / (confusion.table$fp[i] + confusion.table$tn[i])
    metric.table$precision[i] <- confusion.table$tp[i] / (confusion.table$tp[i] + confusion.table$fp[i])
    metric.table$npv[i] <- confusion.table$tn[i] / (confusion.table$tn[i] + confusion.table$fn[i])
    metric.table$fpr[i] <- confusion.table$fp[i] / (confusion.table$fp[i] + confusion.table$tn[i])
    metric.table$fdr[i] <- confusion.table$fp[i] / (confusion.table$fp[i] + confusion.table$tp[i])
    metric.table$fnr[i] <- confusion.table$fn[i] / (confusion.table$fn[i] + confusion.table$tp[i])
    metric.table$accuracy[i] <- (confusion.table$tp[i] + confusion.table$tn[i]) / sum(confusion.table[i,])
    metric.table$f1.score[i] <- 2*confusion.table$tp[i] / (2*confusion.table$tp[i] + confusion.table$fp[i] + confusion.table$fn[i])
    
    if (any(is.nan(unlist(metric.table[i,])) == TRUE)) metric.table[i, which(is.nan(unlist(metric.table[i,])) == TRUE)] <- 0
    
  }
  return(metric.table)
    
}


#This function runs a classifier N times and computes the averages of each performance metric.

runNTimes <- function(classifier.fun, data, N, p){
  
  FUN <- match.fun(classifier.fun)
  metric.sums <- as.data.frame(matrix(rep(0,3*9),3,9))
  names(metric.sums) <- c("sensitivity", "specificity", "precision", "npv", "fpr", "fdr", "fnr", "accuracy", "f1.score")
  for ( i in 1:N){
    
    current.metric.table <- FUN(data, p)
    metric.sums <- metric.sums + current.metric.table
    
  }
  metric.averages <- metric.sums / N
  
  class.label <- c("control", "schizophrenia", "bipolar")
  classifier <- rep(substr(classifier.fun, 4, nchar(classifier.fun)), 3)
  metric.averages <- cbind(classifier, class.label, metric.averages)
  
  return(metric.averages)
  
}

runAll <- function(data, N, p){
  
  classifier.funs <- c("runRBFSVM", "runLinSVM", "runKNN", "runRandomForest", "runGaussianNaiveBayes")
  total.results <- data.frame() 
  for (i in classifier.funs) {
    
    current.result.table <- runNTimes(i, data, N, p)
    total.results <- rbind(total.results, current.result.table)
    
  }
  
  return(total.results)
   
}


#############
### MAIN ####
#############

main <- function(){
  
  #set working directory
  #wd <- "/home/dan/Dropbox/PythonRproject"
  #wd <- "C:\\Users\\dan\\Dropbox\\PythonRproject"
  #wd <- "/home/delores/Desktop/fMRI/data/"
  p <- 0.70
  N <- 100
  
  wd <- readline(prompt = "Please specify the path to the directory containing the data: ")
  #p <- as.numeric(readline(prompt = "Please specify a percentage (0.xx) of testing data: "))
  #N <- as.numeric(readline(prompt = "Please specify the number of times to run each classifier: "))
  data <- loadData(wd)
  data <- addClassLabels(data)
  data <- cutData(data)
  data <- runLDA(data)
  results <- runAll(data, N, p)
  return(results)
}


main()




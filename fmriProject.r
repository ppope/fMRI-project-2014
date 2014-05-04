


#TO DO:
#Extend functionality 
#  run on multiple training set sizes.
#  run multiple feature selection procedures


#####################
### LOAD PACKAGES ###
#####################


library(R.matlab)     #for reading in the data
library(MASS)         #for LDA
library(e1071)        #for SVM, naive Bayes
library(FNN)          #for k-nearest neighbors 
library(randomForest) #for random forest 
library(LiblineaR)    #for L2 regularized logistic regression

##########################
###  DATA PROCESSING #####
##########################

loadAndCombineData <- function(wd){
  
  setwd(wd)
  #read in data
  
  #There are two datasets in Jack's thesis: A and B.
  #Dataset A contains FMRI and FA data, with 163 subjects 
  #Dataset B contains FA, ALFF, and GM data, with 63 subjects
  
  #Read in dataset A
  files.A <- c("fmri_All.mat", "FA.mat")
  data.list.A <- vector(mode="list", length=3)
  for( i in 1:length(files.A))  data.list.A[[i]] <-  (readMat(files.A[i]))[[1]]  
  data.list.A[[3]] <- cbind(data.list.A[[1]], data.list.A[[2]])
  
  for (i in 1:length(data.list.A)) data.list.A[[i]] <- addClassLabels.A(data.list.A[[i]])
    
  #Read in dataset B
  files.B <- "Second_data.mat"
  data.list.B <- vector(mode="list", length=7)
  data.B <- readMat(files.B)
  data.list.B[[1]] <- data.B$FA
  data.list.B[[2]] <- data.B$ALFF
  data.list.B[[3]] <- data.B$GM
  data.list.B[[4]] <- cbind(data.list.B[[1]], data.list.B[[2]])
  data.list.B[[5]] <- cbind(data.list.B[[1]], data.list.B[[3]])
  data.list.B[[6]] <- cbind(data.list.B[[2]], data.list.B[[3]])
  data.list.B[[7]] <- cbind(data.list.B[[1]], data.list.B[[2]], data.list.B[[3]])

  for (i in 1:length(data.list.B)) data.list.B[[i]] <- addClassLabels.B(data.list.B[[i]])
    
  data.list <- append(data.list.A, data.list.B)

  names(data.list) <- c("fMRI.A", "FA.A", "(fMRI+FA).A", "FA.B", "ALFF.B", "GM.B", "(FA+ALFF).B", "(FA+GM).B", "(ALFF+GM).B", "(FA+ALFF+GM).B" )
  return(data.list)
  
}

# The next two functions create a class label for dataset A and dataset B respectively..
addClassLabels.A <- function(data){
  

  # Jack's code (lines 247-266) and thesis (P.22) indicate for that for dataset A,
  # patients 1-62 (62 total) are controls, 63-116 (54 total) have schizophrenia, and 117-164 (62 total) have bipolar disorder.
  class_labels <- as.factor(c(rep(0,62), rep(1,54), rep(2,48)))
  data <- cbind(class_labels, as.data.frame(data))
  names(data)[1] <- "class.labels"
  return(data)
  
}

addClassLabels.B <- function(data){
  
  # Jack's code (lines 330-342) and thesis (P.22) indicate for that for dataset B,
  # patients 1-28 (28 total) are controls, 29-63 (35 total) have schizophrenia. There are no bipolar disorder subjects.
  class_labels <- as.factor(c(rep(0,28), rep(1,35)))
  data <- cbind(class_labels, as.data.frame(data))
  names(data)[1] <- "class.labels"
  return(data)
  
}

cutData <- function(data){
  
  N <- 10
  data <- data[, c(1, sample(2:ncol(data), N))]
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
  
  train.ind <- partDataInd(data, p)
  rbsvm.model <- svm(class.labels ~ ., data = data, subset=train.ind, type = "C")
  rbsvm.model.pred <- predict(rbsvm.model, newdata=data[-train.ind,])
  rbsvm.table <- table(rbsvm.model.pred, data$class.labels[-train.ind])
  rbsvm.metrics <- calcMetrics(rbsvm.table)
  return(rbsvm.metrics)
  
}

#Linear SVM
runLinSVM <- function(data, p){
  
  train.ind <- partDataInd(data, p)
  linsvm.model <- svm(class.labels ~ ., data = data, subset=train.ind, type = "C")
  linsvm.model.pred <- predict(linsvm.model, newdata=data[-train.ind,])
  linsvm.table <- table(linsvm.model.pred, data$class.labels[-train.ind])
  linsvm.metrics <- calcMetrics(linsvm.table)
  return(linsvm.metrics)
  
}

#k-Nearest Neighbors
runKNN <- function(data, p){
  # TO DO:
  # 1. decide whether we should try to find optimal k or not
  
  train.ind <- partDataInd(data, p)
  knn.model <- knn(train = data[train.ind,-1, drop=FALSE], test = data[-train.ind,-1, drop=FALSE], cl = data$class.labels[train.ind], k = 5)
  knn.table <- table(knn.model, data$class.labels[-train.ind])
  n.classes <- ncol(knn.table)
  if ((n.classes == 2) & (dim(knn.table)[1] != 2)) knn.table <- rbind(knn.table, c(0,0)) 
  if ((n.classes == 3) & (dim(knn.table)[1] != 3)) knn.table <- rbind(knn.table, c(0,0,0)) 
  #The above are necessary because the knn model sometimes makes no classifications for a class, 
  #i.e. makes the same class prediction for all samples in the testing set. 
  #This throws off subscripting in the calcMetrics function below.
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
  train.ind <- partDataInd(data, p)
  rf.model <- randomForest(class.labels ~ ., data = data, subset=train.ind, type = "C")
  rf.model.pred <- predict(rf.model, newdata=data[-train.ind,])
  rf.table <- table(rf.model.pred, data$class.labels[-train.ind])
  rf.metrics <- calcMetrics(rf.table)
  return(rf.metrics)
  
}


#L2 Regularized Logistic Regression
# Python and R implementations use the same library
runLogisticRegression <- function(data, p){
  
  train.ind <- partDataInd(data, p)
  logreg.model <- LiblineaR(data = data[train.ind,-1, drop=FALSE], labels = data$class.labels[train.ind], type=0, eps = .0001, verbose = FALSE)
  logreg.model.pred <- predict(logreg.model, data[-train.ind,-1, drop=FALSE])[[1]]
  logreg.table <- table(logreg.model.pred, data$class.labels[-train.ind])
  
  n.classes <- ncol(logreg.table)
  if ((n.classes == 2) & (dim(logreg.table)[1] != 2)) logreg.table <- rbind(logreg.table, c(0,0)) 
  if ((n.classes == 3) & (dim(logreg.table)[1] != 3)) logreg.table <- rbind(logreg.table, c(0,0,0)) 
  #The above are necessary because the logistic model sometimes makes no classifications for a class, 
  #i.e. makes the same class prediction for all samples in the testing set. 
  #This throws off subscripting in the calcMetrics function below.
  logreg.metrics <- calcMetrics(logreg.table)
  return(logreg.metrics)
  
}

#Gaussian Naive Bayes
runGaussianNaiveBayes <- function(data, p){
   
  train.ind <- partDataInd(data, p)
  gnb.model <- naiveBayes(class.labels ~ ., data = data, subset=train.ind)
  gnb.model.pred <- predict(gnb.model, newdata=data[-train.ind,])
  gnb.table <- table(gnb.model.pred, data$class.labels[-train.ind])
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
  
  n.classes <- nrow(confusion.mat)
  #This chunk calculates tp, fn, fp, fn for each class.
  v <- rep(0,n.classes)
  confusion.table <- data.frame("tp" = v, "fp" = v, "tn" = v, "fn" = v)
  N <- sum(confusion.mat)
  for (i in 1:n.classes) {
    
    confusion.table[i,1] <- confusion.mat[i,i] 
    confusion.table[i,2] <- sum(confusion.mat[i,]) - confusion.mat[i,i]
    confusion.table[i,3] <- sum(confusion.mat[,i]) - confusion.mat[i,i]
    confusion.table[i,4] <- N -  (confusion.table[i,1] + confusion.table[i,2] + confusion.table[i,3])
    
  }
  
  #This chunk calculates the performance metrics above for each class.
  v <- rep(0,n.classes+1)
  metric.table <- data.frame("sensitivity" = v, "specificity" = v, "precision" = v, "npv" = v, "fpr" = v, "fdr" = v, "fnr" = v, "accuracy" = v, "f1.score" = v)
  
  for (i in 1:n.classes){
    
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
  
  metric.table[n.classes+1,] <- colMeans(metric.table[1:n.classes,])
  
  return(metric.table)
    
}


#This function runs a classifier N times and computes the averages of each performance metric.

runNTimes <- function(classifier.fun, data, N, p){
  
  FUN <- match.fun(classifier.fun)
  n.classes <- length(levels(data$class.labels))
  metric.sums <- as.data.frame(matrix(rep(0, (n.classes+1)*9), (n.classes+1),9))
  names(metric.sums) <- c("sensitivity", "specificity", "precision", "npv", "fpr", "fdr", "fnr", "accuracy", "f1.score")
  for ( i in 1:N){
    
    current.metric.table <- FUN(data, p)
    metric.sums <- metric.sums + current.metric.table
    
  }
  metric.averages <- metric.sums / N
  
  if (n.classes == 3) class.label <- c("control", "schizophrenia", "bipolar", "average.score")
  if (n.classes == 2) class.label <- c("control", "schizophrenia", "average.score")
  
  classifier <- rep(substr(classifier.fun, n.classes+1, nchar(classifier.fun)), n.classes+1)
  metric.averages <- cbind(classifier, class.label, metric.averages)
  
  return(metric.averages)
  
}

runAllClassifiers <- function(data, N, p){
  
  classifier.funs <- c("runRBFSVM", "runLinSVM", "runKNN", "runRandomForest", "runGaussianNaiveBayes", "runLogisticRegression")
  total.results <- data.frame() 
  for (i in classifier.funs) {
    
    current.result.table <- runNTimes(i, data, N, p)
    total.results <- rbind(total.results, current.result.table)
    
  }
  
  return(total.results)
   
}


runAllDatasets <- function(data.list, N, p){
  
  total.results <- data.frame()
  for (i in 1:length(data.list)){
    
    data <- data.list[[i]]
    data <- cutData(data)
    data <- runLDA(data)
    data.results <- runAllClassifiers(data, N, p)
    data.results <- cbind("dataset" = rep(names(data.list[i]), nrow(data.results)), data.results)
    total.results <- rbind(total.results, data.results)
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
  wd <- "/home/delores/Desktop/fMRI/data/"
  p <- 0.70
  N <- 100
  data.list <- loadAndCombineData(wd)
  results <- runAllDatasets(data.list, N, p)

  #"Second_data.mat" contains data for FA, ALFF, and GM.
  #"ttest_feature.mat"contains data for FA2, ALFF2, and GM2.
  #wd <- readline(prompt = "Please specify the path to the directory containing the data: ")
  #p <- as.numeric(readline(prompt = "Please specify a percentage (0.xx) of testing data: "))
  #N <- as.numeric(readline(prompt = "Please specify the number of times to run each classifier: "))

  return(results)
}


results <- main()




# install.packages("R.matlab")
# install.packages("e1071")

#####################
### LOAD PACKAGES ###
#####################


library(R.matlab)     #for reading in the data
library(MASS)         #for LDA
library(e1071)        #for SVM
library(fmri)

#####################
###    LOAD DATA  ###
#####################

#set working directory
#wd <- "/home/dan/Dropbox/PythonRproject"
#wd <- "C:\\Users\\dan\\Dropbox\\PythonRproject"
wd <- "/home/delores/Desktop/fMRI/data/"
setwd(wd)


#read in data 
#start out with first dataset (A)



loadData <- function(wd){
  
  setwd(wd)
  file <- readline(prompt = "Please specify which file to read in: ")
  data <- readMat(file)
  data <- as.data.frame(data[[1]])
  
}

addClassLabels <- function(data){
  
  # Need to create a class label to the datasets.
  # Jack's code (lines 247-266) and thesis (P.22) indicate for that for dataset A,
  # patients 1-62 (62 total) are controls, 63-116 (54 total) have schizophrenia, and 117-164 (62 total) have bipolar disorder.
  class_labels <- as.factor(c(rep(0,62), rep(1,54), rep(2,48)))
  
  
}


fmri1 <- cbind(class_labels, fmri1)
names(fmri1)[1] <- "class.labels"

fa1 <- cbind(class_labels, fa1)
names(fa1)[1] <- "class.labels"

#4/18/14
#Since we are running into memory problems with LDA, as a temporary fix we will work with the first ten (covariate) columns
#so we can work on the building the classifiers. 
fmri1.cut <- fmri1[, 1:11]


# create function for taking random sample with replacement
rowsample <- function(df, n) { 
  sampled_df <- df[sample(nrow(df), size=n, replace=FALSE),]
  return(sampled_df)
}

# create function for creating training set and test set
# percentage parameter is for size of training set
partdata <- function(df, percentage) {
  # sample row numbers to extract training set
  train_size <- floor(percentage * nrow(df))
  train_ind <- sample((nrow(df)), size = train_size)
  # create training set and test set
  train_set <- df[train_ind,]
  test_set <- df[-train_ind,]
  return(list(train_set, test_set))
}

#The function below returns the indices of the testing set instead of returning the dataframes.
partdataInd <- function(df, percentage) {
  
  train_size <- floor(percentage * nrow(df))
  train_ind <- sample((nrow(df)), size = train_size)
  return(train_ind)
  
}

#################
###   LDA     ###
#################

#Choose a proportion of testing data
p = 0.70

#4/11/14
# My machine is unable to run lda() on fmri1, and throws  the error cannot allocate vector of size 20.1 Gb.
#xx <- lda(fmri1, grouping=fmri1$class_labels, subset = partdataInd(fmri1, p))


#yy <- lda(fmri1.cut, grouping=fmri1.cut$class_labels, subset = partdataInd(fmri1.cut, p))
yy <- lda(fmri1.cut, grouping=class_labels)

#For future reference LDA works on the cut dataset...

#####################
### VISUALIZATION ###
#####################

#install.packages("fmri")
#install.packages("AnalyzeFMRI")

library(AnalyzeFMRI)

test <- readMat("fmri_All.mat")
x <- test[[1]]
xx <- x[1,]

image(matrix(xx[1:260], 10, 26))
image(matrix(xx[1:520], 52, 10))
image(matrix(xx[1:1040], 52, 20))

image(matrix(xx[1:65], 13, 5), main="WTF brain")

fmri1.mat <- as.matrix(fmri1[[1]])
image(fmri1.mat)  #This image is uninspiring.

prime.factors <- c(2,2,2,2,5,5,5,13)

nx <- 

#################
###  RBF SVM  ###
#################

rbsvm.fmri1 <- svm(class.labels ~ ., data = fmri1.cut, type = "C") 
rbsvm.fmri1.pred <- fitted(rbsvm.fmri1)
rbsvm.fmri1.table <- table(pred.rbsvm.fmri1, fmri1.cut$class.labels)
rbsvm.fmri1.table

#################
###  LIN SVM  ###
#################

linsvm.fmri1 <- svm(class.labels ~ ., data = fmri1.cut, type = "C", kernel = "linear") 
linsvm.fmri1.pred <- fitted(linsvm.fmri1)
linsvm.fmri1.table <- table(pred.linsvm.fmri1, fmri1.cut$class.labels)
linsvm.fmri1.table

#################
###   kNN     ###
#################

# decide whether we should try to find optimal k or not
knn.fmri1 <- knn(train = fmri1.cut[,-1], test = fmri1.cut[,-1],
                 cl = class_labels, k = 5)
knn.fmri1.table <- table(knn.fmri1, class_labels)
knn.fmri1.table


########################
###  RANDOM FOREST   ###
########################

library(randomForest)

#TO DO: 
#1. Change the number of trees used in the random forest classifier in Python (10).
#2. Decide if we want to use the default number of trees in R (500).

#ntrees <- 10  
ntrees <- 500
rf.fmri1 <- randomForest(class.labels ~ ., data=fmri1.cut, ntree=ntrees)


################################
###  GAUSSIAN NAIVE BAYES   ####
################################

#Note: the naiveBayes package used below assumes a Gaussian distribution for the predictors.

test.ind <- partdataInd(fmri1.cut, 0.70)
gnb.fmri1 <- naiveBayes(class.labels ~ ., data=fmri1.cut, subset=test.ind)
predict(gnb.fmri1, fmri1.cut[-test.ind,])


#################################
###  MULTIONMIAL NAIVE BAYES ####
#################################

#4/19/14
#First attempt at finding an R package for multinomial naive bayes classification yielded no easy results.
#http://stackoverflow.com/questions/8874058/multinomial-naive-bayes-classifier?rq=1
#library(bnlearn)


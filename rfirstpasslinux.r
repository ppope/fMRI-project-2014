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
fmri1 <- readMat("fmri_All.mat")
fmri1 <- as.data.frame(fmri1[[1]])

fa1 <- readMat("FA.mat")
fa1 <- as.data.frame(fa1[[1]])

# Need to create a class label to the datasets.
# Jack's code (lines 247-266) and thesis (P.22) indicate for that for dataset A,
# patients 1-62 (62 total) are controls, 63-116 (54 total) have schizophrenia, and 117-164 (62 total) have bipolar disorder.
class_labels <- as.factor(c(rep(0,62), rep(1,54), rep(2,48)))

fmri1 <- cbind(class_labels, fmri1)
names(fmri1)[1] <- "class.labels"

fa1 <- cbind(class_labels, fa1)
names(fa1)[1] <- "class.labels"

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

#4/18/14
#Since we are running into memory problems with LDA, as a temporary fix we will work with the first ten (covariate) columns
#so we can work on the building the classifiers. 

fmri1.cut <- fmri1[, 1:10]
#yy <- lda(fmri1.cut, grouping=fmri1.cut$class_labels, subset = partdataInd(fmri1.cut, p))
yy <- lda(fmri1.cut, grouping=class_labels)

#For future reference LDA works on the cut dataset...

#####################
### VISUALIZATION ###
#####################

#install.packages("fmri")

test <- readMat("fmri_All.mat")
fmri1.mat <- as.matrix(fmri1[[1]])
image(fmri1.mat)  #This image is uninspiring.


#################
###  RBF SVM  ###
#################

rbsvm.fmri1 <- svm(class_labels ~ ., data = fmri1.cut, type = "C") 
pred.rbsvm.fmri1 <- fitted(rbsvm.fmri1)
fmri1.rbsvm.table <- table(pred.rbsvm.fmri1, class_labels)
fmri1.rbsvm.table

#################
###  LIN SVM  ###
#################

linsvm.fmri1 <- svm(class_labels ~ ., data = fmri1.cut, type = "C", kernel = "linear") 
pred.linsvm.fmri1 <- fitted(linsvm.fmri1)
fmri1.linsvm.table <- table(pred.linsvm.fmri1, class_labels)
fmri1.linsvm.table









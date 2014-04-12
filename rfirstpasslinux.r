# install.packages("R.matlab")


#####################
### LOAD PACKAGES ###
#####################


library(R.matlab)
library(MASS)         #for LDA


#####################
###    LOAD DATA  ###
#####################

#set working directory
#wd <- "/home/dan/Dropbox/PythonRproject"
wd <- "/home/delores/Desktop/fMRI/data/"
setwd(wd)


#read in data 
#start out with first dataset (A)
fmri1 <- readMat("fmri_All.mat")
fmri1 <- as.data.frame(fmri1[[1]])

fa1 <- readMat("FA.mat")
fa1 <- as.data.frame(fa1[[1]])

# Need to add a class label to the datasets.
# Jack's code (lines 247-266) and thesis (P.22) indicate for that for dataset A,
# patients 1-62 (62 total) are controls, 63-116 (54 total) have schizophrenia, and 117-164 (62 total) have bipolar disorder.
class_labels <- as.factor(c(rep(0,62), rep(1,54), rep(2,48)))
fmri1 <- cbind(class_labels, fmri1)
fa1 <- cbind(class_labels, fa1)

# create function for taking random sample with replacement
rowsample <- function(df, n) { 
  sampled_df <- df[sample(nrow(df), size=n, replace=TRUE),]
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

partdataInd <- function(df, percentage) {
  
  train_size <- floor(percentage * nrow(df))
  train_ind <- sample((nrow(df)), size = train_size)
  return(train_ind)
  
}

#################
###   LDA     ###
#################

xx <- lda(fmri1, grouping=fmri1$class_labels, subset = partdataInd(fmri1, p))

#4/11/14
# My machine is unable to run lda() on fmri1, and throws  the error cannot allocate vector of size 20.1 Gb.






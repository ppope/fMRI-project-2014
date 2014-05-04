#4/11/2014
#This version was edited to remove the multinomial naive bayes classifer which was throwing errors.


#Jack Rogers
#Thesis 2012
#New College of Florida
#jack.rogers@ncf.edu
#954-376-2345

#to do

#allow comparison between ica and pca with differnt amounts of components



######################
### IMPORT MODULES ###
######################
import sklearn
import scipy
import math
import numpy
import scipy.io										 #to load in the mat file


from sklearn import decomposition					   #import decomposition for PCA
from sklearn.lda import LDA							 #import linear discriminant analysis
from sklearn.qda import QDA							 #import quadractic discriminatnt analysis

from sklearn import svm								 #to use the svm class
from sklearn.neighbors import KNeighborsClassifier	  #import KNN
from sklearn.cluster import KMeans					  #import kmeans clustering
from sklearn.naive_bayes import GaussianNB			  #import gaussian naive bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression	 #import logistic regression
from sklearn.ensemble import RandomForestClassifier	 #import random forests

from sklearn.cross_validation import cross_val_score	#import cross validation


##################################
### NEW DATA LOADING FUNCTIONS ###
##################################

def array_to_list(A):

	x = []
	for i in range(len(A)):
		x.append([])
		for j in A[i]:
			x[i].append(j)
	return x

def horzcat(A,B):
	AList = array_to_list(A)
	BList = array_to_list(B)
	for i in range(len(AList)):
		for j in range(len(BList[0])):
			AList[i].append(BList[i][j])
	return numpy.array(AList)

def vertcat(A,B):
	AList = array_to_list(A)
	BList = array_to_list(B)
	for i in range(len(BList)):
		AList.append(BList[i])
	return numpy.array(AList)

def horzcat_target(A,B):
	for i in range(len(B)):
		A.append(B[i])
	return A

def shuffle_inplace(a):
	p = numpy.random.permutation(len(a))
	return a[p]

def shuffle_in_unison_inplace(a, b):
	"""
	Inputs:  Two arrays of equal length

	Outputs:	Two shuffled arrays
	"""
	assert len(a) == len(b)
	p = numpy.random.permutation(len(a))
	return a[p], b[p]

def make_target(size,val):
	x = []
	for i in range(size):
		y = val
		x.append(y)
	return x

def random_sample(A,percent_training_size):
	shuffled = shuffle_inplace(A)
	start = len(shuffled)*0.01*percent_training_size
	start = int(start)
	Train = shuffled[:start]
	Test = shuffled[start:]
	TrainSize = len(Train)
	TestSize = len(Test)
	return Train,Test,TrainSize,TestSize

def random_sample_two_group(A,B,percent_training_size):
	ATrain,ATest,ATrainSize,ATestSize = random_sample(A,percent_training_size)
	BTrain,BTest,BTrainSize,BTestSize = random_sample(B,percent_training_size)

	#train on ATrain and BTrain concatenated
	#TRAINING SET
	#size pf training set is size sum of the size of percent_training_size percent of each Target group
	Train_data = vertcat(ATrain,BTrain)
	Train_target = horzcat_target(make_target(ATrainSize,0),make_target(BTrainSize,1))


	#TESTING SET
	#size pf testing set is size sum of the size of the remainer of percent_training_size percent of each Target group
	Test_data = vertcat(ATest,BTest)
	Test_target = horzcat_target(make_target(ATestSize,0),make_target(BTestSize,1))

	return Train_data,Train_target,Test_data,Test_target

def random_sample_three_group(A,B,C,percent_training_size):
	ATrain,ATest,ATrainSize,ATestSize = random_sample(A,percent_training_size)
	BTrain,BTest,BTrainSize,BTestSize = random_sample(B,percent_training_size)
	CTrain,CTest,CTrainSize,CTestSize = random_sample(C,percent_training_size)

	#train on ATrain and BTrain concatenated

	#TRAINING SET
	#size pf training set is size sum of the size of percent_training_size percent of each Target group
	Train_data2 = vertcat(ATrain,BTrain)
	Train_data = vertcat(Train_data2,CTrain)

	Train_target2 = horzcat_target(make_target(ATrainSize,0),make_target(BTrainSize,1))
	Train_target = horzcat_target(Train_target2,make_target(CTrainSize,2))

	#TESTING SET
	#size pf testing set is size sum of the size of the remainer of percent_training_size percent of each Target group
	Test_data2 = vertcat(ATest,BTest)
	Test_data = vertcat(Test_data2,CTest)

	Test_target2 = horzcat_target(make_target(ATestSize,0),make_target(BTestSize,1))
	Test_target = horzcat_target(Test_target2,make_target(CTestSize,2))


	return Train_data,Train_target,Test_data,Test_target

##########################
### DEFINE CLASSIFIERS ###
##########################

#1
def run_SVM_linear(training_data,training_target,testing_data,testing_target):
	svc = svm.SVC(kernel='linear')
	svc.fit(training_data,training_target)
	return svc.score(testing_data,testing_target)

#2
def run_SVM_RBF(training_data,training_target,testing_data,testing_target):
	svc = svm.SVC(kernel='rbf')
	svc.fit(training_data,training_target)
	return svc.score(testing_data,testing_target)

#3
#neighbors set to 5
def run_KNN(training_data,training_target,testing_data,testing_target):
	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

#4
def run_gaussian_naive_bayes(training_data,training_target,testing_data,testing_target):
	clf = GaussianNB()
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

#5
def run_bernoulli_naive_bayes(training_data,training_target,testing_data,testing_target):
	clf = BernoulliNB()
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

#6
def run_multinomial_naive_bayes(training_data,training_target,testing_data,testing_target):
	clf = MultinomialNB()
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

#7
def run_logistic_regression(training_data,training_target,testing_data,testing_target):
	clf = LogisticRegression()
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

#8
#estimators set to 10
def run_random_forest(training_data,training_target,testing_data,testing_target):
	clf = RandomForestClassifier(n_estimators=10)
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

def reduce_LDA(data,target):
	lda = LDA()
	return  lda.fit(data,target).transform(data)

def reduce_pca(data,target,components):
	pca = decomposition.PCA(n_components=components)
	return pca.fit(data).transform(data)

def reduce_ica(data,target,components):
	ica = decomposition.FastICA(n_components=components)
	return ica.fit(data).transform(data)



def run_all(training_data,training_target,testing_data,testing_target):
	#WITHOUT LDA
	results = []
	results.append(run_SVM_linear(training_data,training_target,testing_data,testing_target))
	results.append(run_SVM_RBF(training_data,training_target,testing_data,testing_target))
	results.append(run_KNN(training_data,training_target,testing_data,testing_target))
	results.append(run_gaussian_naive_bayes(training_data,training_target,testing_data,testing_target))
	results.append(run_bernoulli_naive_bayes(training_data,training_target,testing_data,testing_target))
      #Debugger's log 4/11/14: run_multinomial_naive_bayes throws an error when called. Error indicates that a nonnegative matrix 
      #must be passed to the naive_bayes function.
	#results.append(run_multinomial_naive_bayes(training_data,training_target,testing_data,testing_target))
	results.append(run_logistic_regression(training_data,training_target,testing_data,testing_target))
	results.append(run_random_forest(training_data,training_target,testing_data,testing_target))
	return results


#################
### LOAD DATA ###
#################

#data 1
def load_data1():
	mat1 = scipy.io.loadmat('/home/delores/Desktop/fMRI/data/fmri_All.mat')
	mat2 = scipy.io.loadmat('/home/delores/Desktop/fMRI/data/FA.mat')
	fmri_data = mat1['c']
	fa_data = mat2['fa']
	return fmri_data,fa_data

def make_control_data1(data):
	x = []
	for i in data[0:62]:
		x.append(i)
	control = numpy.array(x)
	return control

def make_schizophrenia_data1(data):
	x = []
	for i in data[62:116]:
		x.append(i)
	schizophrenia = numpy.array(x)
	return schizophrenia

def make_bipolar_data1(data):
	x = []
	for i in data[116:164]:
		x.append(i)
	bipolar = numpy.array(x)
	return bipolar

def make_three_group_target_data1():
	x = []
	for i in range(62):
		x.append(0)
	for i in range(54):
		x.append(1)
	for i in range(48):
		x.append(2)
	return numpy.array(x)

def data1_unpack():
	fmri, fa = load_data1()
	target = make_three_group_target_data1()
	merged = horzcat(fmri,fa)
	return fmri,fa,merged,target

def data1_lda(fmri,fa,merged,target):
	fmri_lda = reduce_LDA(fmri,target)
	fa_lda = reduce_LDA(fa,target)
	merged_lda = reduce_LDA(merged,target)
	return fmri_lda,fa_lda,merged_lda

def data1_ica(fmri,fa,merged,target,components):
	fmri_ica = reduce_ica(fmri,target,components)
	fa_ica = reduce_ica(fa,target,components)
	merged_ica = reduce_ica(merged,target,components)
	return fmri_ica,fa_ica,merged_ica

def data1_pca(fmri,fa,merged,target,components):
	fmri = fmri.byteswap().newbyteorder()
	fa = fa.byteswap().newbyteorder()
	merged = merged.byteswap().newbyteorder()
	fmri_pca = reduce_pca(fmri,target,components)
	fa_pca = reduce_pca(fa,target,components)
	merged_pca = reduce_pca(merged,target,components)
	return fmri_pca,fa_pca,merged_pca

def data1_subgroups(fmri_all,fa_all,merged_all):
	fmri_control = make_control_data1(fmri_all)
	fmri_schizophrenia = make_schizophrenia_data1(fmri_all)
	fmri_bipolar = make_bipolar_data1(fmri_all)

	fa_control = make_control_data1(fa_all)
	fa_schizophrenia = make_schizophrenia_data1(fa_all)
	fa_bipolar = make_bipolar_data1(fa_all)

	merged_all = horzcat(fmri_all,fa_all)
	merged_control = make_control_data1(merged_all)
	merged_schizophrenia = make_schizophrenia_data1(merged_all)
	merged_bipolar = make_bipolar_data1(merged_all)
	return fmri_control,fmri_schizophrenia,fmri_bipolar,fa_control,fa_schizophrenia,fa_bipolar,merged_control,merged_schizophrenia,merged_bipolar


#data 2
def load_data2():
	fn = '/home/delores/Desktop/fMRI/data/Second_data.mat'
	mat = scipy.io.loadmat(fn)
	FA = mat['FA']
	ALFF = mat['ALFF']
	GM = mat['GM']
	return FA,ALFF,GM

def make_control_data2(data):
	x = []
	for i in data[:28]:
		x.append(i)
	control = numpy.array(x)
	return control

def make_schizophrenia_data2(data):
	x = []
	for i in data[28:]:
		x.append(i)
	schizophrenia = numpy.array(x)
	return schizophrenia

def make_two_group_target_data2():
		x = []
		for i in range(28):
			x.append(0)
		for i in range(35):
			x.append(i)
		return numpy.array(x)

def data2_unpack():
	FA,ALFF,GM = load_data2()
	target = make_two_group_target_data2()
	ALL = horzcat(horzcat(FA,ALFF),GM)
	return FA,ALFF,GM,target,ALL

def data2_lda(FA,ALFF,GM,target,ALL):
	lda_FA = reduce_LDA(FA,target)
	lda_ALFF = reduce_LDA(ALFF,target)
	lda_GM = reduce_LDA(GM,target)
	lda_ALL = reduce_LDA(ALL,target)
	return lda_FA,lda_ALFF,lda_GM,lda_ALL

def data2_ica(FA,ALFF,GM,target,ALL,components):
	ica_FA = reduce_ica(FA,target,components)
	ica_ALFF = reduce_ica(ALFF,target,components)
	ica_GM = reduce_ica(GM,target,components)
	ica_ALL = reduce_ica(ALL,target,components)
	return ica_FA,ica_ALFF,ica_GM,ica_ALL

def data2_pca(FA,ALFF,GM,target,ALL,components):
	FA = FA.byteswap().newbyteorder()
	GM = GM.byteswap().newbyteorder()
	ALFF = ALFF.byteswap().newbyteorder()
	ALL = ALL.byteswap().newbyteorder()
	pca_FA = reduce_pca(FA,target,components)
	pca_ALFF = reduce_pca(ALFF,target,components)
	pca_GM = reduce_pca(GM,target,components)
	pca_ALL = reduce_pca(ALL,target,components)
	return pca_FA,pca_ALFF,pca_GM,pca_ALL

def data2_subgroups(FA,ALFF,GM,ALL):
	#FA
	FA_control = make_control_data2(FA)
	FA_schizophrenia = make_schizophrenia_data2(FA)

	#ALFF
	ALFF_control = make_control_data2(ALFF)
	ALFF_schizophrenia = make_schizophrenia_data2(ALFF)

	#GM
	GM_control = make_control_data2(GM)
	GM_schizophrenia = make_schizophrenia_data2(GM)

	#FA_ALFF
	FA_ALFF_control = horzcat(FA_control,ALFF_control)
	FA_ALFF_schizophrenia = horzcat(FA_schizophrenia,ALFF_schizophrenia)

	#FA_GM
	FA_GM_control = horzcat(FA_control,GM_control)
	FA_GM_schizophrenia = horzcat(FA_schizophrenia,GM_schizophrenia)

	#ALFF_GM
	ALFF_GM_control = horzcat(ALFF_control,GM_control)
	ALFF_GM_schizophrenia = horzcat(ALFF_schizophrenia,GM_schizophrenia)

	#ALL
	ALL_control = horzcat(FA_ALFF_control,GM_control)
	ALL_schizophrenia = horzcat(FA_ALFF_schizophrenia,GM_schizophrenia)
	return FA_control,FA_schizophrenia,ALFF_control,ALFF_schizophrenia,GM_control,GM_schizophrenia,FA_ALFF_control,FA_ALFF_schizophrenia,FA_GM_control,FA_GM_schizophrenia,ALFF_GM_control,ALFF_GM_schizophrenia,ALL_control,ALL_schizophrenia

#data 2
def load_data3():
	fn = '/home/delores/Desktop/fMRI/data/ttest_feature.mat'
	mat = scipy.io.loadmat(fn)
	FA = mat['fa2']
	ALFF = mat['alff2']
	GM = mat['gm2']
	return FA,ALFF,GM

def make_control_data3(data):
	x = []
	for i in data[:28]:
		x.append(i)
	control = numpy.array(x)
	return control

def make_schizophrenia_data3(data):
	x = []
	for i in data[28:]:
		x.append(i)
	schizophrenia = numpy.array(x)
	return schizophrenia

def make_two_group_target_data3():
		x = []
		for i in range(28):
			x.append(0)
		for i in range(35):
			x.append(i)
		return numpy.array(x)

def data3_unpack():
	FA,ALFF,GM = load_data2()
	target = make_two_group_target_data2()
	ALL = horzcat(horzcat(FA,ALFF),GM)
	return FA,ALFF,GM,target,ALL

def data3_lda(FA,ALFF,GM,target,ALL):
	lda_FA = reduce_LDA(FA,target)
	lda_ALFF = reduce_LDA(ALFF,target)
	lda_GM = reduce_LDA(GM,target)
	lda_ALL = reduce_LDA(ALL,target)
	return lda_FA,lda_ALFF,lda_GM,lda_ALL

def data3_ica(FA,ALFF,GM,target,ALL,components):
	ica_FA = reduce_ica(FA,target,components)
	ica_ALFF = reduce_ica(ALFF,target,components)
	ica_GM = reduce_ica(GM,target,components)
	ica_ALL = reduce_ica(ALL,target,components)
	return ica_FA,ica_ALFF,ica_GM,ica_ALL

def data3_pca(FA,ALFF,GM,target,ALL,components):
	FA = FA.byteswap().newbyteorder()
	GM = GM.byteswap().newbyteorder()
	ALFF = ALFF.byteswap().newbyteorder()
	ALL = ALL.byteswap().newbyteorder()
	pca_FA = reduce_pca(FA,target,components)
	pca_ALFF = reduce_pca(ALFF,target,components)
	pca_GM = reduce_pca(GM,target,components)
	pca_ALL = reduce_pca(ALL,target,components)
	return pca_FA,pca_ALFF,pca_GM,pca_ALL

def data3_subgroups(FA,ALFF,GM,ALL):
	#FA
	FA_control = make_control_data2(FA)
	FA_schizophrenia = make_schizophrenia_data2(FA)

	#ALFF
	ALFF_control = make_control_data2(ALFF)
	ALFF_schizophrenia = make_schizophrenia_data2(ALFF)

	#GM
	GM_control = make_control_data2(GM)
	GM_schizophrenia = make_schizophrenia_data2(GM)

	#FA_ALFF
	FA_ALFF_control = horzcat(FA_control,ALFF_control)
	FA_ALFF_schizophrenia = horzcat(FA_schizophrenia,ALFF_schizophrenia)

	#FA_GM
	FA_GM_control = horzcat(FA_control,GM_control)
	FA_GM_schizophrenia = horzcat(FA_schizophrenia,GM_schizophrenia)

	#ALFF_GM
	ALFF_GM_control = horzcat(ALFF_control,GM_control)
	ALFF_GM_schizophrenia = horzcat(ALFF_schizophrenia,GM_schizophrenia)

	#ALL
	ALL_control = horzcat(FA_ALFF_control,GM_control)
	ALL_schizophrenia = horzcat(FA_ALFF_schizophrenia,GM_schizophrenia)
	return FA_control,FA_schizophrenia,ALFF_control,ALFF_schizophrenia,GM_control,GM_schizophrenia,FA_ALFF_control,FA_ALFF_schizophrenia,FA_GM_control,FA_GM_schizophrenia,ALFF_GM_control,ALFF_GM_schizophrenia,ALL_control,ALL_schizophrenia

################################
### DEFINE TESTING FUNCTIONS ###
################################

def run_all_n_times(n,A,B,C,percent_training_size):
	results = []
	for i in range(n):
		Train_data,Train_target,Test_data,Test_target = random_sample_three_group(A,B,C,percent_training_size)
		results.append(run_all(Train_data,Train_target,Test_data,Test_target))
	return results

def run_comparison_n_times(n,A,B,percent_training_size):
	results = []
	for i in range(n):
		Train_data,Train_target,Test_data,Test_target = random_sample_two_group(A,B,percent_training_size)
		results.append(run_all(Train_data,Train_target,Test_data,Test_target))
	return results

def analyze_results(results,justMeans = True):
	svm_linear = []
	svm_rbf = []
	knn = []
	gaussian = []
	bernoulli = []
	#multinomial = []
	logistic = []
	forests = []
	for i in results:
		svm_linear.append(i[0])
		svm_rbf.append(i[1])
		knn.append(i[2])
		gaussian.append(i[3])
		bernoulli.append(i[4])
		#multinomial.append(i[5])
		logistic.append(i[5])
		forests.append(i[6])
	means = []
	means.append(numpy.mean(svm_linear))
	means.append(numpy.mean(svm_rbf))
	means.append(numpy.mean(knn))
	means.append(numpy.mean(gaussian))
	means.append(numpy.mean(bernoulli))
	#means.append(numpy.mean(multinomial))
	means.append(numpy.mean(logistic))
	means.append(numpy.mean(forests))
	mins = []
	mins.append(numpy.min(svm_linear))
	mins.append(numpy.min(svm_rbf))
	mins.append(numpy.min(knn))
	mins.append(numpy.min(gaussian))
	mins.append(numpy.min(bernoulli))
	#mins.append(numpy.min(multinomial))
	mins.append(numpy.min(logistic))
	mins.append(numpy.min(forests))
	maxs = []
	maxs.append(numpy.max(svm_linear))
	maxs.append(numpy.max(svm_rbf))
	maxs.append(numpy.max(knn))
	maxs.append(numpy.max(gaussian))
	maxs.append(numpy.max(bernoulli))
	#maxs.append(numpy.max(multinomial))
	maxs.append(numpy.max(logistic))
	maxs.append(numpy.max(forests))
	stds = []
	stds.append(numpy.std(svm_linear))
	stds.append(numpy.std(svm_rbf))
	stds.append(numpy.std(knn))
	stds.append(numpy.std(gaussian))
	stds.append(numpy.std(bernoulli))
	#stds.append(numpy.std(multinomial))
	stds.append(numpy.std(logistic))
	stds.append(numpy.std(forests))
	if justMeans == True:
		return maxs
	else:
		return means,mins,maxs,stds

def print_means(results,name):
	algorithms = ['svm linear','svm rbf','knn','gaussianNB','bernoulliBN','logistic','forests']
     #algorithms = ['svm linear','svm rbf','knn','gaussianNB','bernoulliBN','multinomialNB','logistic','forests']
	means = [results[0],results[1],results[2],results[3],results[4],results[5],results[6]]
     #means = [results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7]]
	#print "\tSVM Linear\tSVM RBF\tKNN\tGaussian\tBernoulli\tmultinomial\t\tLogistic Reg.\t\tRandom Forest"
	print name,"\tA:%5.5f\tB:%5.5f\tC:%5.5f\tD:%5.5f\tE:%5.5f\tG:%5.5f\tH:%5.5f" %(results[0],results[1],results[2],results[3],results[4],results[5],results[6])

def means_save(results,name):
	algorithms = ['svm linear','svm rbf','knn','gaussianNB','bernoulliBN','logistic','forests']
	means = [results[0],results[1],results[2],results[3],results[4],results[5],results[6]]
      #algorithms = ['svm linear','svm rbf','knn','gaussianNB','bernoulliBN','multinomialNB','logistic','forests']
    	#means = [results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7]]
	#print "\tSVM Linear\tSVM RBF\tKNN\tGaussian\tBernoulli\tmultinomial\t\tLogistic Reg.\t\tRandom Forest"
	return name + "\n\tA:%5.5f\tB:%5.5f\tC:%5.5f\tD:%5.5f\tE:%5.5f\tG:%5.5f\tH:%5.5f\n" %(results[0],results[1],results[2],results[3],results[4],results[5],results[6])

def test_data1(n,percent_training_size,fmri_all,fa_all,merged_all):
	print "A = Linear SVM"
	print "B = RBF SVM"
	print "C = KNN"
	print "D = gaussian naive bayes"
	print "E = bernoulli naive bayes"
	#print "F = multinomial naive bayes"
	print "G = logistic regression"
	print "H = random forests"
	fmri_control,fmri_schizophrenia,fmri_bipolar,fa_control,fa_schizophrenia,fa_bipolar,merged_control,merged_schizophrenia,merged_bipolar = data1_subgroups(fmri_all,fa_all,merged_all)
	print_means(analyze_results(run_all_n_times(n,fmri_control,fmri_schizophrenia,fmri_bipolar,percent_training_size)),'fmri\t')
	print_means(analyze_results(run_all_n_times(n,fa_control,fa_schizophrenia,fa_bipolar,percent_training_size)),'fa\t')
	print_means(analyze_results(run_all_n_times(n,merged_control,merged_schizophrenia,merged_bipolar,percent_training_size)),'merged')

	print_means(analyze_results(run_comparison_n_times(n,fmri_control,fmri_schizophrenia,percent_training_size)),'fmri cs')
	print_means(analyze_results(run_comparison_n_times(n,fa_control,fa_schizophrenia,percent_training_size)),'fa cs\t')
	print_means(analyze_results(run_comparison_n_times(n,merged_control,merged_schizophrenia,percent_training_size)),'merged cs')

	print_means(analyze_results(run_comparison_n_times(n,fmri_control,fmri_bipolar,percent_training_size)),'fmri cb')
	print_means(analyze_results(run_comparison_n_times(n,fa_control,fa_bipolar,percent_training_size)),'fa cb\t')
	print_means(analyze_results(run_comparison_n_times(n,merged_control,merged_bipolar,percent_training_size)),'merged cb')

	print_means(analyze_results(run_comparison_n_times(n,fmri_bipolar,fmri_schizophrenia,percent_training_size)),'fmri bs')
	print_means(analyze_results(run_comparison_n_times(n,fa_bipolar,fa_schizophrenia,percent_training_size)),'fa bs\t')
	print_means(analyze_results(run_comparison_n_times(n,merged_bipolar,merged_schizophrenia,percent_training_size)),'merged bs')

def test_data2(n,percent_training_size,FA,ALFF,GM,ALL):
	print "A = Linear SVM"
	print "B = RBF SVM"
	print "C = KNN"
	print "D = gaussian naive bayes"
	print "E = bernoulli naive bayes"
	#print "F = multinomial naive bayes"
	print "G = logistic regression"
	print "H = random forests"
	FA_control,FA_schizophrenia,ALFF_control,ALFF_schizophrenia,GM_control,GM_schizophrenia,FA_ALFF_control,FA_ALFF_schizophrenia,FA_GM_control,FA_GM_schizophrenia,ALFF_GM_control,ALFF_GM_schizophrenia,ALL_control,ALL_schizophrenia = data2_subgroups(FA,ALFF,GM,ALL)
	print_means(analyze_results(run_comparison_n_times(n,FA_control,FA_schizophrenia,percent_training_size)),'FA\t\t\t')
	print_means(analyze_results(run_comparison_n_times(n,ALFF_control,ALFF_schizophrenia,percent_training_size)),'ALFF\t\t\t')
	print_means(analyze_results(run_comparison_n_times(n,GM_control,GM_schizophrenia,percent_training_size)),'GM\t\t\t')
	print_means(analyze_results(run_comparison_n_times(n,FA_ALFF_control,FA_ALFF_schizophrenia,percent_training_size)),'FA_ALFF\t\t\t')
	print_means(analyze_results(run_comparison_n_times(n,FA_GM_control,FA_GM_schizophrenia,percent_training_size)),'FA_GM\t\t\t')
	print_means(analyze_results(run_comparison_n_times(n,ALFF_GM_control,ALFF_GM_schizophrenia,percent_training_size)),'ALFF_GM\t\t\t')
	print_means(analyze_results(run_comparison_n_times(n,ALL_control,ALL_schizophrenia,percent_training_size)),'ALL\t\t\t')

def test_data1_save(n,percent_training_size,fmri_all,fa_all,merged_all):
	fmri_control,fmri_schizophrenia,fmri_bipolar,fa_control,fa_schizophrenia,fa_bipolar,merged_control,merged_schizophrenia,merged_bipolar = data1_subgroups(fmri_all,fa_all,merged_all)
	lines = []
	lines.append(means_save(analyze_results(run_all_n_times(n,fmri_control,fmri_schizophrenia,fmri_bipolar,percent_training_size)),'fmri\t'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,fmri_control,fmri_schizophrenia,percent_training_size)),'fmri cs'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,fmri_control,fmri_bipolar,percent_training_size)),'fmri cb'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,fmri_bipolar,fmri_schizophrenia,percent_training_size)),'fmri bs'))
	lines.append(means_save(analyze_results(run_all_n_times(n,fa_control,fa_schizophrenia,fa_bipolar,percent_training_size)),'fa\t'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,fa_control,fa_schizophrenia,percent_training_size)),'fa cs\t'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,fa_control,fa_bipolar,percent_training_size)),'fa cb\t'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,fa_bipolar,fa_schizophrenia,percent_training_size)),'fa bs\t'))
	lines.append(means_save(analyze_results(run_all_n_times(n,merged_control,merged_schizophrenia,merged_bipolar,percent_training_size)),'merged'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,merged_control,merged_schizophrenia,percent_training_size)),'merged cs'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,merged_control,merged_bipolar,percent_training_size)),'merged cb'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,merged_bipolar,merged_schizophrenia,percent_training_size)),'merged bs'))
	return lines

def test_data2_save(n,percent_training_size,FA,ALFF,GM,ALL):
	FA_control,FA_schizophrenia,ALFF_control,ALFF_schizophrenia,GM_control,GM_schizophrenia,FA_ALFF_control,FA_ALFF_schizophrenia,FA_GM_control,FA_GM_schizophrenia,ALFF_GM_control,ALFF_GM_schizophrenia,ALL_control,ALL_schizophrenia = data2_subgroups(FA,ALFF,GM,ALL)
	lines = []
	lines.append(means_save(analyze_results(run_comparison_n_times(n,FA_control,FA_schizophrenia,percent_training_size)),'FA\t\t\t'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,ALFF_control,ALFF_schizophrenia,percent_training_size)),'ALFF\t\t\t'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,GM_control,GM_schizophrenia,percent_training_size)),'GM\t\t\t'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,FA_ALFF_control,FA_ALFF_schizophrenia,percent_training_size)),'FA_ALFF\t\t\t'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,FA_GM_control,FA_GM_schizophrenia,percent_training_size)),'FA_GM\t\t\t'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,ALFF_GM_control,ALFF_GM_schizophrenia,percent_training_size)),'ALFF_GM\t\t\t'))
	lines.append(means_save(analyze_results(run_comparison_n_times(n,ALL_control,ALL_schizophrenia,percent_training_size)),'ALL\t\t\t'))
	return lines

###########################
### Interface FUNCTIONS ###
###########################
def get_inputs():
	n = input("Enter n: ")
	p = input("Enter a list of percent training sizes: ")
	c = input("number of components for pca and ica: ")
	return n,p,c

def select_dataset():
	selection = raw_input("data1 or data2: ")
	if selection == 'data1':
		return 1
	elif selection == 'data2':
		return 2
	else:
		print "invalid input, try again"
		return select_dataset()

def select_method():
	method = raw_input("ica,pca,lda,or none: ")
	if method == 'none':
		return 0
	elif method == 'lda':
		return 1
	elif method == 'ica':
		return 2
	elif method == 'pca':
		return 3
	else:
		print "invalid input, try again"
		return select_method()
#####################
### MAIN FUNCTION ###
#####################
def main_save():
	n,p,c = get_inputs()
	output_fn = raw_input("enter outfile file path")
	fh = open(output_fn,'w')
	output_dict = {}

	#dataset 1
	fmri,fa,merged,target = data1_unpack()
	output_dict['data1 no feature extraction'] = test_data1_save(n,p,fmri,fa,merged)
	print "no feature extraction, data 1 finished"
	fmri_lda,fa_lda,merged_lda = data1_lda(fmri,fa,merged,target)
	output_dict['data1 lda'] = test_data1_save(n,p,fmri_lda,fa_lda,merged_lda)
	print "lda done"
	fmri_ica,fa_ica,merged_ica = data1_ica(fmri,fa,merged,target,c)
	output_dict['data1 ica'] = test_data1_save(n,p,fmri_ica,fa_ica,merged_ica)
	print "ica done"
	fmri_pca,fa_pca,merged_pca = data1_pca(fmri,fa,merged,target,c)
	output_dict['data1 pca'] = test_data1_save(n,p,fmri_pca,fa_pca,merged_pca)
	print "pca done"
	#dataset 2
	FA,ALFF,GM,target,ALL = data2_unpack()
	#output_dict['data2 no feature extraction'] = test_data2_save(n,p,FA,ALFF,GM,ALL)
	print "no feature extraction, data 2 finished"
	FA_lda,ALFF_lda,GM_lda,ALL_lda = data2_lda(FA,ALFF,GM,target,ALL)
	output_dict['data2 lda'] = test_data2_save(n,p,FA_lda,ALFF_lda,GM_lda,ALL_lda)
	print "lda done"
	FA_ica,ALFF_ica,GM_ica,ALL_ica = data2_ica(FA,ALFF,GM,target,ALL,c)
	output_dict['data2 ica'] = test_data2_save(n,p,FA_ica,ALFF_ica,GM_ica,ALL_ica)
	print "ica done"
	FA_pca,ALFF_pca,GM_pca,ALL_pca = data2_pca(FA,ALFF,GM,target,ALL,c)
	output_dict['data2 pca'] = test_data2_save(n,p,FA_pca,ALFF_pca,GM_pca,ALL_pca)
	print "writing to file"
	fh.write('N = %d\n'%(n))
	fh.write('P = %d\n'%(p))
	fh.write('C = %d\n'%(c))
	fh.write('\n')
	fh.write("ALGORITHM KEY\n")
	fh.write("A = Linear SVM\n")
	fh.write("B = RBF SVM\n")
	fh.write("C = KNN\n")
	fh.write("D = gaussian naive bayes\n")
	fh.write("E = bernoulli naive bayes\n")
	#fh.write("F = multinomial naive bayes\n")
	fh.write("G = logistic regression\n")
	fh.write("H = random forests\n")

	for i in output_dict.keys():
		fh.write('\n%s\n\n'%(i))
		for j in output_dict[i]:
			fh.write(j)


	fh.close()

def main_single_interactive():
	n,p,c = get_inputs()
	d = select_dataset()
	m = select_method()
	#dataset 1
	if d == 1:
		fmri,fa,merged,target = data1_unpack()
		#no feature selection
		if m == 0:
			test_data1(n,p,fmri,fa,merged)
		#lda
		elif m == 1:
			fmri_lda,fa_lda,merged_lda = data1_lda(fmri,fa,merged,target)
			test_data1(n,p,fmri_lda,fa_lda,merged_lda)
		elif m == 2:
			fmri_ica,fa_ica,merged_ica = data1_ica(fmri,fa,merged,target,c)
			test_data1(n,p,fmri_ica,fa_ica,merged_ica)
		elif m == 3:
			fmri_pca,fa_pca,merged_pca = data1_pca(fmri,fa,merged,target,c)
			test_data1(n,p,fmri_pca,fa_pca,merged_pca)

	#dataset 2
	elif d ==2:
		FA,ALFF,GM,target,ALL = data2_unpack()
		if m == 0:
			test_data2(n,p,FA,ALFF,GM,ALL)
		elif m == 1:
			FA_lda,ALFF_lda,GM_lda,ALL_lda = data2_lda(FA,ALFF,GM,target,ALL)
			test_data2(n,p,FA_lda,ALFF_lda,GM_lda,ALL_lda)
		elif m == 2:
			FA_ica,ALFF_ica,GM_ica,ALL_ica = data2_ica(FA,ALFF,GM,target,ALL,c)
			test_data2(n,p,FA_ica,ALFF_ica,GM_ica,ALL_ica)
		elif m == 3:
			FA_pca,ALFF_pca,GM_pca,ALL_pca = data2_pca(FA,
			ALFF,GM,target,ALL,c)
			test_data2(n,p,FA_pca,ALFF_pca,GM_pca,ALL_pca)

def main_all():
	n,p,c = get_inputs()

	fmri,fa,merged,target = data1_unpack()
#   test_data1(n,p,fmri,fa,merged)
	fmri_lda,fa_lda,merged_lda = data1_lda(fmri,fa,merged,target)
	test_data1(n,p,fmri_lda,fa_lda,merged_lda)
	fmri_ica,fa_ica,merged_ica = data1_ica(fmri,fa,merged,target,c)
	test_data1(n,p,fmri_ica,fa_ica,merged_ica)
	fmri_pca,fa_pca,merged_pca = data1_pca(fmri,fa,merged,target,c)
	test_data1(n,p,fmri_pca,fa_pca,merged_pca)

	FA,ALFF,GM,target,ALL = data2_unpack()
#   test_data2(n,p,FA,ALFF,GM,ALL)
	FA_lda,ALFF_lda,GM_lda,ALL_lda = data2_lda(FA,ALFF,GM,target,ALL)
	test_data2(n,p,FA_lda,ALFF_lda,GM_lda,ALL_lda)
	FA_ica,ALFF_ica,GM_ica,ALL_ica = data2_ica(FA,ALFF,GM,target,ALL,c)
	test_data2(n,p,FA_ica,ALFF_ica,GM_ica,ALL_ica)
	FA_pca,ALFF_pca,GM_pca,ALL_pca = data2_pca(FA,
	ALFF,GM,target,ALL,c)
	test_data2(n,p,FA_pca,ALFF_pca,GM_pca,ALL_pca)

def main_lda_raw():
	output_fn = raw_input("enter outfile file path")
	fh = open(output_fn,'w')
	output_dict = {}
	fmri,fa,merged,target = data1_unpack()
	fmri_lda,fa_lda,merged_lda = data1_lda(fmri,fa,merged,target)
	FA1,ALFF1,GM1,target1,ALL1 = data2_unpack()
	FA_lda1,ALFF_lda1,GM_lda1,ALL_lda1 = data2_lda(FA1,ALFF1,GM1,target1,ALL1)
	FA2,ALFF2,GM2,target2,ALL2 = data3_unpack()
	FA_lda2,ALFF2_lda2,GM_lda2,ALL_lda2 = data3_lda(FA2,ALFF2,GM2,target2,ALL2)
	output_dict['fmri after lda, data set 1'] = fmri_lda
	output_dict['fa after lda, data set 1'] = fa_lda
	output_dict['fmri+fa after lda, data set 1'] = merged_lda

	output_dict['fa after lda, data set 2'] = FA_lda1
	output_dict['alff after lda, data set 2'] = ALFF_lda1
	output_dict['gm after lda, data set 2'] = GM_lda1
	output_dict['fa+alff+gm after lda, data set 2'] = ALL_lda1

	output_dict['fa after lda, data set 3'] = FA_lda1
	output_dict['alff after lda, data set 3'] = ALFF_lda1
	output_dict['gm after lda, data set 3'] = GM_lda1
	output_dict['fa+alff+gm after lda, data set 3'] = ALL_lda1

	for i in output_dict.keys():
		print "\n\n\n#####################\n\n\n"
		fh.write("\n\n\n#####################\n\n\n")
		#print i
		fh.write(i)
		fh.write("\n")
		#print "subjects = " + str(len(output_dict[i]))
		fh.write(str("subjects = " + str(len(output_dict[i]))))
		
		fh.write("\n")
		#print "columns = " + str(len(output_dict[i][0]))
		fh.write(str("columns = " + str(len(output_dict[i][0]))))
		fh.write("\n")
		for j in output_dict[i]:
			z = ""
			for y in j:
				z += '%0.6f\t' %(y)
			print z
			fh.write(z)
			fh.write('\n')


	fh.close()
	print "DONE"

def main():
	print "1 for all non-interactive"
	print "2 for single interactive"
	print "3 for all save to file"
	print "4 for lda output to file"
	x = input('')
	if x == 1:
		main_all()
	elif x == 2:
		main_single_interactive()
	elif x == 3:
		main_save()
	elif x == 4:
		main_lda_raw()
	else:
		print "you fucked up"
		main()

main()

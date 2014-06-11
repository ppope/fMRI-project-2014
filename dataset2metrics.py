#5/31/2014
#WARNING!!! CODE ONLY WORKS WITH THE F9 FUNCTION IN SPYDER. DOES NOT WORK WHEN
#RUN IN INTERPRETER AS USUAL. AM INVESTIGATING ISSUE CURRENTLY.

#5/30/2014
#This is a heavily edited version currently only running on the first data set

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
import scipy.io                                         #to load in the mat file


from sklearn import decomposition                       #import decomposition for PCA
from sklearn.lda import LDA                             #import linear discriminant analysis
from sklearn.qda import QDA                             #import quadractic discriminatnt analysis

from sklearn import svm                                 #to use the svm class
from sklearn.neighbors import KNeighborsClassifier      #import KNN
from sklearn.cluster import KMeans                      #import kmeans clustering
from sklearn.naive_bayes import GaussianNB              #import gaussian naive bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression     #import logistic regression
from sklearn.ensemble import RandomForestClassifier     #import random forests

from sklearn.cross_validation import cross_val_score    #import cross validation
from sklearn.metrics import confusion_matrix            #import confusion matrix

from collections import OrderedDict
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

    Outputs:    Two shuffled arrays
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
    pred = svc.fit(training_data,training_target).predict(testing_data)
    return confusion_matrix(testing_target, pred)

#2
def run_SVM_RBF(training_data,training_target,testing_data,testing_target):
    svc = svm.SVC(kernel='rbf')
    pred = svc.fit(training_data,training_target).predict(testing_data)
    return confusion_matrix(testing_target, pred)

#3
#neighbors set to 5
def run_KNN(training_data,training_target,testing_data,testing_target):
    clf = KNeighborsClassifier(n_neighbors=5)
    pred = clf.fit(training_data,training_target).predict(testing_data)
    return confusion_matrix(testing_target, pred)

#4
def run_gaussian_naive_bayes(training_data,training_target,testing_data,testing_target):
    clf = GaussianNB()
    pred = clf.fit(training_data,training_target).predict(testing_data)
    return confusion_matrix(testing_target, pred)

#5
def run_logistic_regression(training_data,training_target,testing_data,testing_target):
    clf = LogisticRegression()
    pred = clf.fit(training_data,training_target).predict(testing_data)
    return confusion_matrix(testing_target, pred)

#6
def run_random_forest(training_data,training_target,testing_data,testing_target):
    clf = RandomForestClassifier(n_estimators=500)
    pred = clf.fit(training_data,training_target).predict(testing_data)
    return confusion_matrix(testing_target, pred)

def reduce_pca(data,target,components):
    pca = decomposition.PCA(n_components=components)
    return pca.fit(data).transform(data)

def run_all(training_data,training_target,testing_data,testing_target):
    results = []
    results.append(run_SVM_linear(training_data,training_target,testing_data,testing_target))
    results.append(run_SVM_RBF(training_data,training_target,testing_data,testing_target))
    results.append(run_KNN(training_data,training_target,testing_data,testing_target))
    results.append(run_gaussian_naive_bayes(training_data,training_target,testing_data,testing_target))
    results.append(run_logistic_regression(training_data,training_target,testing_data,testing_target))
    results.append(run_random_forest(training_data,training_target,testing_data,testing_target))
    return results


#################
### LOAD DATA ###
#################

#data 2
def load_data2():
    fn = '/home/dan/Dropbox/PythonRproject/Second_data.mat'
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


################################
### DEFINE TESTING FUNCTIONS ###
################################

def run_comparison_n_times(n,A,B,percent_training_size):
    results = []
    for i in range(n):
        Train_data,Train_target,Test_data,Test_target = random_sample_two_group(A,B,percent_training_size)
        results.append(run_all(Train_data,Train_target,Test_data,Test_target))
    return results

def basic_counts(confusion_matrices):
    # returns tuples of lists
    true_positives = []    
    false_positives = []
    false_negatives = []
    true_negatives = []    
    
    for cm in confusion_matrices:
        true_positives.append(cm[0][0])
        false_positives.append(cm[0][1])
        false_negatives.append(cm[1][0])
        true_negatives.append(cm[1][1])
    return true_positives, false_positives, false_negatives, true_negatives

def collapse_confusion_matrices(confusion_matrices):
    # takes in list of confusion matrices
    # collapses multiclass confusion matrices for a "one vs rest" approach    
    # returns list of list of collapsed matrices for each level of response 
    n = len(confusion_matrices[0])    
    
    collapsed_matrices = []
    for cls in range(0,n):
        one_vs_rest = []
        for cm in confusion_matrices:
            col_m = np.array([[cm[cls][cls], np.sum(cm[cls]) - cm[cls][cls]], \
[np.sum(cm[:,cls]) - cm[cls][cls], np.sum(cm) + cm[cls][cls] - np.sum(cm[cls]) \
- np.sum(cm[:,cls])]])
            one_vs_rest.append(col_m)
        collapsed_matrices.append(one_vs_rest)
    return collapsed_matrices

def calc_average_metrics(counts):
    # returns ordered dictionary of average of common metrics
    tp, fp, fn, tn = counts    
    tp = np.array(tp)
    fp = np.array(fp)
    fn = np.array(fn)
    tn = np.array(tn)
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(fp+tn)
    precision = tp/(tp+fp)
    negative_predictive_value = tn/(tn+fn)
    false_positive_rate = fp/(fp+tn)
    false_discovery_rate = fp/(fp+tp)
    false_negative_rate = fn/(fn+tp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1_score = 2*tp/(2*tp + fp + fn)
    
    metrics = OrderedDict([
("sensitivity", np.mean(sensitivity)),
("specificity", np.mean(specificity)),
("precision", np.mean(precision)),
("negative predictive value", np.mean(negative_predictive_value)),
("false positive rate", np.mean(false_positive_rate)),
("false discovery rate", np.mean(false_discovery_rate)),
("false negative rate", np.mean(false_negative_rate)),
("accuracy", np.mean(accuracy)),
("f1 score", np.mean(f1_score))])
    return metrics

def analyze_results(results):
    # takes list of list of results
    svm_linear = []
    svm_rbf = []
    knn = []
    gaussian = []
    logistic = []
    forests = []
    for i in results:
        svm_linear.append(i[0])
        svm_rbf.append(i[1])
        knn.append(i[2])
        gaussian.append(i[3])
        logistic.append(i[4])
        forests.append(i[5])  
    # lists of confusion matrices
    
    all_classifiers = [svm_linear, svm_rbf, knn, gaussian, logistic, forests]
    # list of lists of confusion matrices
    
    final_values = []
    final_keys = ["svm_linear", "svm_rbf", "knn", "gaussian", "logistic", "forests"]    
    
    for classifier in all_classifiers:
        avg_metrics = OrderedDict()
        one_vs_rest = collapse_confusion_matrices(classifier)
        count = 0
        for each_response in one_vs_rest:
            avg_metrics[count] = (calc_average_metrics(basic_counts(each_response)))
            count = count + 1
        final_values.append(avg_metrics)
        
    final_results = OrderedDict(zip(final_keys, final_values))
    return final_results
    
def print_results(final_results):
    for classifier, od_class_scores in final_results.items(): 
        print classifier
        for cls, metrics in od_class_scores.items():
            print cls
            print metrics

def test_data2_cm(n,percent_training_size,FA,ALFF,GM,ALL):
     FA_control,FA_schizophrenia,ALFF_control,ALFF_schizophrenia,GM_control,GM_schizophrenia,FA_ALFF_control,FA_ALFF_schizophrenia,FA_GM_control,FA_GM_schizophrenia,ALFF_GM_control,ALFF_GM_schizophrenia,ALL_control,ALL_schizophrenia = data2_subgroups(FA,ALFF,GM,ALL)
     print_results(analyze_results(run_comparison_n_times(n,FA_control, \
     FA_schizophrenia,percent_training_size)))
#    print_results(analyze_results(run_comparison_n_times(n,ALFF_control,ALFF_schizophrenia,percent_training_size))
#    print_results(analyze_results(run_comparison_n_times(n,GM_control,GM_schizophrenia,percent_training_size))
#    print_results(analyze_results(run_comparison_n_times(n,FA_ALFF_control,FA_ALFF_schizophrenia,percent_training_size))
#    print_results(analyze_results(run_comparison_n_times(n,FA_GM_control,FA_GM_schizophrenia,percent_training_size)))
#    print_results(analyze_results(run_comparison_n_times(n,ALFF_GM_control,ALFF_GM_schizophrenia,percent_training_size)))
#    print_results(analyze_results(run_comparison_n_times(n,ALL_control,ALL_schizophrenia,percent_training_size)))

n = 100
p = 70
c = 50

FA,ALFF,GM,target,ALL = data2_unpack()
FA_pca,ALFF_pca,GM_pca,ALL_pca = data2_pca(FA,
ALFF,GM,target,ALL,c)
test_data2_cm(n,p,FA_pca,ALFF_pca,GM_pca,ALL_pca)

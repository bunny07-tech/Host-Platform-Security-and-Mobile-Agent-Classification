#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from numpy import array
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
'''from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV'''
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection,preprocessing
import warnings
warnings.filterwarnings('ignore')

d =pd.read_csv(r'C:\Users\vinod\Desktop\Data\chamcham.csv',names=['Status','message'],encoding='latin-1')   #opening csv dataset into 'd' and just naming our columns,encoding needed for textual data(not necessary):
X=d["message"]
y=d["Status"]
le=preprocessing.LabelEncoder()         #just a labeling for malicious and non-maliciousApi calls (column 1) as either 0 and 1 (if not done in dataset)
y=le.fit_transform(y)

#preprocessing data:
y=y.astype(int)
for i in range (3,10):  #loop for 'n' gram values
    models_initial=[0];
    print("for n=",i)
    f= TfidfVectorizer(ngram_range=(i,i))    #vectorization of APi call sequences according to n-grams in TF-IDF
    X = f.fit_transform(d["message"])
    row,col=X.shape
    print("Working on",col,"features! wait....")
    
    svd = TruncatedSVD(n_components=100, random_state=42)
    svd.fit(X)
    data = svd.fit_transform(X)
    '''scaler=StandardScaler(with_mean=False)
    scaler.fit(X)
    scaled_data=scaler.transform(X)
    scaled_data=scaled_data.toarray()
    pca=PCA(n_components=250)
    pca.fit(scaled_data)
    x_pca=pca.transform(scaled_data)'''      
    print(data.shape)
            
    #splitting dataset into test and train
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data,y, test_size=0.25)

    #logistic regression
    LR = LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True, penalty='l1', tol=0.0001)
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(LR, X_train, y_train, scoring=scoring, cv=5)
    sorted(scores.keys())
    LR_fit_time = scores['fit_time'].mean()
    LR_score_time = scores['score_time'].mean()
    LR_accuracy = scores['test_accuracy'].mean()
    LR_precision = scores['test_precision_macro'].mean()
    LR_recall = scores['test_recall_macro'].mean()
    LR_f1 = scores['test_f1_weighted'].mean()
    LR_roc = scores['test_roc_auc'].mean()
    print("LR...")

    #decision tree
    decision_tree = DecisionTreeClassifier(criterion='gini',min_samples_split=.10,splitter="best")
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(decision_tree, X_train, y_train, scoring=scoring, cv=5)
    sorted(scores.keys())
    dtree_fit_time = scores['fit_time'].mean()
    dtree_score_time = scores['score_time'].mean()
    dtree_accuracy = scores['test_accuracy'].mean()
    dtree_precision = scores['test_precision_macro'].mean()
    dtree_recall = scores['test_recall_macro'].mean()
    dtree_f1 = scores['test_f1_weighted'].mean()
    dtree_roc = scores['test_roc_auc'].mean()
    print("DT...")

    #SVM
    SVM = SVC(kernel='linear', C=1000, gamma=100,probability = True)
    scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(SVM, X_train, y_train, scoring=scoring, cv=5)
    sorted(scores.keys())
    SVM_fit_time = scores['fit_time'].mean()
    SVM_score_time = scores['score_time'].mean()
    SVM_accuracy = scores['test_accuracy'].mean()
    SVM_precision = scores['test_precision_macro'].mean()
    SVM_recall = scores['test_recall_macro'].mean()
    SVM_f1 = scores['test_f1_weighted'].mean()
    SVM_roc = scores['test_roc_auc'].mean()
    print("SVM...")
    
    '''LDA = LinearDiscriminantAnalysis()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(LDA, X_train.toarray(), y_train, scoring=scoring, cv=5)
    sorted(scores.keys())
    LDA_fit_time = scores['fit_time'].mean()
    LDA_score_time = scores['score_time'].mean()
    LDA_accuracy = scores['test_accuracy'].mean()
    LDA_precision = scores['test_precision_macro'].mean()
    LDA_recall = scores['test_recall_macro'].mean()
    LDA_f1 = scores['test_f1_weighted'].mean()
    LDA_roc = scores['test_roc_auc'].mean()

    
    QDA = QuadraticDiscriminantAnalysis()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(QDA, X_train.toarray(), y_train, scoring=scoring, cv=20)
    sorted(scores.keys())
    QDA_fit_time = scores['fit_time'].mean()
    QDA_score_time = scores['score_time'].mean()
    QDA_accuracy = scores['test_accuracy'].mean()
    QDA_precision = scores['test_precision_macro'].mean()
    QDA_recall = scores['test_recall_macro'].mean()
    QDA_f1 = scores['test_f1_weighted'].mean()
    QDA_roc = scores['test_roc_auc'].mean()'''

    #random forest
    random_forest = RandomForestClassifier(n_estimators=100,random_state =1)
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(random_forest, X_train, y_train, scoring=scoring, cv=5)
    sorted(scores.keys())
    forest_fit_time = scores['fit_time'].mean()
    forest_score_time = scores['score_time'].mean()
    forest_accuracy = scores['test_accuracy'].mean()
    forest_precision = scores['test_precision_macro'].mean()
    forest_recall = scores['test_recall_macro'].mean()
    forest_f1 = scores['test_f1_weighted'].mean()
    forest_roc = scores['test_roc_auc'].mean()
    print("RF...")

    #KNN
    KNN = KNeighborsClassifier()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(KNN, X_train, y_train, scoring=scoring, cv=5)
    sorted(scores.keys())
    KNN_fit_time = scores['fit_time'].mean()
    KNN_score_time = scores['score_time'].mean()
    KNN_accuracy = scores['test_accuracy'].mean()
    KNN_precision = scores['test_precision_macro'].mean()
    KNN_recall = scores['test_recall_macro'].mean()
    KNN_f1 = scores['test_f1_weighted'].mean()
    KNN_roc = scores['test_roc_auc'].mean()
    print("KNN...")

    #NB
    bayes = GaussianNB()
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(bayes, X_train, y_train, scoring=scoring, cv=5)
    sorted(scores.keys())
    bayes_fit_time = scores['fit_time'].mean()
    bayes_score_time = scores['score_time'].mean()
    bayes_accuracy = scores['test_accuracy'].mean()
    bayes_precision = scores['test_precision_macro'].mean()
    bayes_recall = scores['test_recall_macro'].mean()
    bayes_f1 = scores['test_f1_weighted'].mean()
    bayes_roc = scores['test_roc_auc'].mean()
    print("NB...")

    #AdaBoost on decision tree
    bclf = AdaBoostClassifier(base_estimator=decision_tree,n_estimators=50)
    scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
    scores = cross_validate(bclf, X_train, y_train, scoring=scoring, cv=5)
    sorted(scores.keys())
    bclf_fit_time = scores['fit_time'].mean()
    bclf_score_time = scores['score_time'].mean()
    bclf_accuracy = scores['test_accuracy'].mean()
    bclf_precision = scores['test_precision_macro'].mean()
    bclf_recall = scores['test_recall_macro'].mean()
    bclf_f1 = scores['test_f1_weighted'].mean()
    bclf_roc = scores['test_roc_auc'].mean()
    print("AdaB...")

    #results modeling
    models_initial = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Random Forest', 'K-Nearest Neighbors', 'Bayes','AdaBoost(on DT)'],
    'Accuracy'    : [LR_accuracy, dtree_accuracy, SVM_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy,bclf_accuracy],
    'Fitting time': [LR_fit_time, dtree_fit_time, SVM_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time,bclf_fit_time],
    'Scoring time': [LR_score_time, dtree_score_time, SVM_score_time, forest_score_time, KNN_score_time, bayes_score_time,bclf_score_time],
    'Precision'   : [LR_precision, dtree_precision, SVM_precision, forest_precision, KNN_precision, bayes_precision,bclf_precision],
    'Recall'      : [LR_recall, dtree_recall, SVM_recall, forest_recall, KNN_recall, bayes_recall,bclf_recall],
    'F1_score'    : [LR_f1, dtree_f1, SVM_f1, forest_f1, KNN_f1, bayes_f1,bclf_f1],
    'AUC_ROC'     : [LR_roc, dtree_roc, SVM_roc, forest_roc, KNN_roc, bayes_roc,bclf_roc],
    }, columns = ['Model', 'Accuracy','Fitting time', 'Scoring time', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])

    models_initial.sort_values(by='Accuracy', ascending=False)
    print(models_initial)


#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle

reload(sys)
sys.setdefaultencoding('utf8')

#Naive Bayes
def naive_bayes_classifier(train_x,train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x,train_y)
    return model

#kNN demo
def knn_classifier(train_x,train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x,train_y)
    return model

#LR
def logistic_regression_classifier(train_x,train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='12')
    model.fit(train_x,train_y)
    return model
def read_data(data_file):
    import gzip
    f = gzip.open(data_file,"rb")
    train,val,test = pickle.load(f)
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
    data_file = "mnist.pkl.gz"
    threash = 0.5
    model_save_file = "model.dat"
    model_save = {}
    test_classifiers = ['NB','KNN','LR']
    classifiers = {
        'NB' :naive_bayes_classifier,
        'KNN' : knn_classifier,
        'LR' : logistic_regression_classifier,
        }
    print('reading training and test data...')
    train_x,train_y,test_x,test_y = read_data(data_file)
    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    is_binary_class = (len(np.unique(train_y)) == 2)
    print('*********************** Data Info *****************')
    print('#training data:%s, #testing_data, %d, dimension: %d '% (num_train,num_test,num_feat))
    for classifier in test_classifiers:
        print("*************** %s ************************" %classifier)
        start_time = time.time()
        print classifier
        model = classifiers[classifier](train_x,train_y)
        print "train took %fs" % (time.time() - start_time)
        predict=  model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model_save

        if is_binary_class:
             precision = metrics.predicsion_score(test_x,predict)
             recall = metrics.precision_score(test_y,predict)
             print "precision %.2f recall:%.2ff " % (100*precision, 100*recall)
        accuracy = metrics.accuracy_score(test_y,predict)
        print "accuracy: %2.f%%" % (100*accuracy)
    if model_save_file != None:
        pickle.dump(model_save,open(model_save_file,'wb'))
                

        

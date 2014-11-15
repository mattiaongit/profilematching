from Classifier import *
from PreProcessor import *

import sklearn.preprocessing
import pdb

pp = PreProcessor(0)
data, targets = pp.datatargets()

clf = Classifier('SGDClassifier',data, targets)

clf.splitDataTrainingTest(10)

# scaler from sklearn, es: MinMaxScaler, StandardScaler
clf.normalizeData(sklearn.preprocessing.StandardScaler)


tuning_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['f1', 'recall']

#best_estimator = SVMClf1.gridSearch(tuning_parameters, scores)


params = {'C':100, 'cache_size':200, 'class_weight':None, 'coef0':0.0, 'degree':3, 'gamma':0.0001, 'kernel':'linear', 'max_iter':-1, 'probability':False, 'random_state':None, 'shrinking':True, 'tol':0.001, 'verbose':False}

clf.train(params)

clf.test(output = True)

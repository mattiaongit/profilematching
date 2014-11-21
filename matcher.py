from Classifier import *
from PreProcessor import *

import sklearn.preprocessing
import pdb


filterCandidate = ['Google+']
filterPriors = ['YouTube']
filterFeatures = ['distances']#,'humanlimitations','exogenousqwerty','exogenousdvorak']

pp = PreProcessor(minPriors=24, filterFeatures = filterFeatures)
data, targets = pp.datatargets()

clf = Classifier('SGDClassifier',data, targets)

clf.splitDataTrainingTest(10)

# scaler from sklearn, es: MinMaxScaler, StandardScaler
clf.normalizeData(sklearn.preprocessing.StandardScaler)


tuning_parameters = {
    'loss': ('log', 'hinge'),
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [0.001, 0.0001, 0.00001, 0.000001],
    'shuffle': (True, False)
}

scores = ['accuracy']


best_params = clf.gridSearch(tuning_parameters, scores)
#best_params = {'penalty': 'l2', 'alpha': 0.001, 'loss': 'log', 'shuffle': True}
#best_params = {'penalty': 'elasticnet', 'alpha': 0.001, 'loss': 'log'}

#SVM grid
#best_params = {'C':100, 'cache_size':200, 'class_weight':None, 'coef0':0.0, 'degree':3, 'gamma':0.0001, 'kernel':'rbf', 'max_iter':-1, 'probability':False, 'random_state':None, 'shrinking':True, 'tol':0.001, 'verbose':False}


print(best_params)
clf.train(best_params)
clf.test()

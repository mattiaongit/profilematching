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


tuning_parameters = {
    'loss': ('log', 'hinge'),
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [0.001, 0.0001, 0.00001, 0.000001]
}

scores = ['f1', 'recall']


best_params = clf.gridSearch(tuning_parameters, scores)

clf.train(best_params)

clf.test(output = True)

from features import *
from distances import *

from random import shuffle
from itertools import combinations
from SVMClassifier import *
import sklearn.preprocessing

import pymongo

import pdb


#IMPORT DATA FROM DB
connection = pymongo.Connection()
db = connection['alternion']
dbprofiles = db.profiles.find({},{'_id':0})
profilesPairs = dict()
socialNetowrks = set()

# UTILS
def shuffleProfiles(profilePair):
  l1,l2 = zip(*profilePair)
  l = list(l2)
  shuffle(l)
  while sum([el[0] == el[1] for el in zip(l,l2)]) > 0 :
      print('needed another shuffle')
      shuffle(l)
  l2 = tuple(l)
  return  list(zip(l1,l2))


features_functions = [sameUsername, ull, uucl, alphabetDistribution, eachFingerRate, rowsRate,lambda x,y : sameRate(x,y,granularitiesFunction=sameHand),lambda x,y : sameRate(x,y,granularitiesFunction=sameFinger),eachFingerRate(layout='dvorak'), rowsRate(layout='dvorak'),lambda x,y : sameRate(x,y,granularitiesFunction=sameHand,layout='dvorak'),lambda x,y : sameRate(x,y,granularitiesFunction=sameFinger,layout='dvorak'),shannonEntropy,lcsubstring,lcs]

def vectorize(pair, debug = False):
    if debug:
        return [(f.__name__,f(pair[0],[pair[1]])) for f in features_functions]
    else:
        features_vector = []
        [features_vector.extend(f(pair[0],[pair[1]])) for f in features_functions]
        return features_vector

for profile in list(dbprofiles):
  profilePairs = list(combinations(profile.items(),2))
  for pair in profilePairs:
    sn1 = pair[0][0].capitalize()
    sn2 = pair[1][0].capitalize()
    socialNetowrks.update(set((sn1,sn2)))
    key = tuple(sorted((sn1,sn2)))
    if key not in profilesPairs.keys():
        profilesPairs[key] = []
    profilesPairs[key].append((pair[0][1]['username'],pair[1][1]['username']))


# Extract pairs of a specific social networks pair
dataset = profilesPairs[('Facebook','Twitter')]
dataset = [pair for pair in dataset if len(pair[0]) > 0 and len(pair[1]) > 0]
raw_data = dataset + shuffleProfiles(dataset)
# LABELS OF DATASET (1: positive match, 0: negative match)
# THE SHUFFLED USERNAME PAIRS IS GOING TO BE LABELLED AS 0
targets = [1] * len(dataset) + [0] * len(dataset)

data = []
# BUILDING FEATURES INPUT VECTOR
for sample in raw_data:
  data.append(vectorize(sample))


SVMClf1 = SVMClassifier(data, targets)

SVMClf1.splitDataTrainingTest(2)

# scaler from sklearn, es: MinMaxScaler, StandardScaler
SVMClf1.normalizeData(sklearn.preprocessing.StandardScaler)


tuning_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['f1', 'recall']

#SVMClf1.gridSearch(tuning_parameters, scores)


params = {'C':10, 'cache_size':200, 'class_weight':None, 'coef0':0.0, 'degree':3, 'gamma':0.0001, 'kernel':'linear', 'max_iter':-1, 'probability':False, 'random_state':None, 'shrinking':True, 'tol':0.001, 'verbose':False}
SVMClf1.train(params)

SVMClf1.test(output = True)

from features import *
from distances import *

from random import shuffle
from itertools import combinations
from SVMClassifier import *

import pymongo


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
  l2 = tuple(l)
  return  list(zip(l1,l2))


features_functions = [ull, uucl, alphabetDistribution, eachFingerRate, rowsRate,lambda x : sameRate(x,granularitiesFunction=sameHand),lambda x : sameRate(x,granularitiesFunction=sameFinger),levenshtein,jaccard,shannonEntropy,lcsubstring,lcs]

def vectorize(pair, debug = False):
    if debug:
        return [(f.__name__,f(pair)) for f in features_functions]
    else:
        features_vector = []
        [features_vector.extend(f(pair)) for f in features_functions]
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


SVMClf = SVMClassifier(data, targets)

SVMClf.splitDataTrainingTest(2)

params = {'C'=100,'verbose'=False}
SVMClf.train(params)

SVMClf.test(output = True)

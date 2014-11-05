from features import *
from distances import *

from random import shuffle
from itertools import combinations

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
        return [(f.__name__,f(pair)) for f in features_fucntions]
    else:
        features_vector = []
        [features_vector.extend(f(pair)) for f in features_fucntions]
        return features_vector

vectorize(['mattia','mattiadmr'])

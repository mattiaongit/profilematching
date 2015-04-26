from sklearn.externals import joblib
from features import *
from Classifier import *
from PreProcessor import *


all_features = {
    'humanlimitations': [sameUsername, ull, uucl],
    'exogenousqwerty': [lambda x,y: sameRate(x,y,granularitiesFunction=sameHand),lambda x,y : sameRate(x,y,granularitiesFunction=sameFinger),eachFingerRate,rowsRate],
    'exogenousdvorak': [lambda x,y: sameRate(x,y,granularitiesFunction=sameHand,layout='dvorak'),lambda x,y : sameRate(x,y,granularitiesFunction=sameFinger,layout='dvorak'),lambda x,y: eachFingerRate(x,y,layout='dvorak'), lambda x,y: rowsRate(x,y,layout='dvorak')],
    'endogenous': [alphabetDistribution,shannonEntropy,lcsubstring,lcs],
    'distances' : [levenshtein, jaccard]
}

features = []

for feature in all_features.keys():
    features.extend(all_features[feature])


def load_classifier(model_file):
    return joblib.load(model_file)


def vectorize(pair, debug = False):
    if debug:
        print [(f.__name__,len(f(pair[0],pair[1]))) for f in features]
    features_vector = []
    [features_vector.extend(f(pair[0],pair[1])) for f in features]
    return features_vector

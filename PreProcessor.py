'''Transform raw data from dataset to a processable data format for classificators
   'datatargets' method provide input vector data and data target  '''

import dataset
from features import *
from distances import *
from random import shuffle, sample
from itertools import combinations

import sklearn.decomposition

import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pdb



class PreProcessor():

	# priors 0 returns all possible priors for each candidate
    def __init__(self, filterCandidate = False, filterPriors = False, minPriors = 1, filterFeatures = False):
        self.rawdata = dataset.raw_data()

        self.filterCandidate = filterCandidate
        self.filterPriors = filterPriors
        self.minPriors = minPriors

        self.filterFeatures = filterFeatures

        self.features = {
            'humanlimitations': [sameUsername, ull, uucl],
            'exogenousqwerty': [lambda x,y: sameRate(x,y,granularitiesFunction=sameHand),lambda x,y : sameRate(x,y,granularitiesFunction=sameFinger),eachFingerRate,rowsRate],
            'exogenousdvorak': [lambda x,y: sameRate(x,y,granularitiesFunction=sameHand,layout='dvorak'),lambda x,y : sameRate(x,y,granularitiesFunction=sameFinger,layout='dvorak'),lambda x,y: eachFingerRate(x,y,layout='dvorak'), lambda x,y: rowsRate(x,y,layout='dvorak')],
            'endogenous': [alphabetDistribution,shannonEntropy,lcsubstring,lcs],
            'distances' : [levenshtein, jaccard]
        }

        self.selected_features = []
        self.ppdata = []
        self.data = []
        self.targets = []

    def preprocess(self):
        print('Preprocessing data, as requested:')
        print('Candidate filter: {0}'.format(self.filterCandidate))
        print('Priors filter: {0}'.format(self.filterPriors))
        print('Min number of priors: {0}'.format(self.minPriors))
        profiles = []
        for profile in list(self.rawdata):
            for cKey,cUsername in profile.items():
                if not self.filterCandidate or cKey in self.filterCandidate:
                    profiles.append((cUsername['username'], [pUsername['username'] for pKey,pUsername in profile.items() if cKey != pKey and ( not self.filterPriors or pKey in self.filterPriors)]))

        profiles = [x for x in profiles if len(x[1]) >= self.minPriors and len(x[0]) > 0 and all([len(u) > 0 for u in x[1]])]
        candidates, priors = zip(*profiles)
        tmp = list(candidates)
        shuffle(tmp)
        candidates = tuple(tmp)
        shuffledProfiles = zip(candidates,priors)
        self.ppdata = profiles + shuffledProfiles
        self.targets = [1] * len(profiles) + [0] * len(profiles)
        print("Samples ready to extract features, n samples: {0}".format(len(self.ppdata)))

    def vectorize(self, pair, debug = False):
        if debug:
            return [(f.__name__,f(pair[0],pair[1])) for f in self.selected_features]
        else:
            features_vector = []
            [features_vector.extend(f(pair[0],pair[1])) for f in self.selected_features]
            return features_vector


    def vectorizeData(self,timer = False, debug = False):
        for feature in self.features.keys():
            if not self.filterFeatures or feature in self.filterFeatures:
                self.selected_features.extend(self.features[feature])

        print("Extracting features")
        print('Features used: {0} - Classes: {1}'.format(len(self.selected_features),(not self.filterFeatures and 'All') or self.filterFeatures))
        counter = 0
        #self.data = map(self.vectorize, sample)
        for sample in self.ppdata:
            counter += 1
            self.data.append(self.vectorize(sample))
            if debug and counter % (len(self.ppdata)/10) == 0:
                print("{0}0% done ... ({1}/{2}) samples".format(counter/(len(self.ppdata)/10),counter,len(self.ppdata)))
                #print(self.vectorize(sample,debug = True))

    def pca(self, ncomponents, plot = False):
        print "pca!"
        pca = sklearn.decomposition.PCA(n_components=ncomponents)
        pca.fit(self.data)
        self.data = pca.transform(self.data)

        if plot and n_components == 3:
            np.random.seed(5)

            centers = [[1, 1], [-1, -1], [1, -1]]
            X = self.data
            y = self.targets

            fig = plt.figure(1, figsize=(8, 6))
            plt.clf()
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=18, azim=54)

            plt.cla()

            for name, label in [('Sameuser', 0), ('Different', 1)]:
                ax.text3D(X[y == label, 0].mean(),
                          X[y == label, 1].mean() + 1.5,
                          X[y == label, 2].mean(), name,
                          horizontalalignment='center',
                          bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
            # Reorder the labels to have colors matching the cluster results
            y = np.choose(y, [1, 2, 0]).astype(np.float)
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)

            x_surf = [X[:, 0].min(), X[:, 0].max(),
                      X[:, 0].min(), X[:, 0].max()]
            y_surf = [X[:, 0].max(), X[:, 0].max(),
                      X[:, 0].min(), X[:, 0].min()]
            x_surf = np.array(x_surf)
            y_surf = np.array(y_surf)
            v0 = pca.transform(pca.components_[0])
            v0 /= v0[-1]
            v1 = pca.transform(pca.components_[1])
            v1 /= v1[-1]

            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])

            plt.savefig("plot2D.png")


    def datatargets(self):
        self.preprocess()
        self.vectorizeData(debug=True)
        self.pca(20)
        return self.data, self.targets

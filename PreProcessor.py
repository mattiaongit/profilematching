'''Transform raw data from dataset to a processable data format for classificators
   Returning input vector data and data target  '''

import dataset
from features import *
from distances import *
from random import shuffle, sample
from itertools import combinations

import pdb



class PreProcessor():

	# priors 0 returns all possible priors for each candidate
	def __init__(self, filterCandidate = False, filterPriors = False, minPriors = 1):
		self.rawdata = dataset.raw_data()

		self.filterCandidate = filterCandidate
		self.filterPriors = filterPriors
		self.minPriors = minPriors

		self.features = {
			'humanlimitations': [sameUsername, ull, uucl],
			'exogenous': {
				'qwerty': [lambda x,y: sameRate(x,y,granularitiesFunction=sameHand),lambda x,y : sameRate(x,y,granularitiesFunction=sameFinger),eachFingerRate,rowsRate],
				'dvorak': [lambda x,y: sameRate(x,y,granularitiesFunction=sameHand,layout='dvorak'),lambda x,y : sameRate(x,y,granularitiesFunction=sameFinger,layout='dvorak'),lambda x,y: eachFingerRate(x,y,layout='dvorak'), lambda x,y: rowsRate(x,y,layout='dvorak')],
				},
			'endogenous': [alphabetDistribution,shannonEntropy,lcsubstring,lcs],
			'distances' : [jaccard]
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
		print("Raw data is ready to extract features, n items:{0}".format(len(self.ppdata)))

	def vectorize(self, pair, debug = False):
		if debug:
			return [(f.__name__,f(pair[0],pair[1])) for f in self.selected_features]
		else:
			features_vector = []
			[features_vector.extend(f(pair[0],pair[1])) for f in self.selected_features]
			return features_vector


	def vectorizeData(self,timer = False, debug = False):
		self.selected_features = self.features['distances'] #+ self.features['humanlimitations']
		#self.features['humanlimitations'] + self.features['exogenous']['qwerty'] + self.features['exogenous']['dvorak'] + self.features['endogenous'] +
		counter = 0
		#self.data = map(self.vectorize, sample)
		for sample in self.ppdata:
			counter += 1
			self.data.append(self.vectorize(sample))
			if debug and counter % 1000 == 0:
				print(sample)
				print(self.vectorize(sample,debug = True))

	def datatargets(self):
		self.preprocess()
		self.vectorizeData()
		return self.data, self.targets

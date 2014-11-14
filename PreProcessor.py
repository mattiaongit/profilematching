'''Transform raw data from dataset to a processable data format for classificators
   Returning input vector data and data target  '''

import dataset
from features import *
from random import shuffle
from itertools import combinations



class PreProcessor():

	# priors 0 returns all possible priors for each candidate
	def __init__(self, priors = 0, priorsDistribution=True):
		self.priors = priors
		self.pdistribution = priorsDistribution
		self.rawdata = dataset.raw_data()

		self.features = {
			'humanlimitations': [sameUsername, ull, uucl],
			'exogenous': {
				'qwerty': [lambda x,y: sameRate(x,y,granularitiesFunction=sameHand),lambda x,y : sameRate(x,y,granularitiesFunction=sameFinger),eachFingerRate,rowsRate],
				'dvorak': [lambda x,y: sameRate(x,y,granularitiesFunction=sameHand,layout='dvorak'),lambda x,y : sameRate(x,y,granularitiesFunction=sameFinger,layout='dvorak'),lambda x,y: eachFingerRate(x,y,layout='dvorak'), lambda x,y: rowsRate(x,y,layout='dvorak')],
				},
			'endogenous': [alphabetDistribution,shannonEntropy,lcsubstring,lcs]
		}

		self.selected_features = []

		self.ppdata = []
		self.data = []
		self.targets = []



	def preprocess(self):
		print('Preprocessing data')
		if self.priors == 0:
			profiles =[(profile['Alternion']['username'],[v['username'] for k,v in profile.items() if k != 'Alternion' and len(v['username']) > 0]) for profile in list(self.rawdata)]
			profiles = [x for x in profiles if len(x[1]) > 0]
			candidates, priors = zip(*profiles)
			tmp = list(candidates)
			shuffle(tmp)
			candidate = tuple(tmp)
			shuffledProfiles = zip(candidates,priors)
			self.ppdata = profiles + shuffledProfiles
			self.targets = [1] * len(profiles) + [0] * len(profiles)
			print("Raw data is ready to extract features, n items:{0}".format(len(self.ppdata)))

	def vectorize(self, pair, debug = False):
		if debug:
			return [(f.__name__,f(pair[0],pair[1])) for f in self.selecte_features]
		else:
			features_vector = []
			[features_vector.extend(f(pair[0],pair[1])) for f in self.selected_features]
			return features_vector


	def vectorizeData(self,timer = False):
		self.selected_features = self.features['humanlimitations'] +
				   self.features['exogenous']['qwerty'] +
				   self.features['endogenous']

		for sample in self.ppdata:
			self.data.append(vectorize(sample))

	def datatargets(self):
		self.preprocess()
		self.vectorizeData()
		return {'data': self.data, 'targets': self.targets}

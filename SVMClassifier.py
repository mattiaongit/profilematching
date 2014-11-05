from sklearn import cross_validation, grid_search, metrics, preprocessing, svm

class SVMClassifier:

  def __init__(self, data, targets):
    #Filling instance variables with datafunction
    self.data = data
    self.targets = targets
    self.X_train, self.X_test, self.y_train, self.y_test = [],[],[],[]

    # Classifier
    self.clf = []


  def splitDataTrainingTest(fraction):
    #Split training set from data
    self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split( self.data, self.targets, test_size = len(self.data)/fraction , random_state = 0 )


  def normalizeData(axis,type):
    # self.X_train = preprocessing.scale(X_train)
    # self.X_test = preprocessing.scale(X_test)
    pass

  def train(params):
    self.clf = svm.SVC(**params)
    self.clf.fit(self.X_train,self.X_test)


  def score(scoring, tuned_parameters):
    pass

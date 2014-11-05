from sklearn import cross_validation, grid_search, metrics, preprocessing, svm

class SVMClassifier:

  def __init__(self, data, targets):
    #Filling instance variables with datafunction
    self.data = data
    self.targets = targets
    self.X_train, self.X_test, self.y_train, self.y_test = [],[],[],[]

    # Classifier
    self.clf = []


  def splitDataTrainingTest(self,fraction):
    #Split training set from data
    self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split( self.data, self.targets, test_size = len(self.data)/fraction , random_state = 0 )


  def normalizeData(self,scaler_function):
    scaler = preprocessing.__dict__[scaler_function.__name__](copy=False)
    scaler.fit_transform(self.X_train)
    scaler.fit_transform(self.X_test)

  def train(self,params):
    print("training ...")
    self.clf = svm.SVC(**params)
    self.clf.fit(self.X_train,self.y_train)


  def test(self, output = False):
    y_pred = []
    #TEST
    print("predictions ...")
    for t in self.X_test:
      y_pred.append(self.clf.predict(t))

    # OUTPUT CLASSIFICATOR SCORES
    # Consider to move this metrics on self
    precision = metrics.precision_score(self.y_test,y_pred)
    recall = metrics.recall_score(self.y_test,y_pred)
    f1 = 2 * ( (precision * recall) / (precision + recall) )
    if output:
      print("prec: {0}, rec: {1}, f1: {2}".format(precision,recall,f1))

  def score(self,scoring, tuned_parameters):
    pass

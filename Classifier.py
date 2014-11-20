from sklearn import cross_validation, grid_search, metrics, preprocessing, svm, linear_model


class Classifier:

  def __init__(self, learning_model, data, targets):
    self.model = learning_model
    self.hparams = None
    self.clf = None

    self.data = data
    self.targets = targets
    self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4




  def splitDataTrainingTest(self,fraction):
    #Split training set from data
    self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split( self.data, self.targets, test_size = len(self.data)/fraction , random_state = 0 )
    print(self.y_train)
    print(self.y_test)


  def normalizeData(self,scaler_function):
    scaler = preprocessing.__dict__[scaler_function.__name__](copy=False)
    scaler.fit_transform(self.X_train)
    scaler.fit_transform(self.X_test)


  def gridSearch(self,tuning_parameters,scores):
    bests = {}
    for score in scores:
        print("# Tuning hyper-parameters for %s \r\n" % score)
        clf = grid_search.GridSearchCV(linear_model.__dict__[self.model](), tuning_parameters, scoring=score)
        clf.fit(self.X_train, self.y_train)
        print("Best parameters set found on development set:\r\n")
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:\r\n")
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
        print()
        print("Detailed classification report:\r\n")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.\r\n")
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        print(metrics.classification_report(y_true, y_pred))
        bests[clf.best_score_] = clf.best_params_
    return bests[max(bests.keys())]

  def train(self, hyperparams = {}):
    print("training ...")
    self.hparams = hyperparams
    self.clf = linear_model.__dict__[self.model](**self.hparams)

    print("Classifier model {0}".format(self.model))
    print("With hyperparms\r\n: {0}".format(self.hparams))

    self.clf.fit(self.X_train,self.y_train)
    print("trained, classifer internal status:")
    print("Features vector length {0}".format(len(self.clf.coef_[0])))
    print(self.clf.coef_)


  def test(self):
    y_pred = []
    #TEST
    #TODO to list comprehension
    print("predictions ...")
    for t in self.X_test:
      y_pred.append(self.clf.predict(t))

    # OUTPUT CLASSIFICATOR SCORES
    # Consider to move this metrics on self
    precision = metrics.precision_score(self.y_test,y_pred)
    recall = metrics.recall_score(self.y_test,y_pred)
    f1 = 2 * ( (precision * recall) / (precision + recall) )
    acc = metrics.accuracy_score(self.y_test,y_pred)

    fpr, tpr, tresholds = metrics.roc_curve(self.y_test,y_pred, pos_label = 1)
    auc = metrics.auc(fpr,tpr)
    print("accuracy: {0}, prec: {1}, rec: {2}, f1: {3}, AUC: {4}".format(acc,precision,recall,f1,auc))

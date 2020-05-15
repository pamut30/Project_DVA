import csv
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.svm import SVC

class SVM_model():
    def __init__(self,train,test,flatten=True,prob=True):
        self.X_train=train.values[:,:-1]
        self.y_train=train.values[:,-1]
        self.X_test = test.values[:,:-1]
        self.y_test = test.values[:,-1]
        self.best_model = None
        self.best_param = None
        self.test_score = None
        self.flatten = flatten
        self.prob = prob

    def optimize_model(self,params):
        model = SVC(random_state=123,probability=self.prob,gamma='scale',shrinking=True,degree=2)
        gscv = GridSearchCV(model, params)
        gscv.fit(self.X_train, self.y_train)
        self.best_model = gscv.best_estimator_
        self.best_param = gscv.best_params_

    def get_test_score(self):
        if self.flatten:
            self.y_pred = np.where(self.best_model.predict(self.X_test).flatten() < 0.5,0,1)
        else:
            self.y_pred = self.best_model.predict(self.X_test)
        self.y_true = self.y_test.astype(int)
        self.test_score = {'accuracy':accuracy_score(self.y_true,self.y_pred),
                           'precision':precision_score(self.y_true,self.y_pred),
                           'recall':recall_score(self.y_true,self.y_pred),
                           'f1_score':f1_score(self.y_true,self.y_pred)}

    def save(self, filename):
        with open(f'{filename}.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(self.y_pred)

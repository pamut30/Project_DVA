import numpy as np
import csv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier

class KNN_model():
    def __init__(self,train,test,flatten=True):
        self.X_train=train.values[:,:-1]
        self.y_train=train.values[:,-1]
        self.X_test = test.values[:,:-1]
        self.y_test = test.values[:,-1]
        self.best_model = None
        self.best_param = None
        self.test_score = None
        self.flatten = flatten

    def optimize_model(self,params):
        model = KNeighborsClassifier()
        gscv = GridSearchCV(model, params)
        gscv.fit(self.X_train, self.y_train)
        self.best_model = gscv.best_estimator_
        self.best_param = gscv.best_params_

    def get_test_score(self):
        self.y_pred = np.where(self.best_model.predict(self.X_test).flatten() < 0.5, 0, 1)
        self.y_true = self.y_test.astype(int)
        self.test_score = {'accuracy':accuracy_score(self.y_true,self.y_pred),
                           'precision':precision_score(self.y_true,self.y_pred),
                           'recall':recall_score(self.y_true,self.y_pred),
                           'f1_score':f1_score(self.y_true,self.y_pred)}

    def save(self, filename):
        with open(f'{filename}.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(self.y_pred)
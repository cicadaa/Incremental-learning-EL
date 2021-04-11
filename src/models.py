import pickle
import logging
import os
import os.path

from sklearn.svm import SVR
from sklearn import preprocessing as pre
from sklearn.linear_model import SGDRegressor


class SVRModel:

    def __init__(self, modelPath, updateStatus=False, acceptable=True, version=0, kernel='rbf', C=10, gamma=0.04, epsilon=.01):
        self.modelPath = modelPath
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.scaler = pre.StandardScaler()

    def learn(self, X, y):
        self.scaler = self.scaler.partial_fit(X)
        XTrans = self.scaler.transform(X)
        self.model.fit(XTrans, y)

    def predict(self, X):
        self.scaler = self.scaler.partial_fit(X)
        XTrans = self.scaler.transform(X)
        return self.model.predict(XTrans)


class OSVRModel:

    def __init__(self, learning_rate='constant', eta0=0.4, loss='epsilon_insensitive', penalty='l2'):
        self.model = SGDRegressor(
            learning_rate=learning_rate, eta0=eta0, loss=loss, penalty=penalty)
        self.scaler = pre.StandardScaler()

    def learn(self, X, y):
        self.scaler = self.scaler.partial_fit(X)
        XTrans = self.scaler.transform(X)
        self.model.partial_fit(XTrans, y)

    def predict(self, X):
        self.scaler = self.scaler.partial_fit(X)
        XTrans = self.scaler.transform(X)
        return self.model.predict(XTrans)

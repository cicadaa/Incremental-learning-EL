import pickle
import logging
from sklearn.svm import SVR
from sklearn import preprocessing as pre


class SVRModel:

    def __init__(self, modelPath, update=False, acceptable=True, version=0, kernel='rbf', C=100, gamma=0.04, epsilon=.01):
        self.modelPath = modelPath
        self.update = update
        self.acceptable = acceptable
        self.version = version

        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.scaler = pre.StandardScaler()

    def save(self):
        modelFilePath = self.modelPath + \
            'svr' + str(self.version) + '.pkl'
        with open(modelFilePath, 'wb') as file:
            pickle.dump(self.model, file)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.version += 1
        self.acceptable = True
        self.update = True
        self.save()

    # def update(self):

    def predict(self, X):
        XTrans = self.scaler.transform(X)  # normalize input
        return self.model.predict(XTrans)

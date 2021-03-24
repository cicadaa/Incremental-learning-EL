from sklearn.svm import SVR
import numpy as np
import pickle


class Simulator:
    def __init__(self):
        self.model = SVR(kernel='rbf', C=10, gamma=0.04, epsilon=.01)

    def getModel(self, modelPath):
        with open(modelPath, 'rb') as f:
            model = pickle.load(f)
        return model

    def saveModel(self, model, modelPath):
        with open(modelPath, 'wb') as f:
            pickle.dump(model, f)

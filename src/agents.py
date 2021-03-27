from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np


class Evaluator:
    def __init__(self, method, basescore):
        self.baseline = 0.5
        self.method = method
        self.basescore = basescore

    def evaluate(self, prediction, actual):
        if self.method == 'r2':
            score = r2_score(actual, prediction)
            if score > self.basescore*0.9:
                return True, score
            else:
                return False, score


class Dataset:
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def getTrainData(self, XSource, ySource, timeIdx, window, scaler):

        X = XSource[timeIdx - window: timeIdx]
        y = ySource[timeIdx - window: timeIdx]

        # XTrain, XTest, yTrain, yTest = [np.array(item) for item in train_test_split(
        #     X, y, test_size=0.01, random_state=42)]
        XTrain = X
        yTrain = y
        XTrain = scaler.transform(XTrain)
        # XTest = scaler.transform(XTest)

        return XTrain, yTrain

    def getSourceData(self, columns, prevFrom, prevTo):
        df = self.data
        for i in range(prevFrom, prevTo+1):
            for c in columns:
                df['prev_' + c + str(i)] = df[c].shift(periods=i)
                df['pre_' + c + str(i)] = df[c].shift(periods=i)
        df = df.dropna()
        y = df['meter']
        times = df['datetime']

        del df['datetime']
        del df['meter']
        del df['temp']

        X = df
        return X, y, times

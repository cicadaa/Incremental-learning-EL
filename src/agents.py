
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, path):
        # TODO: check if the path is valid
        self.data = pd.read_csv(path)

    def getTrainData(self, XSource, ySource, idxFrom, idxTo, scaler):
        XTrain = XSource[idxFrom: idxTo].copy()
        yTrain = ySource[idxFrom: idxTo].copy()
        XTrain = scaler.transform(XTrain)
        return XTrain, yTrain

    def getSourceData(self, columns, prevFrom, prevTo):
        df = self.data
        for i in range(prevFrom, prevTo+1):
            for c in columns:
                df['prev_' + c + str(i)] = df[c].shift(periods=i)
                df['pre_' + c + str(i)] = df[c].shift(periods=i)
        df = df.dropna()

        y = df['meter'].copy()
        times = df['datetime'].copy()

        del df['datetime']
        del df['meter']
        del df['temp']

        X = df.copy()
        return X, y, times

import logging
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing as pre
from torch.autograd import Variable


class Dataset:
    def __init__(self, dataPath, shiftFeatures, shiftRange, isTorch=False, removeSet=set(['index', 'datetime', 'meter', 'temp'])):
        self.isTorch = isTorch
        self.dataPath = dataPath
        self.shiftFeatures = shiftFeatures
        self.shiftRange = shiftRange
        self.removeSet = removeSet
        self.data = self._initData()
        self.scaler = pre.StandardScaler()
        self._X, self._y, self._times = self._splitData(self.data)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        if type(value) is not pd.DataFrame:
            raise ValueError('X should be dataframe')
        self._X = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if type(value) is not pd.Series:
            raise ValueError('y should be Series')
        self._y = value

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, value):
        if type(value) is not pd.Series:
            raise ValueError('times should be Series')
        self._times = value

    def _initData(self):
        df = self._readData(self.dataPath)
        df = self._shiftData(df=df)

        return df[self.shiftRange[0]:].reset_index().copy()

    def _shiftData(self, df):
        for i in range(self.shiftRange[0], self.shiftRange[1] + 1):
            for f in self.shiftFeatures:
                df['prev_' + f + str(i)] = df[f].shift(periods=i)
        return df.dropna().copy()

    def _splitData(self, df):
        features = [e for e in list(df.columns) if e not in self.removeSet]
        return df[features].copy(), df['meter'].copy(), df['datetime'].copy()

    def _readData(self, dataPath):
        try:
            df = pd.read_csv(dataPath)
        except Exception as e:
            logging.error(e)
            raise e
        return df.dropna()

    def getTrainData(self, idxFrom, idxTo):

        X = np.array(self.X[idxFrom: idxTo].copy())
        y = np.array(self.y[idxFrom: idxTo].copy())

        self.scaler = self.scaler.partial_fit(X)
        XTrans = self.scaler.transform(X)

        if self.isTorch:
            return Variable(torch.Tensor([XTrans])),  Variable(torch.Tensor([y]))
        return XTrans, y

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
        self.inputLength = None
        self.shiftRange = shiftRange
        self.removeSet = removeSet
        self.data, self._y, self._times = self._initData()
        self.scaler = pre.MinMaxScaler()
        
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if type(value) is not pd.Series:
            raise ValueError('times should be Series')
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
        df = df.sort_values(by=['datetime'])
        df = self._shiftData(df=df)
        df = df[self.shiftRange[0]:].reset_index()
        return self._splitData(df)

    def _shiftData(self, df):
        for i in range(self.shiftRange[0], self.shiftRange[1] + 1):
            for f in self.shiftFeatures:
                df['prev_' + f + str(i)] = df[f].shift(periods=i)
        return df.dropna().copy()

    def _splitData(self, df):
        features = [e for e in list(df.columns) if e not in self.removeSet] +['meter']
        self.inputLength = len(features) - 1
        return df[features].copy(), df['meter'].copy(), df['datetime'].copy()


    def _readData(self, dataPath):
        try:
            df = pd.read_csv(dataPath)
        except Exception as e:
            logging.error(e)
            raise e
        return df.dropna()

    def getTrainData(self, idxFrom, idxTo):
        
        trainData = self.data.iloc[idxFrom:idxTo, :self.inputLength+1].copy()
        trainData = np.array(trainData)[0]

        self.scaler = self.scaler.partial_fit(trainData[:, np.newaxis])
        trainData = self.scaler.transform(trainData[:, np.newaxis])

        X = np.array(trainData[:self.inputLength].copy())
        y = np.array(trainData[self.inputLength:].copy())
 
        if self.isTorch:
            return Variable(torch.Tensor([X])),  Variable(torch.Tensor(y))
        return X, y

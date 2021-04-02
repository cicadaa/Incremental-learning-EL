import pandas as pd
import logging
import numpy as np


class Dataset:
    def __init__(self, dataPath, features, shiftRange, removeSet=set(['datetime', 'meter', 'temp'])):
        self.dataPath = dataPath
        self.features = features
        self.shiftRange = shiftRange
        self.removeSet = removeSet
        self.data = self._initData()
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
        if type(value) is not pd.DataFrame:
            raise ValueError('y should be dataframe')
        self._y = value

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, value):
        if type(value) is not pd.DataFrame:
            raise ValueError('times should be dataframe')
        self._times = value

    def _initData(self):
        df = self._readData(self.dataPath)
        df = self._shiftData(df=df, features=self.features,
                             shiftFrom=self.shiftRange[0], shiftTo=self.shiftRange[1])

        return df[self.shiftRange[0]:].reset_index().copy()

    def _shiftData(self, df, features, shiftFrom, shiftTo):
        for i in range(shiftFrom, shiftTo + 1):
            for f in features:
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
        return np.array(self.X[idxFrom: idxTo].copy()), np.array(self.y[idxFrom: idxTo].copy())

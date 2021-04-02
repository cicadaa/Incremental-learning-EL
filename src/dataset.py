import pandas as pd
import logging
import numpy as np


class Dataset:
    def __init__(self, dataPath, features, shiftRange, removeSet=set(['datetime', 'meter', 'temp'])):
        self.dataPath = dataPath
        self.features = features
        self.shiftRange = shiftRange
        self.removeSet = removeSet
        self.data = self._getData()
        self.X, self.y, self.times = self._splitData(self.data)

    def _getData(self):
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
        removeSet = set(['datetime', 'meter', 'temp'])
        features = [e for e in list(df.columns) if e not in removeSet]
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

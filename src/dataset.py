import logging
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing as pre
from torch.autograd import Variable


class Dataset:
    def __init__(self, dataPath, categoryFeatures, shiftFeatures, shiftRange, isTorch=False, removeFeatures=set(['index', 'datetime', 'meter', 'temp', 'Unnamed: 0'])):
        self.isTorch = isTorch
        self.dataPath = dataPath
        self.shiftFeatures = shiftFeatures
        self.categoryFeatures = categoryFeatures
        self.scaleFeatureLength = None
        self.shiftRange = shiftRange
        self.removeFeatures = removeFeatures
        self.scaleData, self.nonscaleData, self._y, self._times = self._initData()
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
        allFeatures = [f for f in list(df.columns) if f not in self.removeFeatures]
        scaleFeatures = [f for f in allFeatures if f not in self.categoryFeatures]+['meter']
        self.scaleFeatureLength = len(scaleFeatures)-1
        nonscaleFeatures = [f for f in allFeatures if f in self.categoryFeatures]
        return df[scaleFeatures].copy(), df[nonscaleFeatures].copy(), df['meter'].copy(), df['datetime'].copy()


    def _readData(self, dataPath):
        try:
            df = pd.read_csv(dataPath)
        except Exception as e:
            logging.error(e)
            raise e
        return df.dropna()

    def getTrainData(self, idxFrom, idxTo):
        
        scaleData = self.scaleData.iloc[idxFrom:idxTo,:self.scaleFeatureLength+1].copy()
        scaleData = np.array(scaleData)[0]

        nonscaleData = self.nonscaleData.iloc[idxFrom:idxTo,:].copy().to_numpy()
        nonscaleData =np.reshape(nonscaleData, (len(self.nonscaleData.columns), 1))
  
        self.scaler = self.scaler.partial_fit(scaleData[:, np.newaxis])
        trainData = self.scaler.transform(scaleData[:, np.newaxis])
        
        Feature = np.array(trainData[:self.scaleFeatureLength].copy())
        Feature = np.concatenate((Feature, nonscaleData), axis=0)
        Feature = np.reshape(Feature, (1,1,41))
        # print(Feature)
 
        y = np.array(trainData[self.scaleFeatureLength:].copy())
    
        if self.isTorch:
            return Variable(torch.Tensor(Feature)),  Variable(torch.Tensor(y))
        return Feature, y

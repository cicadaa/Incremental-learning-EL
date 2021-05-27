import logging
from os import PRIO_PGRP
import numpy as np
from numpy.core.fromnumeric import size
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
        print('features', scaleFeatures)
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


    def _getPairData(self, idxFrom, idxTo):
        scaleData = self.scaleData.iloc[idxFrom:idxTo,:len(self.scaleData.columns)].copy()
        scaleData = np.array(scaleData)[0]
        nonscaleData = self.nonscaleData.iloc[idxFrom:idxTo,:].copy().to_numpy()
     
        self.scaler = self.scaler.partial_fit(scaleData[:, np.newaxis])
        scaleData = self.scaler.transform(scaleData[:, np.newaxis])

        scaleData = np.array(scaleData[:len(self.scaleData.columns)-1].copy())
        nonscaleData =np.reshape(nonscaleData, (len(self.nonscaleData.columns), 1))
        
        input = np.concatenate((scaleData, nonscaleData), axis=0)
        # trainData = np.reshape(scaleData, (1,30)) #30 features
        y = scaleData[-1:].copy()[0]
        
        if self.isTorch:
            return input, y
   
    def getTrainData(self, idxFrom, idxTo):   
        inputs = []
        ys = []
        for i in range(idxFrom, idxTo):
            input, y = self._getPairData(i,i+1)
            inputs.append(input)
            ys.append(y)
        if self.isTorch:
            return Variable(torch.Tensor(inputs)),  Variable(torch.Tensor(ys))
        return inputs, ys

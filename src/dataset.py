import logging
from re import S
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import torch
from sklearn import preprocessing as pre
from torch.autograd import Variable


class Dataset:
    def __init__(self, dataPath, categoryFeatures, shiftFeatures, shiftRange, isTorch=False, useTimeFeature=False, removeFeatures=set(['index', 'datetime', 'meter', 'temp', 'Unnamed: 0'])):
        
        self.dataPath = dataPath
        self.shiftFeatures = shiftFeatures
        self.categoryFeatures = categoryFeatures
        self.scaleFeatureLength = None
        self.shiftRange = shiftRange
        self.removeFeatures = removeFeatures

        self.isTorch = isTorch
        self.useTimeFeature = useTimeFeature
        self.seqLength = shiftRange[1]- shiftRange[0]
        self.lagLength = shiftRange[0]

        self.deepScaler = pre.MinMaxScaler()
        self.scaler = pre.StandardScaler() 
        
        if isTorch:
            self.scaleData, self._y, self._times = self._initDeepData()
        else:
            self.scaleData, self.nonscaleData, self._y, self._times = self._initData()
        
        
        # print(np.array(self.scaleData))
        # self.deepScaler.fit(self.scaleData['meter'].values)
    
    
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


    def _initDeepData(self):
        df = self._readData(self.dataPath)
        df = df.sort_values(by=['datetime'])
        trainData = df['meter'].values.reshape(-1, 1)
        scaler = pre.MinMaxScaler().fit(trainData[:])
        trainData = scaler.transform(trainData)


        hourOfDay = df['hourOfDay'].values.reshape(-1, 1)
        hourOfDay = pre.MinMaxScaler().fit_transform(hourOfDay)
 
        dayOfWeek = df['dayOfWeek'].values.reshape(-1, 1)
        dayOfWeek = pre.MinMaxScaler().fit_transform(dayOfWeek)

        holiday = df['holiday'].values.reshape(-1, 1)
        holiday = pre.MinMaxScaler().fit_transform(holiday)

        dayOfYear = df['dayOfYear'].values.reshape(-1, 1)
        dayOfYear = pre.MinMaxScaler().fit_transform(dayOfYear)

        times = df[self.seqLength+self.lagLength:]['datetime'].values
        x, y = self._slidingWindows(trainData, hourOfDay, dayOfWeek, dayOfYear, holiday)
        
        # times = times
        return x, y, times

    
    def _slidingWindows(self, data, hourOfDay, dayofWeek, dayOfYear, holiday):
        x = []
        y = []
        for i in range(len(data)-self.seqLength-self.lagLength):
            _x = data[i:(i+self.seqLength)]
            if self.useTimeFeature:  
                _hourOfDay = hourOfDay[i:(i+self.seqLength)]
                _dayofWeek = dayofWeek[i:(i+self.seqLength)]
                _dayOfYear = dayOfYear[i:(i+self.seqLength)]
                _holiday = holiday[i:(i+self.seqLength)]
                _x = np.hstack((_x,_hourOfDay, _dayofWeek, _dayOfYear, _holiday))
            
            _y = data[i+self.seqLength+self.lagLength]
            x.append(_x)
            y.append(_y)

        return np.array(x),np.array(y)


    def _initData(self):
        df = self._readData(self.dataPath)
        df = df.sort_values(by=['datetime'])
        df = self._shiftData(df=df)
        df = df.reset_index()
        features = [f for f in list(df.columns) if f not in self.removeFeatures]
        df = df[features]  

        hourOfDay = df['hourOfDay'].values.reshape(-1, 1)
        dayOfWeek = df['dayOfWeek'].values.reshape(-1, 1)
        holiday = df['holiday'].values.reshape(-1, 1)
        dayOfYear = df['dayOfYear'].values.reshape(-1, 1)
        df['hourOfDay'] = list(pre.MinMaxScaler().fit_transform(hourOfDay).ravel())
        df['dayOfWeek'] = list(pre.MinMaxScaler().fit_transform(dayOfWeek).ravel())
        df['holiday'] = list(pre.MinMaxScaler().fit_transform(holiday).ravel())
        df['dayOfYear'] = list(pre.MinMaxScaler().fit_transform(dayOfYear).ravel())

        return self._splitData(df)


    def _shiftData(self, df):
        for i in range(self.shiftRange[0], self.shiftRange[1] + 1):
            for f in self.shiftFeatures:
                df['prev_' + f + str(i)] = df[f].shift(periods=i)
        return df.dropna().copy()


    def _splitData(self, df):    
        scaleFeatures = [f for f in df.columns if f not in self.categoryFeatures and f not in ['datetime']]
        nonscaleFeatures = [f for f in df.columns if f in self.categoryFeatures]
        return df[scaleFeatures].copy(), df[nonscaleFeatures].copy(), df['meter'].copy(), df['datetime'].copy()


    def _readData(self, dataPath):
        try:
            df = pd.read_csv(dataPath)
        except Exception as e:
            logging.error(e)
            raise e
        return df.dropna()


    def _getPairData(self, idxFrom, idxTo):

        if self.isTorch:
            return self.scaleData[idxFrom: idxTo], self.y[idxFrom: idxTo]  
        else:
            scaleData = self.scaleData.iloc[idxFrom:idxTo,:].copy()
            scaleData = np.array(scaleData)[0]
            nonscaleData = self.nonscaleData.iloc[idxFrom:idxTo,:].copy().to_numpy()

            self.scaler.partial_fit(scaleData[:, np.newaxis])
            scaleData = self.scaler.transform(scaleData[:, np.newaxis])

            scaleData = np.array(scaleData[1:].copy())
            nonscaleData =np.reshape(nonscaleData, (len(self.nonscaleData.columns), 1))
            
            input = np.concatenate((scaleData, nonscaleData), axis=0)
            y = self._y[idxFrom:idxTo]
            return input, y


    def getTrainData(self, idxFrom, idxTo):
        if self.isTorch:   
            input, y = self._getPairData(idxFrom, idxTo)   
            return Variable(torch.Tensor(np.array(input))), Variable(torch.Tensor(np.array(y)))
    
        else:          
            inputs = np.zeros(shape=(idxTo - idxFrom, len(self.scaleData.columns)+ len(self.nonscaleData.columns) -1))
            ys = np.zeros(shape=(idxTo - idxFrom,))
            for i in range(idxFrom, idxTo): 
                input, y = self._getPairData(i,i+1)
                inputs[i-idxFrom] = input.ravel()
                ys[i-idxFrom] = y
            return inputs, ys

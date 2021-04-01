import os
import os.path
import time
import threading
import numpy as np
from .utils import *
import logging
from sklearn.svm import SVR
from .config import LocalConfig
from sklearn import preprocessing as pre
from sklearn.metrics import r2_score, mean_absolute_percentage_error


__all__ = ['Runner']


class Runner:
    def __init__(self, dataPath, modelPath, features, shiftRange=[8, 24], modelName='svr'):
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.modelName = modelName
        self.modelFilePath = None
        self.features = features
        self.shiftRange = shiftRange

        self.X, self.y, self.times = self._prepareData()

        self.model = self._getModel(modelName)
        self.scaler = pre.StandardScaler()

        self.predList, self.actualList, self.scoreList = [], [], []
        self.update = False
        self.acceptable = True
        self.idx = 0

    def _prepareData(self):
        df = getSourceData(self.dataPath)
        df = shiftData(df=df, features=self.features,
                       shiftFrom=self.shiftRange[0], shiftTo=self.shiftRange[1])
        df = df[self.shiftRange[0]:].reset_index().copy()
        return splitData(df)

    def _getModel(self, modelName):
        return {
            'svr': SVR(kernel='rbf', C=100, gamma=0.04, epsilon=.01),
        }[modelName]

    def _warmStart(self, idxFrom, idxTo):
        # pretrain model
        XTrain, yTrain = getTrainData(
            X=self.X, y=self.y, idxFrom=idxFrom, idxTo=idxTo)

        XTrain = self.scaler.fit_transform(XTrain)
        self.model.fit(XTrain, yTrain)

        # get new data pool
        self.X, self.y, self.times = self.X[idxTo:
                                            ], self.y[idxTo:], self.times[idxTo:]
        self.modelFilePath = self.modelPath + self.modelName + '.pkl'

        saveModel(self.model, self.modelFilePath)

    def _updateModel(self):
        if self.update and os.path.isfile(self.modelFilePath):
            self.model = loadModel(self.modelFilePath)
            os.remove(self.modelFilePath)
            self.update = False
            self.idx += 1

    def _evaluateResult(self, method, idxFrom, idxTo, baseScore):
        yPred, yActual = self.predList[idxFrom:
                                       idxTo], self.actualList[idxFrom: idxTo]
        if not self.update:

            if method == 'r2':
                score = r2_score(y_pred=yPred, y_true=yActual)
                self.update = False if score > baseScore else True

            elif method == 'mape':
                score = mean_absolute_percentage_error(
                    y_pred=yPred, y_true=yActual)
                self.update = False if score < baseScore else True

            self.scoreList.append(score)

    def _predictResult(self, idxFrom, idxTo):
        X = np.array(self.X[idxFrom: idxTo])
        XTrans = self.scaler.transform(X)  # normalize input
        predVal = self.model.predict(XTrans)
        self.predList.append(predVal[0])
        self.actualList.append(self.y[idxFrom:idxTo].values[0])

    def _retrainModel(self, idxFrom, idxTo):
        XTrain, yTrain = getTrainData(
            X=self.X, y=self.y, idxFrom=idxFrom, idxTo=idxTo)
        self.scaler = pre.StandardScaler()
        XTrans = self.scaler.fit_transform(XTrain)

        self.model.fit(XTrans, yTrain)

        self.modelFilePath = self.modelPath + self.modelName + self.idx + '.pkl'
        print(self.modelFilePath)
        saveModel(self.model, self.modelFilePath)
        self.update = True

    def run(self, duration, interval):
        begin = time.time()
        cur = 0
        nxt = 48
        self._warmStart(idxFrom=cur, idxTo=nxt)

        # streaming loop
        while time.time() - begin < duration:
            cur = nxt
            nxt = cur + 1

            time.sleep(interval)
            self.update = False

            # update model
            self._updateModel()
            self._predictResult(idxFrom=cur, idxTo=nxt)

            # evaluate model
            evaluateThld = 48 * 2
            if cur >= evaluateThld and not self.update:
                self._evaluateResult(
                    method='mape', idxFrom=cur-evaluateThld, idxTo=cur, baseScore=0.3)

            # retrain model
            trainThld = 48 * 2
            if self.update and cur > trainThld:
                print('retrain')
                train = threading.Thread(
                    target=self._retrainModel(idxFrom=cur-48, idxTo=cur), args=(1,))
                train.start()

        # logging.info(self.scoreList)
        plotResult(self.actualList, self.predList)

import os
import os.path
import time
import threading
import numpy as np
from .utils import *
import logging
from .config import LocalConfig
from sklearn.metrics import r2_score, mean_absolute_percentage_error


__all__ = ['Runner']


class Runner:
    def __init__(self, dataPath, features, shiftRange, model):
        self.dataPath = dataPath
        self.features = features
        self.shiftRange = shiftRange
        self.X, self.y, self.times = self._prepareData()

        self.model = model
        self.predList, self.actualList, self.scoreList = [], [], []

    def _prepareData(self):
        df = getSourceData(self.dataPath)
        df = shiftData(df=df, features=self.features,
                       shiftFrom=self.shiftRange[0], shiftTo=self.shiftRange[1])
        df = df[self.shiftRange[0]:].reset_index().copy()
        return splitData(df)

    def _warmStart(self, idxFrom, idxTo):
        # pretrain model
        XTrain, yTrain = getTrainData(
            X=self.X, y=self.y, idxFrom=idxFrom, idxTo=idxTo)

        XTrain = self.model.scaler.fit_transform(XTrain)
        self.model.fit(XTrain, yTrain)

        # get new data pool
        self.X, self.y, self.times = self.X[idxTo:
                                            ], self.y[idxTo:], self.times[idxTo:]
        self.model.save()

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

    def _predictAndLog(self, idxFrom, idxTo):
        XTrain, yTrain = getTrainData(
            X=self.X, y=self.y, idxFrom=idxFrom, idxTo=idxTo)
        predVal = self.model.predict(XTrain)

        self.predList.append(predVal[0])
        self.actualList.append(yTrain[0])

    def run(self, duration, interval):
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO,
                            datefmt="%H:%M:%S")
        begin = time.time()
        cur = 0
        nxt = 48
        self._warmStart(idxFrom=cur, idxTo=nxt)

        # streaming
        while time.time() - begin < duration:
            cur = nxt
            nxt = cur + 1

            time.sleep(interval)
            self.update = False

            # predict
            self._predictAndLog(idxFrom=cur, idxTo=nxt)

            # evaluate model
            evaluateThld = 48 * 2
            if cur >= evaluateThld and not self.update:
                self._evaluateResult(
                    method='mape', idxFrom=cur-evaluateThld, idxTo=cur, baseScore=0.3)

            # retrain model
            trainThld = 48 * 2
            if self.update and cur > trainThld:
                logging.info('retrain')
                XTrain, yTrain = getTrainData(
                    X=self.X, y=self.y, idxFrom=cur-48, idxTo=cur)
                train = threading.Thread(
                    target=self.model.fit(XTrain, yTrain), args=(1,))
                train.start()

        # logging.info(self.scoreList)
        plotResult(self.actualList, self.predList)

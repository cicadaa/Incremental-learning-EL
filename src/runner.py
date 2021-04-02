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
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.X = self.dataset.X
        self.y = self.dataset.y
        self.model = model
        self.predList, self.actualList, self.scoreList = [], [], []

    def _warmStart(self, idxFrom, idxTo):
        # pretrain model
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=idxFrom, idxTo=idxTo)
        XTrain = self.model.scaler.fit_transform(XTrain)
        self.model.fit(XTrain, yTrain)

        # update dataset
        self.dataset.X = self.dataset.X[idxTo:]
        self.dataset.y = self.dataset.y[idxTo:]
        self.dataset.times = self.dataset.times[idxTo:]

        self.model.save()

    def _evaluate(self, method, idxFrom, idxTo, baseScore):
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
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=idxFrom, idxTo=idxTo)
        predVal = self.model.predict(XTrain)
        self.predList.append(predVal[0])
        self.actualList.append(yTrain[0])

    def run(self, duration, interval, evaluateThreshold=96, trainThreshold=96):

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
            evaluateThld = evaluateThreshold
            if cur >= evaluateThld and not self.update:
                self._evaluate(
                    method='mape', idxFrom=cur-evaluateThld, idxTo=cur, baseScore=0.3)

            # retrain model
            trainThld = trainThreshold
            if self.update and cur > trainThld:
                logging.info('retrain')
                XTrain, yTrain = self.dataset.getTrainData(
                    idxFrom=cur-48, idxTo=cur)
                train = threading.Thread(
                    target=self.model.fit(XTrain, yTrain), args=(1,))
                train.start()

        # logging.info(self.scoreList)
        plotResult(self.actualList, self.predList)

import os
import os.path
import time
import threading
import numpy as np
from .utils import *
import logging
from .config import LocalConfig
from .models import DeepTrainer
from sklearn.metrics import r2_score, mean_absolute_percentage_error


__all__ = ['Runner']


class Runner:
    def __init__(self, warmStartPoint, dataset, model, deep=False):
        self.cur = None
        self.nxt = None

        self.X = dataset.X
        self.y = dataset.y
        self.times = dataset.times
        self.dataset = dataset

        self.model = model
        self.startPont = warmStartPoint
        self.predList, self.actualList, self.scoreList = [], [], []
        self.isdeep = deep
        if self.isdeep:
            self.trainer = DeepTrainer(learningRate=0.01, model=self.model)

    def _warmStart(self):
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=0, idxTo=self.startPont)
        if self.isdeep:
            print('train: ', XTrain)
            self.trainer.learn(XTrain, yTrain)
        else:
            self.model.learn(XTrain, yTrain)

    def _evaluate(self, method, range, baseScore):
        idxFrom, idxTo = self.cur - range, self.cur
        if self.model.acceptable:
            yPred, yActual = self.predList[idxFrom:
                                           idxTo], self.actualList[idxFrom: idxTo]
            if method == 'r2':
                score = r2_score(y_pred=yPred, y_true=yActual)
                self.model.acceptable = False if score < baseScore else True
            elif method == 'mape':
                score = mean_absolute_percentage_error(
                    y_pred=yPred, y_true=yActual)
                self.model.acceptable = False if score > baseScore else True
            self.scoreList.append(score)

    def _predict(self, log=True):
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=self.cur, idxTo=self.nxt)
        yPred = self.model.predict(XTrain)
        if log:
            self.predList.append(yPred[0])
            self.actualList.append(yTrain[0])

    def _learn(self):
        XTrain, yTrain = self.dataset.getTrainData(
            idxFrom=self.cur, idxTo=self.nxt)
        self.model.learn(XTrain, yTrain)

    def _update(self):
        self.cur += 1
        self.nxt = self.cur + 1

    def run(self, duration, plotname, interval, evaluate=False, deep=False):

        begin = time.time()
        self._warmStart()
        self.cur = self.startPont

        # streaming
        while time.time() - begin < duration:
            time.sleep(interval)
            # print(self.cur)
            self._update()
            self._predict()

            # evaluate model
            acceptable = self._evaluate('r2', 24, 0.7) if evaluate else False

            if not acceptable:
                self._learn()

        logging.info('scorelist : ' + str(self.scoreList))
        plotlyplot(actual=self.actualList, prediction=self.predList,
                   times=self.times[:self.cur-1], plotname=plotname)

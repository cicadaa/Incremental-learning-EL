
import os
import os.path
import time
import threading
import numpy as np
from .utils import *
from .agents import Dataset
from sklearn.svm import SVR
from .config import LocalConfig
from sklearn import preprocessing as pre


__all__ = ['Runner']


class Runner:
    def __init__(self, config):
        self.datapath = config.datapath
        self.features = config.features
        self.prevFrom = config.prevFrom
        self.prevTo = config.prevTo
        self.modelPath = config.modelPath

        self.dataset = Dataset(self.datapath)
        self.X, self.y, self.times = self.dataset.getSourceData(
            self.features, self.prevFrom, self.prevTo)

        self.onlineModel = SVR(
            kernel='rbf', C=10, gamma=0.04, epsilon=.01)
        self.scaler = pre.StandardScaler()
        self.predList, self.actualList, self.scoreList = [], [], []
        self.update = False
        self.acceptable = True

    def _warmStart(self, idxFrom, idxTo):
        # pretrain model
        XTrain, yTrain = np.array(
            self.X[idxFrom: idxTo]), np.array(self.y[idxFrom: idxTo])
        XTrain = self.scaler.fit_transform(XTrain)
        self.onlineModel.fit(XTrain, yTrain)

        # get new data pool
        self.X, self.y, self.times = self.X[idxTo:
                                            ], self.y[idxTo:], self.times[idxTo:]

        saveModel(self.onlineModel, self.modelPath)

    def _updateModel(self, update=False):
        if update and os.path.isfile(self.modelPath):
            self.onlineModel = loadModel(self.modelPath)
            os.remove(self.modelPath)
            self.update = False

    def _evaluateResult(self, method, idxFrom, idxTo, update, basescore):
        if idxFrom >= 12 and not update:
            if method == 'r2':
                score = r2_score(
                    self.predList[idxFrom: idxTo], self.actualList[idxFrom: idxTo])

                self.scoreList.append(score)
                self.update = False if score > basescore*0.9 else True

    def _predictResult(self, idxFrom, idxTo):
        XTrain = np.array(self.X[idxFrom: idxTo])
        XTrain = self.scaler.transform(XTrain)  # normalize input
        prediction = self.onlineModel.predict(XTrain)

        self.predList.append(prediction[0])
        self.actualList.append(self.y[idxFrom:idxTo].values[0])

    def run(self, duration, interval):
        begin = time.time()
        self._warmStart(idxFrom=0, idxTo=12)
        cur = 0

        # streaming loop
        while time.time() - begin < duration:

            time.sleep(interval)
            nxt = cur + 1
            update = False

            # update model
            self._updateModel(update=update)
            self._predictResult(cur, nxt)

            # evaluate model
            if cur >= 12 and not update:
                self._evaluateResult(
                    method='r2', idxFrom=cur-12, idxTo=cur, update=update, basescore=0.8)

            # retrain model
            if not self.acceptable and not self.update and cur > 24:
                logging.info('retrain')
                XTrain, yTrain = self.dataset.getTrainData(
                    XSource=self.X, ySource=self.y, idxFrom=cur-24, idxTo=cur, scaler=self.scaler)
                train = threading.Thread(
                    target=trainAndUpdateModel(model=self.onlineModel, XTrain=XTrain, yTrain=yTrain), args=(1,))
                train.start()
                self.update = True

            cur = nxt

        print(self.scoreList)
        plotResult(self.actualList, self.predList)

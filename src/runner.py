
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
from sklearn.metrics import r2_score, mean_absolute_percentage_error


__all__ = ['Runner']


class Runner:
    def __init__(self, dataPath, modelPath, features, shiftRange=[12, 36], model='SVR'):
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.features = features

        self.shiftRange = shiftRange
        self.dataset = Dataset(self.dataPath)
        self.X, self.y, self.times = self.dataset.getSourceData(
            self.features, self.shiftRange[0], self.shiftRange[1])

        self.model = self._getModel(model)
        self.scaler = pre.StandardScaler()

        self.predList, self.actualList, self.scoreList = [], [], []
        self.update = False
        self.acceptable = True

    def _getModel(self, model):
        return {
            'svr': SVR(kernel='rbf', C=10, gamma=0.04, epsilon=.01),
        }[model]

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
            # os.remove(self.modelPath)
            self.update = False

    def _evaluateResult(self, method, idxFrom, idxTo, update, basescore):
        yPred, yActual = self.predList[idxFrom:
                                       idxTo], self.actualList[idxFrom: idxTo]
        if idxFrom >= 12 and not update:

            if method == 'r2':
                score = r2_score(y_pred=yPred, y_true=yActual)
                self.update = False if score > basescore else True

            elif method == 'mape':
                score = mean_absolute_percentage_error(
                    y_pred=yPred, y_true=yActual)
                self.update = False if score < basescore else True
                self.scoreList.append(score)

    def _predictResult(self, idxFrom, idxTo):
        X = np.array(self.X[idxFrom: idxTo])
        X = self.scaler.transform(X)  # normalize input

        prediction = self.onlineModel.predict(X)
        self.predList.append(prediction[0])
        self.actualList.append(self.y[idxFrom:idxTo].values[0])

    def run(self, duration, interval):
        begin = time.time()
        self._warmStart(idxFrom=0, idxTo=48)
        cur = 0
        i = 0

        # streaming loop
        while time.time() - begin < duration:

            time.sleep(interval)
            nxt = cur + 1
            self.update = False

            # update model
            self._updateModel(update=self.update)
            self._predictResult(cur, nxt)

            # evaluate model
            evaluateThld = 48
            if cur >= evaluateThld and not self.update:
                self._evaluateResult(
                    method='mape', idxFrom=cur-evaluateThld, idxTo=cur, update=self.update, basescore=0.3)

            # retrain model
            trainThld = 48
            if not self.acceptable and not self.update and cur > trainThld:
                XTrain, yTrain = self.dataset.getTrainData(
                    XSource=self.X, ySource=self.y, idxFrom=0, idxTo=cur)
                self.scaler = pre.StandardScaler()
                XTrain = self.scaler.fit_transform(XTrain)
                train = threading.Thread(
                    target=trainAndUpdateModel(model=self.onlineModel, XTrain=XTrain, yTrain=yTrain, id=i), args=(1,))
                train.start()
                self.modelPath = 'src/models/svrLatest{0}.pkl'.format(i)
                self.update = True

            cur = nxt

        logging.info(self.scoreList)
        plotResult(self.actualList, self.predList)

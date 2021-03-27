# from .. import DMI
# from .retriever import DataRetriever

import time
from .config import LocalConfig
import threading
from sklearn.svm import SVR
from sklearn import preprocessing as pre
import numpy as np
from datetime import datetime, timedelta
from .agents import Evaluator, Dataset
from .helper import *
import os
import os.path


def getNextTime(start, interval):
    end = datetime.strptime(
        start, "%Y-%m-%d %H:%M:%S") + timedelta(hours=interval)
    return end.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":

    begin = time.time()
    dataset = Dataset('data.csv')
    X, y, times = dataset.getSourceData(['meter'], 12, 24)

    # init online model
    onlineModel = SVR(kernel='rbf', C=10, gamma=0.04, epsilon=.01)

    # warm start
    XTrain, yTrain = np.array(X[: 24]), np.array(y[: 24])
    scaler = pre.StandardScaler().fit(XTrain)
    XTrain = scaler.transform(XTrain)
    onlineModel.fit(XTrain, yTrain)
    saveModel(onlineModel, 'src/models/svr_base.pkl')

    # init evaluator and offline training
    X, y, times = X[24:], y[24:], times[24:],
    cur = 0

    evaluator = Evaluator('r2', 0.8)

    predList = []
    actualList = []
    scoreList = []
    updateModel = False

    # streaming loop
    while time.time() - begin < 20:
        # print(cur)
        time.sleep(0.5)
        nxt = cur + 1

        # update model
        if updateModel and os.path.isfile('src/models/latestModel.pkl'):
            onlineModel = loadModel('src/models/latestModel.pkl')
            os.remove("src/models/latestModel.pkl")
            updateModel = False

        # predict
        input = np.array(X[cur: nxt])
        input = scaler.transform(input)  # normalize input
        prediction = onlineModel.predict(input)

        predList.append(prediction[0])
        actualList.append(y[cur:nxt].values[0])
        # scoreList.append(onlineModel.score(input, np.array(y[cur:nxt])))

        # print('pred: ', predList)
        # evaluate
        isAcceptable = True

        if cur >= 12 and not updateModel:
            isAcceptable, eScore = evaluator.evaluate(
                predList[cur-12: cur], actualList[cur-12: cur])
            scoreList.append(eScore)

        # retrain model
        if not isAcceptable and not updateModel and cur > 24:
            print('retrain')
            XTrain, yTrain = dataset.getTrainData(
                XSource=X, ySource=y, timeIdx=cur, window=24, scaler=scaler)
            train = threading.Thread(
                target=trainAndUpdateModel(model=onlineModel, XTrain=XTrain, yTrain=yTrain), args=(1,))
            train.start()
            updateModel = True

        cur = nxt

    print(scoreList)
    plotResult(actualList, predList)

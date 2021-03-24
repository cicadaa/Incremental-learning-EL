# from .. import DMI
import time
from .config import LocalConfig
# from .retriever import DataRetriever
from sklearn.svm import SVR
from sklearn import preprocessing as pre
import numpy as np
from datetime import datetime, timedelta
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from .simulator import Simulator


def plotResult(actual, prediction):
    fig, ax = plt.subplots(figsize=(26, 10))  # Create a figure and an axes.
    ax.plot(actual, label='Actual', color='blue')
    ax.plot(prediction, label='Prediction', color='red')
    ax.legend()
    plt.show()


def getNextTime(start, interval):
    end = datetime.strptime(
        start, "%Y-%m-%d %H:%M:%S") + timedelta(hours=interval)
    return end.strftime("%Y-%m-%d %H:%M:%S")


def getSourceData(path, columns, prevFrom, prevTo):
    df = pd.read_csv(path)
    for i in range(prevFrom, prevTo+1):
        for c in columns:
            df['prev_' + c + str(i)] = df[c].shift(periods=i)
            df['pre_' + c + str(i)] = df[c].shift(periods=i)
    df = df.dropna()
    y = df['meter']
    times = df['datetime']

    del df['datetime']
    del df['meter']
    del df['temp']

    X = df
    return X, y, times


if __name__ == "__main__":

    begin = time.time()
    X, y, times = getSourceData('data.csv', ['meter'], 12, 24)
    cur = 12
    nxt = 36

    simulator = Simulator()
    model = simulator.model

    scaler = pre.StandardScaler().fit(np.array(X[cur: nxt]))
    XTrain = scaler.transform(np.array(X[cur: nxt]))

    model.fit(XTrain, np.array(y[cur: nxt]))

    simulator.saveModel(model, 'src/models/svr_base.pkl')

    predList = []
    actualList = []
    while time.time() - begin < 0.1:

        cur = nxt
        nxt = cur + 1

        input = np.array(X[cur: nxt])
        input = scaler.transform(input)

        prediction = model.predict(input)

        predList.append(prediction[0])
        actualList.append(y[cur:nxt].values[0])

    plotResult(actualList, predList)

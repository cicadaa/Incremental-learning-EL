import time
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


__all__ = ['loadModel', 'saveModel',
           'trainAndUpdateModel', 'plotResult', 'getSourceData', 'getTrainData', 'shiftData', 'splitData']
# Model Management=============================================================


def loadModel(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def saveModel(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def trainModel(model, XTrain, yTrain):
    logging.info('start training')
    model.fit(XTrain, yTrain)
    logging.info('finished training')


def trainAndUpdateModel(model, XTrain, yTrain, id):
    model.fit(XTrain, yTrain)
    saveModel(model, 'src/models/latestModel{0}.pkl'.format(id))


# Visualization ===============================================================

def plotResult(actual, prediction, figsize=(26, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(actual, label='Actual', color='blue')
    ax.plot(prediction, label='Prediction', color='red')
    ax.legend()
    plt.show()


# Time Formater ===============================================================

def getNextTime(start, interval):
    end = datetime.strptime(
        start, "%Y-%m-%d %H:%M:%S") + timedelta(hours=interval)
    return end.strftime("%Y-%m-%d %H:%M:%S")


# Prepare Data ===============================================================

def getTrainData(X, y, idxFrom, idxTo):
    XTrain = np.array(X[idxFrom: idxTo].copy())
    yTrain = np.array(y[idxFrom: idxTo].copy())
    return XTrain, yTrain


def shiftData(df, features, shiftFrom, shiftTo):
    for i in range(shiftFrom, shiftTo+1):
        for f in features:
            df['prev_' + f + str(i)] = df[f].shift(periods=i)
    return df.dropna().copy()


def splitData(df):
    removeLst = set(['datetime', 'meter', 'temp'])
    features = [e for e in list(df.columns) if e not in removeLst]
    return df[features].copy(), df['meter'].copy(), df['datetime'].copy()


def getSourceData(dataPath):
    try:
        df = pd.read_csv(dataPath)
    except:
        raise Exception
    return df.dropna()

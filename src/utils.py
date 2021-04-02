import time
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


__all__ = ['loadModel', 'saveModel',
           'trainAndUpdateModel', 'plotResult']
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


def trainAndUpdateModel(model, XTrain, yTrain, modelId):
    model.fit(XTrain, yTrain)
    saveModel(model, 'src/models/latestModel{0}.pkl'.format(modelId))


# Visualization ===============================================================

def plotResult(actual, prediction, figsize=(26, 10)):
    _, ax = plt.subplots(figsize=figsize)
    ax.plot(actual, label='Actual', color='blue')
    ax.plot(prediction, label='Prediction', color='red')
    ax.legend()
    plt.show()


# Time Formater ===============================================================

def getNextTime(start, interval):
    timeFormat = "%Y-%m-%d %H:%M:%S"
    end = datetime.strptime(start, timeFormat) + timedelta(hours=interval)
    return end.strftime(timeFormat)

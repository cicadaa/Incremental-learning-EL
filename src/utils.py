import time
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


__all__ = ['loadModel', 'plotResult']

# Model Management=============================================================


def loadModel(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

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

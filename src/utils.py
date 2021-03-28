import time
import pickle
import logging
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from .retriever import *


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


def trainAndUpdateModel(model, XTrain, yTrain):
    begin = time.time()
    logging.info('start training')
    model.fit(XTrain, yTrain)
    logging.info('finished training')
    saveModel(model, 'src/models/latestModel.pkl')
    logging.info('updates model')
    logging.info('dutation'+str(time.time()-begin))


# Fetch Data ===============================================================


def getChunckData(self, startDate, endDate, path, url) -> pd.DataFrame:
    # 'temp_mean_past1h', 'temp_dry'-> every 10 min
    dmiRetriever = DMIRetriever(path=path, url=url)
    sqlRetriever = SQLRetriever()

    # clean temperature data frame
    dfTemp = dmiRetriever.getWeatherData(
        startDate=startDate, endDate=endDate, stationId="06123", field='temp_mean_past1h', limit='100000')

    # clean consumptiond dataframe
    dfConsumption = sqlRetriever.getConsumption(columns=['datetime', 'sum'], table='consumptionAggregated',
                                                startDate=startDate, endDate=endDate)

    dfConsumption = dfConsumption.resample('1H').sum()

    # merge temp and consumption data
    return dfConsumption.join(dfTemp)


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

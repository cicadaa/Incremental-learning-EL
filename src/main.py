# from .. import DMI
import time
from .config import LocalConfig
from .stream import Streamer
from sklearn.svm import SVR
from sklearn import preprocessing as pre
import numpy as np


if __name__ == "__main__":
    start = time.time()
    # while time.time() - start < 0.05:

    streamer = Streamer(path=LocalConfig.apiKeyPathDMI, url=LocalConfig.urlDMI)
    df = streamer.getChunck(startDate='2019-01-01', endDate='2019-2-21')
    timeWindow = 7
    for i in range(1, timeWindow+1):
        df['prev_meter' + str(i)] = df['meter'].shift(periods=i)
        df['pre_temp' + str(i)] = df['temp'].shift(periods=i)

    XTrain = df['2019-01-02':'2019-01-21']
    del XTrain['meter']
    del XTrain['temp']

    XTest = df['2019-02-02':'2019-02-10']
    del XTest['meter']
    del XTest['temp']

    yTrain = df['meter']['2019-01-02':'2019-01-21']
    yTest = df['meter']['2019-02-02':'2019-02-10']

    XTrain = np.array(XTrain)
    yTrain = np.array(yTrain)

    XTest = np.array(XTest)
    yTest = np.array(yTest)

    scaler = pre.StandardScaler().fit(XTrain)
    XTrain = scaler.transform(XTrain)
    XTest = scaler.transform(XTest)

    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr.fit(XTrain, yTrain)
    print('score', svr.score(XTest, yTest))

    # svr.fit

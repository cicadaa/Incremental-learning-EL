import time
import pickle
import logging
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score


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


# Evaluate Model ===============================================================

# def evaluate(method, prediction, actual, basescore=0.8):
#     if method == 'r2':
#         score = r2_score(actual, prediction)
#         if score > basescore*0.9:
#             return True, score
#         else:
#             return False, score


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

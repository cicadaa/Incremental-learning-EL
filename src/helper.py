import pickle
from matplotlib import pyplot as plt
import time

# model management


def loadModel(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def saveModel(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def trainModel(model, XTrain, yTrain):
    # TODO: use logging instead of print
    print('start training')
    model.fit(XTrain, yTrain)
    print('finished training')
    return model


def trainAndUpdateModel(model, XTrain, yTrain):
    begin = time.time()
    print('start training')
    model.fit(XTrain, yTrain)
    print('finished training')
    saveModel(model, 'src/models/latestModel.pkl')
    print('updates model')
    print('dutation', time.time()-begin)

    # return model


# visualization

def plotResult(actual, prediction):
    # TODO: figsize should also be a parameter with default value (26, 10)
    fig, ax = plt.subplots(figsize=(26, 10))  # Create a figure and an axes.
    ax.plot(actual, label='Actual', color='blue')
    ax.plot(prediction, label='Prediction', color='red')
    ax.legend()
    plt.show()

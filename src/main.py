from .config import LocalConfig
from .runner import Runner
from .models import SVRModel, OSVRModel, LSTM
from .dataset import Dataset
import logging


if __name__ == "__main__":
    logformat = "%(asctime)s: %(message)s"
    logging.basicConfig(format=logformat, level=logging.INFO,
                        datefmt="%H:%M:%S")

    dataPath = 'data_accurateTemp.csv'
    modelPath = LocalConfig.modelPath
    shiftFeatures = LocalConfig.features
    shiftRange = LocalConfig.shiftRange
    removeSet = LocalConfig.removeSet

    # model = OSVRModel(learning_rate='constant', eta0=0.3,
    #                   loss='epsilon_insensitive', penalty='l2')

    learning_rate = 0.01
    input_size = 1
    hidden_size = 2
    num_layers = 1
    num_classes = 1

    dataset = Dataset(dataPath=dataPath, shiftFeatures=['meter'],
                      shiftRange=shiftRange, removeSet=removeSet, isTorch=True)

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

    runner = Runner(warmStartPoint=1, dataset=dataset, model=lstm, deep=True)
    runner.run(duration=1, interval=0.2, plotname='withouttemp')

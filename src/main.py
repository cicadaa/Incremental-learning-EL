from numpy import record
from torch.nn.modules.module import T
from .config import LocalConfig
from .runner import Runner
from .models import SVRModel, OSVRModel, LSTM
from .dataset import Dataset
import logging


if __name__ == "__main__":
    logformat = "%(asctime)s: %(message)s"
    logging.basicConfig(format=logformat, level=logging.INFO,
                        datefmt="%H:%M:%S")

    dataPath = '/Users/cicada/Documents/DTU_resource/Thesis/Incremental-learning-EL/src/NI_hourly.csv'
    modelPath = LocalConfig.modelPath
    shiftFeatures = LocalConfig.features
    shiftRange = LocalConfig.shiftRange
    removeSet = LocalConfig.removeSet

    # model = OSVRModel(learning_rate='constant', eta0=0.3,
    #                   loss='epsilon_insensitive', penalty='l2')

    learning_rate = 0.1
    input_size = 1
    hidden_size = 128
    num_layers = 1
    num_classes = 1

    dataset = Dataset(dataPath=dataPath, shiftFeatures=['meter'],
                      shiftRange=shiftRange, removeSet=removeSet, isTorch=True)
  
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

    runner = Runner(warmStartPoint=1, dataset=dataset, model=lstm, deep=True, learningRate=learning_rate)
    runner.run(duration=10, interval=0.01, name='OLSTM', plot=True, record=True)

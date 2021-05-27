# from distutils import version
from numpy import record
from torch.nn.modules.module import T
from .config import LocalConfig
from .runner import Runner
from .models import SVRModel, OSVRModel, OLSTM
from .dataset import Dataset
import logging


if __name__ == "__main__":
    logformat = "%(asctime)s: %(message)s"
    logging.basicConfig(format=logformat, level=logging.INFO,
                        datefmt="%H:%M:%S")

    dataPath = '/Users/cicada/Documents/DTU_resource/Thesis/Incremental-learning-EL/src/NI_hourly_all.csv'
    modelPath = LocalConfig.modelPath
    categoryFeatures = LocalConfig.categoryFeatures
    shiftFeatures = LocalConfig.shiftFeatures
    shiftRange = LocalConfig.shiftRange
    removeFeatures = LocalConfig.removeFeatures

    osvr = OSVRModel(learning_rate='constant', eta0=0.3,
                      loss='epsilon_insensitive', penalty='l2')

    olstm = OLSTM(num_classes=1, input_size=1, hidden_size=128, num_layers=1)

    learning_rate = 0.01 #best rate


    dataset = Dataset(dataPath=dataPath, shiftFeatures=['meter'], categoryFeatures=['dayOfYear','hourOfDay','dayOfWeek','holiday','weekend'],
                      shiftRange=shiftRange, removeFeatures=removeFeatures, isTorch=False)
    
    runner = Runner(warmStartPoint=1, dataset=dataset, model=osvr, deep=False, learningRate=learning_rate, lazy=False)
    runner.run(duration=0.1, interval=0, name='OSVR', plot=True, record=True, verbose=False)

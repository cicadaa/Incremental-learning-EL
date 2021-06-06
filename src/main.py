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

    dataPath = LocalConfig.dataPath
    categoryFeatures = LocalConfig.categoryFeatures
    shiftFeatures = LocalConfig.shiftFeatures
    shiftRange = LocalConfig.shiftRange
    removeFeatures = LocalConfig.removeFeatures
    learning_rate = 0.01 #best rate
    
    osvr = OSVRModel(learning_rate='constant', eta0=0.4,
                      loss='epsilon_insensitive', penalty='l2')

    olstm = OLSTM(num_classes=1, input_size=1, hidden_size=256, num_layers=1)

    dataset_osvr = Dataset(dataPath=dataPath, shiftFeatures=['meter'], categoryFeatures=[],
                      shiftRange=shiftRange, removeFeatures=removeFeatures, isTorch=False )

    dataset_olstm = Dataset(dataPath=dataPath, shiftFeatures=['meter'], categoryFeatures=[],
                      shiftRange=shiftRange, removeFeatures=removeFeatures, isTorch=True )
    dataset_olstm.getTrainData(0,1)
    # dataset.getTrainData(1,2)

    # #osvr
    # runner = Runner(warmStartPoint=1, dataset=dataset_osvr, model=osvr, deep=False, learningRate=learning_rate, lazy=False)
    # runner.run(duration=30, interval=0, name='osvr', plot=True, record=True, verbose=True)
    # '''best-0.0425'''

    #olstm
    # runner = Runner(warmStartPoint=100, dataset=dataset_olstm, model=olstm, deep=True, learningRate=learning_rate, lazy=False)
    # runner.run(duration=1, interval=0, name='olstm', plot=True, record=True, verbose=True)


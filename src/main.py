from .config import LocalConfig
from .runner import Runner
from .models import SVRModel, OSVRModel
from .dataset import Dataset
import logging


if __name__ == "__main__":
    logformat = "%(asctime)s: %(message)s"
    logging.basicConfig(format=logformat, level=logging.INFO,
                        datefmt="%H:%M:%S")

    dataPath = 'datafull.csv'
    modelPath = LocalConfig.modelPath
    features = LocalConfig.features
    shiftRange = LocalConfig.shiftRange
    removeSet = LocalConfig.removeSet

    model = OSVRModel(learning_rate='constant', eta0=0.4,
                      loss='epsilon_insensitive', penalty='l2')

    dataset = Dataset(dataPath=dataPath, shiftFeatures=['meter'],
                      shiftRange=shiftRange, removeSet=removeSet)

    runner = Runner(warmStartPoint=1, dataset=dataset, model=model)
    runner.run(duration=60, interval=0.01)

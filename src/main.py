from .config import LocalConfig
from .runner import Runner
from .models import SVRModel
from .dataset import Dataset
import logging


if __name__ == "__main__":
    logformat = "%(asctime)s: %(message)s"
    logging.basicConfig(format=logformat, level=logging.INFO,
                        datefmt="%H:%M:%S")

    dataPath = LocalConfig.dataPath
    modelPath = LocalConfig.modelPath
    features = LocalConfig.features
    shiftRange = LocalConfig.shiftRange

    model = SVRModel(modelPath=modelPath, kernel='rbf',
                     C=100, gamma=0.04, epsilon=.01)
    dataset = Dataset(dataPath=dataPath, features=features,
                      shiftRange=shiftRange)

    runner = Runner(dataset=dataset, model=model)
    runner.run(duration=1, interval=0.1)

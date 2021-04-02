from .config import LocalConfig
from .runner import *
from .models import SVRModel


if __name__ == "__main__":
    dataPath = LocalConfig.dataPath
    modelPath = LocalConfig.modelPath
    features = LocalConfig.features
    shiftRange = LocalConfig.shiftRange

    model = SVRModel(modelPath=modelPath, kernel='rbf',
                     C=100, gamma=0.04, epsilon=.01)

    runner = Runner(dataPath=dataPath, features=features,
                    shiftRange=shiftRange, model=model)
    runner.run(duration=1, interval=0.1)

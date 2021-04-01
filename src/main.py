from .config import LocalConfig
from .runner import *


if __name__ == "__main__":
    dataPath = LocalConfig.dataPath
    modelPath = LocalConfig.modelPath
    features = LocalConfig.features
    shiftRange = LocalConfig.shiftRange
    model = LocalConfig.model

    runner = Runner(dataPath=dataPath, modelPath=modelPath, features=features,
                    shiftRange=shiftRange, modelName=model, )
    runner.run(duration=1, interval=0.01)

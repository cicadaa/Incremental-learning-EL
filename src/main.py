from .config import LocalConfig
from .runner import *
from .models import SVRModel


if __name__ == "__main__":
    dataPath = LocalConfig.dataPath
    modelPath = LocalConfig.modelPath
    features = LocalConfig.features
    shiftRange = LocalConfig.shiftRange

    model = SVRModel(modelPath=modelPath)

    runner = Runner(dataPath=dataPath, features=features,
                    shiftRange=shiftRange, model=model)
    runner.run(duration=1, interval=0.01)

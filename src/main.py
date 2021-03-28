
from .config import LocalConfig
from .runner import *


# TODO: create a Runner class to handle the logic below
if __name__ == "__main__":
    runner = Runner(LocalConfig)
    runner.run(duration=10, interval=0.2)

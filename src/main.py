from .config import LocalConfig
from .runner import *


if __name__ == "__main__":
    runner = Runner(LocalConfig)
    runner.run(duration=10, interval=0.2)

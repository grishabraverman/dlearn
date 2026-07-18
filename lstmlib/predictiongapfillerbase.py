import logging
from abc import abstractmethod, ABC

log = logging.getLogger(__name__)

class PredictionGapFillerBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get(self, time_stamp:str, features:list) -> float:
        pass
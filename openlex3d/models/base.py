from abc import ABC, abstractmethod
from typing import List


class Model(ABC):
    pass


class VisualLanguageEncoder(Model):
    @abstractmethod
    def compute_text_features(self, input_text: List[str], batch_size=64):
        pass

from abc import ABC, abstractmethod

from datasets import Dataset


class Tuner(ABC):
    @abstractmethod
    def train(self, dataset: Dataset, output_dir: str):
        raise NotImplementedError()
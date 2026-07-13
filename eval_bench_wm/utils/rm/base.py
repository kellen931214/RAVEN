from abc import ABC, abstractmethod


class BaseRemover(ABC):
    @abstractmethod
    def remove(self, image, **kwargs):
        pass

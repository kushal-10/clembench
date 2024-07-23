from abc import ABC, abstractmethod

class BaseVLM(ABC):

    @abstractmethod
    def prepare_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate_outputs(self, *args, **kwargs):
        pass





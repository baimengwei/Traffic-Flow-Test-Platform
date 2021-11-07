from abc import ABCMeta, abstractmethod


class EnvBase(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_agents_info(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def bulk_log(self):
        pass

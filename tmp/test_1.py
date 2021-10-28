from abc import abstractmethod
from abc import ABCMeta

class AbstractMethod(metaclass=ABCMeta):
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def get_name(self):
        pass

    def get_name_(self):
        return self._name

class AbstractImpl(AbstractMethod):
    def get_name_(self):
        return self._name

    def get_name(self):
        return None

if __name__ == '__main__':
    a = AbstractImpl('asd')
    print(a.get_name_())


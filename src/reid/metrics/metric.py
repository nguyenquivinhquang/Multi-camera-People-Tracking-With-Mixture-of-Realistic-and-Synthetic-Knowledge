from pyparsing import Any
from abc import ABC, abstractmethod


class Metric_Interface(object):
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, output):
        pass

    @abstractmethod
    def compute(self):
        return

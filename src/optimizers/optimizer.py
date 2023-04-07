from abc import ABC, abstractmethod


class Optimizer(ABC):
    '''
    Abstract base class of setting out template for optimizer classes.
    '''
    @abstractmethod
    def solve(
        self,
        problem
    ):
        ...

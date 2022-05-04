import numpy as np
from abc import ABC, abstractmethod


class Manifold(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @property
    @abstractmethod
    def initial(self) -> np.ndarray:
        ...

    @abstractmethod
    def metric(self, p: np.ndarray, v: np.ndarray, u: np.ndarray) -> float:
        ...

    @abstractmethod
    def norm(self, p: np.ndarray, v: np.ndarray) -> float:
        ...

    @abstractmethod
    def retraction(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def transport(self, p: np.ndarray, v: np.ndarray, u: np.ndarray, is_scaled: bool=True) -> np.ndarray:  
        ...

    @abstractmethod
    def gradient(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        ...

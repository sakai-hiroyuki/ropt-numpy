import numpy as np
from abc import ABC, abstractmethod


class Manifold(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @property
    @abstractmethod
    def initial(
        self
    ) -> np.ndarray:
        ...

    @abstractmethod
    def metric(
        self,
        point: np.ndarray,
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray
    ) -> float:
        ...

    @abstractmethod
    def norm(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> float:
        ...

    @abstractmethod
    def retraction(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> np.ndarray:
        ...

    @abstractmethod
    def transport(
        self,
        point: np.ndarray,
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray,
        is_scaled: bool=True
    ) -> np.ndarray:  
        ...

    @abstractmethod
    def egrad2rgrad(
        self,
        point: np.ndarray,
        egrad: np.ndarray
    ) -> np.ndarray:
        ...

import numpy as np
from abc import ABC, abstractmethod


class Manifold(ABC):
    @abstractmethod
    def inner_product(
        self,
        point: np.ndarray,
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray
    ) -> float:
        return tangent_vector1 @ tangent_vector2

    @abstractmethod
    def norm(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> float:
        return np.linalg.norm(tangent_vector)

    @abstractmethod
    def retraction(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> np.ndarray:
        return point + tangent_vector

    @abstractmethod
    def vector_transport(
        self,
        point: np.ndarray, 
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray,
        is_scaled: bool=True
    ) -> np.ndarray:  
        return tangent_vector2

    @abstractmethod
    def egrad2rgrad(
        self,
        point: np.ndarray,
        egrad: np.ndarray
    ) -> np.ndarray:
        return egrad

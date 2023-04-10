from math import sin, cos
import numpy as np
from manifolds import Manifold


class Sphere(Manifold):
    def __init__(
        self,
        n: int
    ) -> None:
        self.n = n
    
    def inner_product(
        self,
        point: np.ndarray,
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray
    ) -> float:
        return super().inner_product(point, tangent_vector1, tangent_vector2)
    
    def norm(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> float:
        return super().norm(point, tangent_vector)

    def retraction(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> np.ndarray:
        return (point + tangent_vector) / np.linalg.norm(point + tangent_vector)

    def exponential_map(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> np.ndarray:
        k: float = np.linalg.norm(tangent_vector)
        return cos(k) * point + sin(k) * tangent_vector / k

    def vector_transport(
        self,
        point: np.ndarray, 
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray,
        is_scaled: bool=False
    ) -> np.ndarray:
        _v = point + tangent_vector1
        _p = np.reshape(_v, (self.n, 1)) / np.linalg.norm(_v)
        _proj = np.eye(self.n) - _p @ _p.T
        _res = _proj @ tangent_vector2 / np.linalg.norm(_v)

        scale = 1.
        if is_scaled:
            scale = min(1., np.linalg.norm(tangent_vector2) / np.linalg.norm(_res))
        
        return scale * _res

    def egrad2rgrad(
        self,
        point: np.ndarray,
        egrad: np.ndarray
    ) -> np.ndarray:
        _p = np.reshape(point, (self.n, 1))
        _proj = np.eye(self.n) - _p @ _p.T
        return _proj @ egrad

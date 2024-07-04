from math import sin, cos
import numpy as np
from ropt.manifolds import Manifold


class Sphere(Manifold):
    def __init__(self, dim: int) -> None:
        self._dim = dim

    def __str__(self) -> str:
        return f'{self.dim}-dimensional Sphere'

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def initial(self) -> np.ndarray:
        u = np.array([n for n in range(self.dim + 1)])
        return u / np.linalg.norm(u)

    def metric(
        self,
        point: np.ndarray,
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray
    ) -> float:
        return tangent_vector1 @ tangent_vector2

    def norm(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> float:
        return np.sqrt(self.metric(point, tangent_vector, tangent_vector))

    def retraction(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> np.ndarray:
        return (point + tangent_vector) / np.linalg.norm(point + tangent_vector)
        
    def transport(
        self,
        point: np.ndarray,
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray,
        is_scaled: bool=True
    ) -> np.ndarray:
        v = point + tangent_vector1
        _point = np.reshape(v, (point.size, 1))
        direction: np.ndarray = (np.identity(point.size) - (_point @ _point.T) / (v @ v)) @ tangent_vector2

        if is_scaled:
            scale: float = min(1.0, np.linalg.norm(tangent_vector2) / np.linalg.norm(direction))
            return scale * direction
        else:
            return direction

    def egrad2rgrad(
        self,
        point: np.ndarray,
        egrad: np.ndarray
    ) -> np.ndarray:
        _point = np.reshape(point, (point.size, 1))
        return egrad - np.dot(np.dot(_point, _point.T), egrad)

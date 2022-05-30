from math import sin, cos
import numpy as np
from ropt.manifolds import Manifold

from math import sqrt


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

    def metric(self, p: np.ndarray, v: np.ndarray, u: np.ndarray) -> float:
        return np.dot(v, u)

    def norm(self, p: np.ndarray, v: np.ndarray) -> float:
        return sqrt(self.metric(p, v, v))

    def retraction(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (p + v) / np.linalg.norm(p + v)
    
    def transport(self, p: np.ndarray, v: np.ndarray, u: np.ndarray, is_scaled: bool=True) -> np.ndarray:   
        w = np.reshape(p + v, (p.size, 1))
        direction: np.ndarray = np.dot(np.identity(p.size) - np.dot(w, w.T) / np.dot(p + v, p + v), u)

        if is_scaled:
            scale: float = min(1.0, np.linalg.norm(u) / np.linalg.norm(direction))
            return scale * direction
        else:
            return direction

    def gradient(self, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        w = np.reshape(p, (p.size, 1))
        return g - np.dot(np.dot(w, w.T), g)
    
    def exp(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        k: float = np.linalg.norm(v)
        return cos(k) * p + sin(k) * v / k

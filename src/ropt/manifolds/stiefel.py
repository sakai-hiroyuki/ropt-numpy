import numpy as np

from ropt.manifolds import Manifold


class Stiefel(Manifold):
    def __init__(self, p: int, n: int) -> None:
        self._p = p
        self._n = n

    def __str__(self) -> str:
        return f'Stiefel Manifold St({self._p},{self._n})'

    @property
    def dim(self) -> int:
        return int(self._n * self._p - self._p * (self._p + 1) / 2)

    @property
    def initial(self) -> np.ndarray:
        return qf(np.ones((self._n, self._p)))

    def metric(
        self,
        point: np.ndarray,
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray
    ) -> float:
        return np.trace(tangent_vector1.T @ tangent_vector2)

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
        return qf(point + tangent_vector)

    def transport(
        self,
        point: np.ndarray,
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray,
        is_scaled: bool=True
    ) -> np.ndarray:
        
        v: np.ndarray = Dqf(point + tangent_vector1, tangent_vector2)

        if is_scaled:
            r = self.norm(point, tangent_vector2) / self.norm(self.retraction(point, tangent_vector1), v)
            scale = min(1.0, r)
            return scale * v
        else:
            return v
    
    def egrad2rgrad(
        self,
        point: np.ndarray,
        egrad: np.ndarray
    ) -> np.ndarray:
        return point @ skew(point.T @ egrad) + (np.identity(self._n) - point @ point.T) @ egrad


def Dqf(
    tangent_vector1: np.ndarray,
    tangent_vector2: np.ndarray
) -> np.ndarray:
    n = tangent_vector1.shape[0]

    w = tangent_vector2 @ np.linalg.inv(qf(tangent_vector1).T @ tangent_vector1)

    v = np.identity(n) - qf(tangent_vector1) @ qf(tangent_vector1).T
    v = v @ w
    v = v + qf(tangent_vector1) @ rhoskew(qf(tangent_vector1).T @ w)
    return v


def qf(m: np.ndarray) -> np.ndarray:
    return np.linalg.qr(m)[0]


def rhoskew(m: np.ndarray) -> np.ndarray:
    tril = np.tril(np.ones_like(m))
    return tril * m - (tril * m).T


def sym(m: np.ndarray) -> np.ndarray:
    return (m + m.T) / 2


def skew(m: np.ndarray) -> np.ndarray:
    return (m - m.T) / 2

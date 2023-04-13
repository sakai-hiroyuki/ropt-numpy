import numpy as np
from manifolds import Manifold


class Stiefel(Manifold):
    def __init__(
        self,
        p: int,
        n: int
    ) -> None:
        self.p: int = p
        self.n: int = n
        self._d: np.ndarray = np.array([[1 if i > j else 0 if i == j else -1 for j in range(p)] for i in range(p)])

    @property
    def dim(self) -> int:
        return int(self.n * self.p - self.p * (self.p + 1) / 2)

    def inner_product(
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
        return np.sqrt(np.trace(tangent_vector.T @ tangent_vector))

    def retraction(
        self,
        point: np.ndarray,
        tangent_vector: np.ndarray
    ) -> np.ndarray:
        return self._qf(point + tangent_vector)

    def vector_transport(
        self,
        point: np.ndarray, 
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray,
        is_scaled: bool=True
    ) -> np.ndarray:
        return self._differentiated_qf(point + tangent_vector1, tangent_vector2)

    def egrad2rgrad(
        self,
        point: np.ndarray,
        egrad: np.ndarray
    ) -> np.ndarray:
        return egrad - point @ self._sym(point.T @ egrad)
    
    def _qf(
        self,
        matrix: np.ndarray
    ) -> np.ndarray:
        return np.linalg.qr(matrix)[0]
    
    def _sym(
        self,
        matrix: np.ndarray
    ) -> np.ndarray:
        return (matrix + matrix.T) / 2

    def _skew(
        self,
        matrix: np.ndarray
    ) -> np.ndarray:
        return (matrix - matrix.T) / 2

    def _differentiated_qf(
        self,
        tangent_vector1: np.ndarray,
        tangent_vector2: np.ndarray,
    ) -> np.ndarray:
        _q = self._qf(tangent_vector1)
        _s = tangent_vector2 @ np.linalg.inv(_q.T @ tangent_vector1)
        _en = np.eye(self.n)

        t1 = _q @ (self._d * (_q.T @ _s))
        t2 = (_en - _q @ _q.T) @ _s
        
        _res = t1 + t2
        return _res

import numpy as np
import numpy.linalg as la
from math import sqrt

from ropt.manifolds import Manifold


class Stiefel(Manifold):
    def __init__(self, p: int, n: int) -> None:
        self._p = p
        self._n = n

    def __str__(self) -> str:
        return 'Stiefel Manifold St({0},{1})'.format(self._p, self._n)

    @property
    def dim(self) -> int:
        t1 = self._n * self._p
        t2 = self._p * (self._p + 1) / 2
        return int(t1 - t2)

    @property
    def initial(self) -> np.ndarray:
        u = np.ones((self._n, self._p))
        q, _ = la.qr(u)
        return q

    def metric(self, p: np.ndarray, v: np.ndarray, u: np.ndarray) -> float:
        return np.trace(np.dot(v.T, u))

    def norm(self, p: np.ndarray, v: np.ndarray) -> float:
        return sqrt(self.metric(p, v, v))
    
    def retraction(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self._qf(p + v)

    def transport(self, p: np.ndarray, v: np.ndarray, u: np.ndarray, is_scaled: bool=True) -> np.ndarray:
        direction: np.ndarray = self._Dqf(p + v, u)

        if is_scaled:
            scale: float = min(1.0, self.norm(p, u) / self.norm(self.retraction(p, v), direction))
            return scale * direction
        else:
            return direction
    
    def gradient(self, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        w1 = np.dot(p, self._skew(np.dot(p.T, g)))
        w2 = np.dot(np.identity(self._n) - np.dot(p, p.T), g)
        return w1 + w2
    
    def _sym(self, v: np.ndarray) -> np.ndarray:
        return (v + v.T) / 2

    def _skew(self, v: np.ndarray) -> np.ndarray:
        return (v - v.T) / 2

    def _qf(self, v: np.ndarray) -> np.ndarray:
        return la.qr(v)[0]
    
    def _rhoskew(self, v: np.ndarray) -> np.ndarray:
        n: int = v.shape[0]
        w: np.ndarray = v
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    w[i][j] = 0.0
                else:
                    w[i][j] = -v[i][j]
        return w

    def _Dqf(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        w1 = np.dot(self._qf(v), self._rhoskew(np.dot(self._qf(v).T, np.dot(u, la.inv(np.dot(self._qf(v).T, v))))))
        w2 = np.dot(np.identity(self._n) - np.dot(self._qf(v), self._qf(v).T), np.dot(u, la.inv(np.dot(self._qf(v).T, v))))
        return w1 + w2

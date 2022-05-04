import numpy as np
from autograd import grad


class Problem(object):
    def __init__(self, manifold, loss) -> None:
        self.manifold = manifold
        self._loss = loss

    def loss(self, x: np.ndarray) -> np.ndarray:
        return self._loss(x)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        egrad = grad(self._loss)
        return self.manifold.gradient(x, egrad(x))

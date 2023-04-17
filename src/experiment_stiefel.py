from sklearn.datasets import make_spd_matrix
from matplotlib import pyplot as plt
import autograd.numpy as np

from problem import Problem
from manifolds import Stiefel
from optimizers import (SteepestDescent, ConjugateGradient, LinesearchWolfe)


def create_loss(
    A: np.ndarray,
    N: np.ndarray
):
    # The brockett cost function
    def loss(X: np.ndarray) -> float:
        return np.trace(X.T @ A @ X @ N)
    return loss


if __name__ == '__main__':
    n = 50
    p = 10
    max_iter = 1000
    
    A = make_spd_matrix(n)
    N = np.diag(np.random.rand(p))

    manifold = Stiefel(p, n)
    initial_point = np.linalg.qr(np.ones((n, p)))[0]
    
    loss = create_loss(A, N)

    problem = Problem(manifold, loss)

    linesearch = LinesearchWolfe(c1=1e-4, c2=0.999)
    optimizer = SteepestDescent(linesearch)
    y = optimizer.solve(problem, initial_point, max_iter=max_iter)
    x = range(len(y))
    plt.plot(x, y, label='SD')

    betatypes = ['FR', 'DY', 'PRP', 'HS']
    for betatype in betatypes:
        optimizer = ConjugateGradient(linesearch, betatype=betatype)
        y = optimizer.solve(problem, initial_point, max_iter=max_iter)
        x = range(len(y))
        plt.plot(x, y, label=f'CG({betatype})')

    plt.yscale('log')
    plt.legend()
    plt.grid(which='major')
    plt.show()

from typing import Callable
import autograd.numpy as np
from sklearn.datasets import make_spd_matrix

from ropt import Problem
from ropt.utils import rlog_show
from ropt.manifolds import Manifold, Stiefel
from ropt.optimizers import Optimizer, SD, CG, Linesearch, LinesearchWolfe


def create_loss(A, N):
    # The brockett cost function
    def loss(X: np.ndarray) -> float:
        U = np.dot(X.T, np.dot(A, np.dot(X, N)))
        return np.trace(U)

    return loss


if __name__ == '__main__':
    np.random.seed(seed=0)
    p: int = 2
    n: int = 5

    M: Manifold = Stiefel(p, n)
    A: np.ndarray = make_spd_matrix(n)
    N: np.ndarray = np.diag(np.array([n + 1 for n in range(p)]))
    loss: Callable = create_loss(A, N)
    problem: Problem = Problem(M, loss)
    linesearch: Linesearch = LinesearchWolfe()

    cglist = ['FR', 'DY', 'PRP', 'HS', 'Hybrid1', 'Hybrid2']

    opts: list[Optimizer] = []
    
    opts.append(SD(linesearch=linesearch))
    for betype in cglist:
        opt = CG(betype=betype, linesearch=linesearch)
        opts.append(opt)

    results = []

    for opt in opts:
        results.append(opt.solve(problem))

    rlog_show(results)

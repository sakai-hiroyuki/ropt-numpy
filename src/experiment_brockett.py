import autograd.numpy as np
from argparse import ArgumentParser
from sklearn.datasets import make_spd_matrix

from ropt import Problem
from ropt.utils import rlog_show
from ropt.manifolds import Manifold, Stiefel
from ropt.optimizers import Optimizer, SD, CG, Linesearch, LinesearchWolfe


def create_loss(A: np.ndarray, N: np.ndarray):
    # The brockett cost function
    def loss(X: np.ndarray) -> float:
        U = np.dot(X.T, np.dot(A, np.dot(X, N)))
        return np.trace(U)

    return loss


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', default=5, type=int)
    parser.add_argument('-p', default=2, type=int)
    parser.add_argument('-m', '--matrix', default='random', type=str)
    args = parser.parse_args()

    n: int = args.n
    p: int = args.p
    matrix: str = args.matrix

    if matrix == 'random':
        A = make_spd_matrix(n)
    elif matrix == 'diag':
        A = np.diag([np.random.randint(1, 10) for _ in range(n)])
    else:
        raise Exception()

    M: Manifold = Stiefel(p, n)
    N: np.ndarray = np.diag(np.array([n + 1 for n in range(p)]))
    loss = create_loss(A, N)
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

from argparse import ArgumentParser
from sklearn.datasets import make_spd_matrix
import autograd.numpy as np

from ropt import Problem
from ropt.utils import rlog_show
from ropt.manifolds import Sphere
from ropt.optimizers import SD, CG, LinesearchWolfe, Linesearch


def create_loss(A: np.ndarray):
    def loss(x: np.ndarray) -> float:
        return np.dot(x, np.dot(A, x))

    return loss


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--n_vertex', default=100, type=int)
    parser.add_argument('-m', '--matrix', default='random', type=str)
    args = parser.parse_args()

    n: int = args.n_vertex
    matrix: str = args.matrix

    if matrix == 'random':
        A = make_spd_matrix(n)
    elif matrix == 'diag':
        A = np.diag([np.random.randint(1, 10) for _ in range(n)])
    else:
        raise Exception()

    M = Sphere(n - 1)
    loss = create_loss(A)
    problem = Problem(M, loss)

    linesearch: Linesearch = LinesearchWolfe()

    cglist = ['FR', 'DY', 'PRP', 'HS', 'Hybrid1', 'Hybrid2', 'HZ']

    opts = []

    opts.append(SD(linesearch=linesearch))
    for betype in cglist:
        opt = CG(betype=betype, linesearch=linesearch)
        opts.append(opt)

    results = []

    for opt in opts:
        results.append(opt.solve(problem))

    rlog_show(results)
    for result in results:
        result.to_csv()

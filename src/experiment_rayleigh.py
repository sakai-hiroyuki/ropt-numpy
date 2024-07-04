from sklearn.datasets import make_spd_matrix
import autograd.numpy as np

from ropt import Problem
from ropt.utils import log_show
from ropt.manifolds import Sphere
from ropt.optimizers import SD, CG, LinesearchWolfe, Linesearch


def create_loss(A: np.ndarray):
    def loss(x: np.ndarray) -> float:
        return x @ A @ x

    return loss


if __name__ == '__main__':
    n: int = 100
    A = np.diag([k + 1 for k in range(n)])

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

    log_show(results)
    for result in results:
        result.to_csv()

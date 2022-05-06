import autograd.numpy as np

from ropt import Problem
from ropt.utils import rlog_show
from ropt.manifolds import Sphere
from ropt.optimizers import SD, CG, LinesearchWolfe


def create_loss(A):
    def loss(x):
        return np.dot(x, np.dot(A, x))

    return loss


if __name__ == '__main__':
    n = 100

    M = Sphere(n - 1)
    A = np.diag(np.array([n + 1 for n in range(n)]))
    loss = create_loss(A)
    problem = Problem(M, loss)

    wolfe = LinesearchWolfe()

    cglist = ['FR', 'DY', 'PRP', 'HS', 'Hybrid1', 'Hybrid2']

    opts = []

    opts.append(SD(linesearch=wolfe))
    for betype in cglist:
        opt = CG(betype=betype, linesearch=wolfe)
        opts.append(opt)

    results = []

    for opt in opts:
        results.append(opt.solve(problem))

    rlog_show(results)
    for result in results:
        result.to_csv()

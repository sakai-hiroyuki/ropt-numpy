from argparse import ArgumentParser
from sklearn.datasets import make_spd_matrix
from matplotlib import pyplot as plt
import autograd.numpy as np

from problem import Problem
from manifolds import Sphere
from optimizers import (SteepestDescent, ConjugateGradient, MemorylessQuasiNewton, LinesearchWolfe, LinesearchArmijo)


def create_loss(A: np.ndarray):
    def loss(x: np.ndarray) -> float:
        return np.dot(x, np.dot(A, x))

    return loss


if __name__ == '__main__':
    n = 100
    max_iter = 1000
    A = np.diag([k + 1 for k in range(n)])
    # A = make_spd_matrix(n)
    manifold = Sphere(n)
    initial_point = np.random.rand(n) / np.sqrt(n)

    loss = create_loss(A)
    problem = Problem(manifold, loss)

    linesearch = LinesearchWolfe(c1=1e-4, c2=0.9)

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

    optimizer = MemorylessQuasiNewton(linesearch)
    y = optimizer.solve(problem, initial_point, max_iter=max_iter)
    x = range(len(y))
    plt.plot(x, y, label='memoryless BFGS')

    plt.yscale('log')
    plt.legend()
    plt.grid(which='major')
    plt.show()

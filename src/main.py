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
    _A = np.random.randn(n, n)
    A = (_A + _A.T) / 2
    manifold = Sphere(n)
    initial_point = np.ones(n) / np.sqrt(n)

    loss = create_loss(A)
    problem = Problem(manifold, loss)

    linesearch = LinesearchWolfe()

    optimizer = SteepestDescent(linesearch)
    y = optimizer.solve(problem, initial_point)
    x = range(len(y))
    plt.plot(x, y)

    optimizer = ConjugateGradient(linesearch, betatype='FR')
    y = optimizer.solve(problem, initial_point)
    x = range(len(y))
    plt.plot(x, y)

    optimizer = ConjugateGradient(linesearch, betatype='DY')
    y = optimizer.solve(problem, initial_point)
    x = range(len(y))
    plt.plot(x, y)

    optimizer = MemorylessQuasiNewton(linesearch)
    y = optimizer.solve(problem, initial_point)
    x = range(len(y))
    plt.plot(x, y)

    plt.yscale('log')
    plt.show()


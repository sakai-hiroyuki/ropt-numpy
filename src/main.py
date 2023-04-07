from argparse import ArgumentParser
from sklearn.datasets import make_spd_matrix
from matplotlib import pyplot as plt
import autograd.numpy as np

from problem import Problem
from manifolds import Sphere
from optimizers import SteepestDescent, ConjugateGradient, LinesearchArmijo


def create_loss(A: np.ndarray):
    def loss(x: np.ndarray) -> float:
        return np.dot(x, np.dot(A, x))

    return loss


if __name__ == '__main__':
    n = 20
    # A = np.diag([k + 1 for k in range(n)])
    A = make_spd_matrix(n)
    manifold = Sphere(n)
    initial_point = np.ones(n) / np.sqrt(n)

    loss = create_loss(A)
    problem = Problem(manifold, loss)

    linesearch = LinesearchArmijo()

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

    plt.yscale('log')
    plt.show()


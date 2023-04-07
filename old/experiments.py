import autograd.numpy as np
import networkx as nx
import pandas as pd
from sklearn.datasets import make_spd_matrix

from ropt import Problem
from ropt.utils import performance_profile
from ropt.manifolds import Sphere
from ropt.optimizers import SD, CG, LinesearchWolfe, Linesearch


def create_loss_stability(G: nx.Graph):
    def loss(x: np.ndarray) -> float:
        e0 = [e[0] for e in G.edges()]
        e1 = [e[1] for e in G.edges()]
        x0 = x[e0] ** 2
        x1 = x[e1] ** 2
        return np.sum(x ** 4) + 2 * np.dot(x0, x1)

    return loss


def create_loss_rayleigh(A: np.ndarray):
    def loss(x: np.ndarray) -> float:
        return np.dot(x, np.dot(A, x))

    return loss


if __name__ == '__main__':
    n: int = 100
    p: float = 0.1

    results1 = []
    results2 = []
    for _ in range(100):
        G: nx.Graph = nx.fast_gnp_random_graph(n = n, p = 0.25)
        M = Sphere(n - 1)
        loss = create_loss_stability(G)
        problem = Problem(M, loss)
        linesearch: Linesearch = LinesearchWolfe()
        cglist = ['FR', 'DY', 'PRP', 'HS', 'Hybrid1', 'Hybrid2', 'HZ']
        opts = []
        opts.append(SD(linesearch=linesearch, max_iter=1000))
        for betype in cglist:
            opt = CG(betype=betype, linesearch=linesearch, max_iter=1000)
            opts.append(opt)
        row1 = []
        row2 = []
        for opt in opts:
            rlog = opt.solve(problem)
            row1.append(rlog.n_iter)
            row2.append(rlog.log[-1][2])
        results1.append(row1)
        results2.append(row2)
    
    for _ in range(100):
        A = make_spd_matrix(n)
        M = Sphere(n - 1)
        loss = create_loss_rayleigh(A)
        problem = Problem(M, loss)
        linesearch: Linesearch = LinesearchWolfe()
        cglist = ['FR', 'DY', 'PRP', 'HS', 'Hybrid1', 'Hybrid2', 'HZ']
        opts = []
        opts.append(SD(linesearch=linesearch, max_iter=1000))
        for betype in cglist:
            opt = CG(betype=betype, linesearch=linesearch, max_iter=1000)
            opts.append(opt)
        row1 = []
        row2 = []
        for opt in opts:
            rlog = opt.solve(problem)
            row1.append(rlog.n_iter)
            row2.append(rlog.log[-1][2])
        results1.append(row1)
        results2.append(row2)

    df1 = pd.DataFrame(results1, columns=['SD', 'FR', 'DY', 'PRP', 'HS', 'Hybrid1', 'Hybrid2', 'HZ'])
    df1.to_csv('./output/result1.csv', index=None)
    performance_profile('./output/result1.csv')

    df2 = pd.DataFrame(results2, columns=['SD', 'FR', 'DY', 'PRP', 'HS', 'Hybrid1', 'Hybrid2', 'HZ'])
    df2.to_csv('./output/result2.csv', index=None)
    performance_profile('./output/result2.csv')

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

    results = []
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
        row = []
        for opt in opts:
            row.append(opt.solve(problem).n_iter)
        results.append(row)
    
    for _ in range(100):
        A = make_spd_matrix(n)
        M = Sphere(n - 1)
        loss = create_loss_rayleigh(A)
        problem = Problem(M, loss)
        linesearch: Linesearch = LinesearchWolfe()
        cglist = ['FR', 'DY', 'PRP', 'HS', 'Hybrid1', 'Hybrid2', 'HZ']
        opts = []
        opts.append(SD(linesearch=linesearch))
        for betype in cglist:
            opt = CG(betype=betype, linesearch=linesearch)
            opts.append(opt)
        row = []
        for opt in opts:
            row.append(opt.solve(problem).n_iter)
        results.append(row)

    df = pd.DataFrame(results, columns=['SD', 'FR', 'DY', 'PRP', 'HS', 'Hybrid1', 'Hybrid2', 'HZ'])
    df.to_csv('./output/result.csv', index=None)
    performance_profile('./output/result.csv')

from argparse import ArgumentParser
import autograd.numpy as np
import networkx as nx

from ropt import Problem
from ropt.utils import rlog_show
from ropt.manifolds import Sphere
from ropt.optimizers import SD, CG, LinesearchWolfe, Linesearch


def create_loss(G: nx.Graph):
    def loss(x: np.ndarray) -> float:
        e0 = [e[0] for e in G.edges()]
        e1 = [e[1] for e in G.edges()]
        x0 = x[e0] ** 2
        x1 = x[e1] ** 2
        return np.sum(x ** 4) + 2 * np.dot(x0, x1)

    return loss


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--n_vertex', default=100, type=int)
    parser.add_argument('-p', '--probability', default=0.25, type=float)
    parser.add_argument('-g', '--graph', default='random', type=str)
    args = parser.parse_args()

    n: int = args.n_vertex
    p: float = args.probability
    graph: str = args.graph

    if graph == 'random':
        G: nx.Graph = nx.fast_gnp_random_graph(n = n, p = 0.25)
    elif graph == 'cycle':
        G: nx.Graph = nx.cycle_graph(n)
    else:
        raise Exception()

    M = Sphere(n - 1)
    loss = create_loss(G)
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

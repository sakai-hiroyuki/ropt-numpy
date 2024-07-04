import numpy as np

n = 4

A = np.random.randint(1, 10, (n, n))


def rho_skew(A):
    tril = np.tril(np.ones_like(A))
    return tril * A - (tril * A).T

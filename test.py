import numpy as np
from math import pi


if __name__ == '__main__':
    n = 3
    d = [[1 if i > j else 0 if i == j else -1 for j in range(n)] for i in range(n)]

    v = np.linalg.qr(np.random.rand(5, 2))[0]
    print(v)
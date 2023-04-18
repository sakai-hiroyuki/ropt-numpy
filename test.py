import numpy as np

x = np.array([1., 1., 1.]) / np.sqrt(3)
Bx = np.array([
    [1 / np.sqrt(2), 1 / np.sqrt(6)],
    [-1 / np.sqrt(2), 1 / np.sqrt(6)],
    [0., -2 / np.sqrt(6)]
])

y = np.array([0., 0., 1.])
By = np.array([
    [1., 0.],
    [0., -1.],
    [0., 0.]
])

z = np.array([0., 1., 0.])
Bz = np.array([
    [1., 0.],
    [0., 0.],
    [0., 1.]
])

v = np.array([2, 0, 3])
print(np.linalg.norm(By @ Bz.T @ v))
print(np.linalg.norm(v))

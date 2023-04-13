import numpy as np

d = 5

I = np.eye(d)

x = np.random.randn(d)
px = np.reshape(x, (d, 1))

y = np.random.randn(d)
py = np.reshape(y, (d, 1))

alpha = 3
beta = 4

H = I + alpha * (px @ px.T) + beta * (py @ py.T)

xnorm = np.linalg.norm(x)
ynorm = np.linalg.norm(y)

det = 1 + alpha * xnorm ** 2 + beta * ynorm ** 2 + alpha * beta * (xnorm ** 2 * ynorm ** 2 - (x @ y) ** 2)

print(np.linalg.eig(H)[0])
print(np.linalg.det(H), det)


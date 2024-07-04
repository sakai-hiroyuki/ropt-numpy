import time
from typing import Optional
from ropt.utils import Logger
from ropt.optimizers import Optimizer, LinesearchArmijo, Linesearch


class SD(Optimizer): 
    '''
    Steepest Descent Method

    ----------
    Parameters
    ----------
    linesearch : Linesearch=None
        Specify the line search algorithm when calculating the step size.
    
    ----------
    References
    ----------
    [1] Absil, P-A., Robert Mahony, and Rodolphe Sepulchre. Optimization algorithms on matrix manifolds.
        Princeton University Press, 2009.
    '''
    def __init__(
            self,
            linesearch: Linesearch,
            name: str=None,
            max_iter: int=300,
            min_gn: float=1e-6
        ) -> None:

        if linesearch is None:
            self.linesearch = LinesearchArmijo()
        else:
            self.linesearch = linesearch

        if name is None:
            name = 'SD'
        super(SD, self).__init__(name=name, max_iter=max_iter, min_gn=min_gn)

    def solve(
        self,
        problem
    ) -> Logger:
        
        M = problem.manifold
        xk = M.initial

        logger = Logger(name=str(self))

        self._tic = time.time()
        t: float = 0.
        k: int = 0
        while True:
            k = k + 1
            g = problem.gradient(xk)
            grad_norm = M.norm(xk, g)
            v = problem.loss(xk)

            logger.append(grad_norm, v, t)
            t = time.time() - self._tic
    
            d = -g
            step = self.linesearch(problem, xk, d)

            if k == 1 or k % 100 == 0:
                print(f'{k:>5}, {v:.5e}, {grad_norm:.5e}, {t:.3f}')

            if self.stop(k, grad_norm):
                break

            xk = M.retraction(xk, step * d)

        return logger
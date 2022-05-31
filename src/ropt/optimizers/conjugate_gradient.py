import time
import numpy as np
from ropt.utils import RoptLogger
from ropt.optimizers import Optimizer, LinesearchArmijo, Linesearch


class CG(Optimizer):
    '''
    Conjugate Gradient Method

    ----------
    Parameters
    ----------
    betype: str='FR'
        Select the suitable method from 'FR' (see [2]), 'DY' (see [3]),
        'PRP', 'PRP+', 'HS', 'HS+', 'Hybrid1', 'Hybrid2' and 'HZ'.
    linesearch :
        Specify the line search algorithm when calculating the step size.
    
    ----------
    References
    ----------
    [1] Absil, P-A., Robert Mahony, and Rodolphe Sepulchre. Optimization algorithms on matrix manifolds.
        Princeton University Press, 2009.
    [2] Sato, Hiroyuki, and Toshihiro Iwai. "A new, globally convergent Riemannian conjugate gradient method."
        Optimization 64.4 (2015): 1011-1031.
    [3] Sato, Hiroyuki. "A Dai-Yuan-type Riemannian conjugate gradient method with the weak Wolfe conditions."
        Computational Optimization and Applications 64.1 (2016): 101-118.
    '''
    def __init__(
        self,
        betype: str='FR',
        linesearch: Linesearch=None,
        reset: bool=True,
        name: str=None,
        max_iter: int=300,
        min_gn: float=1e-6
    ) -> None:

        self.betype = betype
        if linesearch is None:
            self.linesearch = LinesearchArmijo()
        else:
            self.linesearch = linesearch
        self.reset = reset

        if name is None:
            name = 'CG(' + self.betype + ')'
        super(CG, self).__init__(name=name, max_iter=max_iter, min_gn=min_gn)

    def solve(self, problem) -> RoptLogger:
        M = problem.manifold
        xk = M.initial

        rlogger = RoptLogger(name=str(self))
        
        g: np.ndarray = problem.gradient(xk)
        d: np.ndarray = -g

        self._tic = time.time()
        t: float = 0.
        k: int = 0
        while True:
            k = k + 1

            g = problem.gradient(xk)
            gn = M.norm(xk, g)
            v = problem.loss(xk)
            rlogger.writelog(gn=gn, val=v, time=t)
            t = time.time() - self._tic

            if self.reset and M.metric(xk, d, g) > 0:
                d = -g
            
            step = self.linesearch(problem, xk, d)

            if k == 1 or k % 100 == 0:
                print(f'{k:>5}, {v:.5e}, {gn:.5e}, {t:.3f}')

            if self.stop(k, gn):
                break

            xknew = M.retraction(xk, step * d)
            gnew = problem.gradient(xknew)

            beta = None
            try:
                if self.betype == 'FR':
                    beta = _compute_FR(M, d, step, xk, g, xknew, gnew)
                elif self.betype == 'DY':
                    beta = _compute_DY(M, d, step, xk, g, xknew, gnew)
                elif self.betype == 'PRP':
                    beta = _compute_PRP(M, d, step, xk, g, xknew, gnew)
                elif self.betype == 'PRP+':
                    betaPRP = _compute_PRP(M, d, step, xk, g, xknew, gnew)
                    beta = max(0.0, betaPRP)
                elif self.betype == 'HS':
                    beta = _compute_HS(M, d, step, xk, g, xknew, gnew)
                elif self.betype == 'HS+':
                    betaHS = _compute_HS(M, d, step, xk, g, xknew, gnew)
                    beta = max(0.0, betaHS)
                elif self.betype == 'Hybrid1':
                    betaDY = _compute_DY(M, d, step, xk, g, xknew, gnew)
                    betaHS = _compute_HS(M, d, step, xk, g, xknew, gnew)
                    beta = max(0, min(betaDY, betaHS))
                elif self.betype == 'Hybrid2':
                    betaFR = _compute_FR(M, d, step, xk, g, xknew, gnew)
                    betaPRP = _compute_PRP(M, d, step, xk, g, xknew, gnew)
                    beta = max(0, min(betaFR, betaPRP))
                elif self.betype == 'HZ':
                    beta = _compute_HZ(M, d, step, xk, g, xknew, gnew)
                else:
                    raise Exception(f'Exception: Unknown beta type: {self.betype}')
            except Exception as e:
                print(e)

            d = -gnew + beta * M.transport(xk, step * d, d)
            xk = xknew

        return rlogger


def _compute_FR(M, d, step, xk, g, xknew, gnew):
    _div = M.metric(xk, g, g)
    z = gnew / _div

    return M.metric(xknew, gnew, z)


def _compute_DY(M, d, step, xk, g, xknew, gnew):
    _div = M.metric(xknew, gnew, M.transport(xk, step * d, d)) - M.metric(xk, g, d)
    z = gnew / _div
    
    return M.metric(xknew, gnew, z)


def _compute_PRP(M, d, step, xk, g, xknew, gnew):
    _y = gnew - M.transport(xk, step * d, g)
    _div = M.metric(xk, g, g)
    z = _y / _div
    
    return M.metric(xknew, gnew, z)


def _compute_HS(M, d, step, xk, g, xknew, gnew):
    _y = gnew - M.transport(xk, step * d, g)
    _div = M.metric(xknew, gnew, M.transport(xk, step * d, d)) - M.metric(xk, g, d)
    z = _y / _div
    
    return M.metric(xknew, gnew, z)


def _compute_HZ(M, d, step, xk, g, xknew, gnew, mu: float=2.0):
    if not mu > 1/4:
        raise ValueError(f'Invalid value: mu = {mu}.')
    
    _y = gnew - M.transport(xk, step * d, g)
    _gd = M.metric(xk, g, d)
    _gnewd = M.metric(xknew, gnew, M.transport(xk, step * d, d))
    _div = (_gnewd - _gd) ** 2
    _m = M.norm(xknew, _y) ** 2
    z = _m * _gnewd / _div
    return _compute_HS(M, d, step, xk, g, xknew, gnew) - mu * z

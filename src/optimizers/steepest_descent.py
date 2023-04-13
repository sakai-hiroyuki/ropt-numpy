import time
import numpy as np
from optimizers import Optimizer, Linesearch


class SteepestDescent(Optimizer): 
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
        linesearch: Linesearch
    ) -> None:
        self.linesearch = linesearch

    def solve(
        self,
        problem,
        initial_point: np.ndarray,
        max_iter: int=1000
    ) -> list[float]:
        
        manifold = problem.manifold
        point = initial_point

        history = []

        for _ in range(max_iter):
            rgrad = problem.gradient(point)
            rgrad_norm = manifold.norm(point, rgrad)
            history.append(rgrad_norm)
            if rgrad_norm <= 1e-6:
                break
            descent_direction = -rgrad
            step = self.linesearch.search(problem, point, descent_direction)
            point = manifold.retraction(point, step * descent_direction)
        
        return history

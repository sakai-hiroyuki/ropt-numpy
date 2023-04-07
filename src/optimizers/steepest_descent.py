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
        initial_point: np.ndarray
    ) -> list[float]:
        
        manifold = problem.manifold
        point = initial_point

        history = []

        for _ in range(300):
            rgrad = problem.gradient(point)
            history.append(manifold.norm(point, rgrad))
            descent_direction = -rgrad
            step = self.linesearch.search(problem, point, descent_direction)
            point = manifold.retraction(point, step * descent_direction)
        
        return history

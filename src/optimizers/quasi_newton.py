import numpy as np
from optimizers import Optimizer, Linesearch
from manifolds import Manifold


class MemorylessQuasiNewton(Optimizer):
    def __init__(
        self,
        linesearch: Linesearch,
    ) -> None:
        self.linesearch = linesearch

    def solve(
        self,
        problem,
        initial_point: np.ndarray
    ) -> list[float]:
        
        manifold: Manifold = problem.manifold
        point: np.ndarray = initial_point

        history = []

        rgrad = problem.gradient(point)
        descent_direction = -rgrad

        for _ in range(300):
            rgrad_norm = manifold.norm(point, rgrad)
            history.append(rgrad_norm)
            if rgrad_norm <= 1e-6:
                break
            
            step: float = self.linesearch.search(problem, point, descent_direction)

            point_next = manifold.retraction(point, step * descent_direction)
            rgrad_next = problem.gradient(point_next)

            y = rgrad_next - manifold.vector_transport(point, step * descent_direction, rgrad)
            s = manifold.vector_transport(point, step * descent_direction, step * descent_direction)

            w = s / (s @ y) - y / (y @ y)

            descent_direction = - rgrad_next - (y @ rgrad_next) / (y @ y) * y + (s @ rgrad_next) / (s @ y) * s + (y @ y) * (w @ rgrad_next) * w
            
            point = point_next
            rgrad = rgrad_next

        return history

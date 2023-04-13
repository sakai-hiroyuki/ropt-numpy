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
        initial_point: np.ndarray,
        max_iter: int=1000
    ) -> list[float]:
        
        manifold: Manifold = problem.manifold
        point: np.ndarray = initial_point

        history = []

        rgrad = problem.gradient(point)
        descent_direction = -rgrad

        for _ in range(max_iter):
            rgrad_norm = manifold.norm(point, rgrad)
            history.append(rgrad_norm)
            if rgrad_norm <= 1e-6:
                break
            
            step: float = self.linesearch.search(problem, point, descent_direction)

            point_next = manifold.retraction(point, step * descent_direction)
            rgrad_next = problem.gradient(point_next)

            rgrad_sub = rgrad_next - manifold.vector_transport(point, step * descent_direction, rgrad)
            s = manifold.vector_transport(point, step * descent_direction, step * descent_direction)

            w = s / (s @ rgrad_sub) - rgrad_sub / (rgrad_sub @ rgrad_sub)

            mu = (s @ s) * (rgrad_sub @ rgrad_sub) / (s @ rgrad_sub) ** 2
            theta = max(1 / (1 - mu), -1e-5)
            phi = ((0.1 * theta) - 1) / (0.1 * theta * (1 - mu) - 1)

            descent_direction = - rgrad_next
            descent_direction += - (rgrad_sub @ rgrad_next) / (rgrad_sub @ rgrad_sub) * rgrad_sub
            descent_direction += (s @ rgrad_next) / (s @ rgrad_sub) * s
            descent_direction += phi * (rgrad_sub @ rgrad_sub) * (w @ rgrad_next) * w

            point = point_next
            rgrad = rgrad_next

        return history

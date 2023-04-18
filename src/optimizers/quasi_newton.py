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

            if manifold.inner_product(point, descent_direction, rgrad) >= 0:
                descent_direction = -rgrad
            
            step: float = self.linesearch.search(problem, point, descent_direction)

            point_next = manifold.retraction(point, step * descent_direction)
            rgrad_next = problem.gradient(point_next)

            y = rgrad_next - manifold.vector_transport(point, step * descent_direction, rgrad)
            s = manifold.vector_transport(point, step * descent_direction, step * descent_direction)

            y_dot_y = manifold.inner_product(point_next, y, y)
            s_dot_y = manifold.inner_product(point_next, s, y)

            w = s / s_dot_y - y / y_dot_y

            descent_direction = - rgrad_next
            descent_direction += - manifold.inner_product(point_next, y, rgrad_next) / y_dot_y * y
            descent_direction += manifold.inner_product(point_next, s, rgrad_next) / s_dot_y * s
            descent_direction += y_dot_y * manifold.inner_product(point_next, w, rgrad_next) * w

            point = point_next
            rgrad = rgrad_next

        return history

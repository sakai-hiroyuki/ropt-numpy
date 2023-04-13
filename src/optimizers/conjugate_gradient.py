import numpy as np
from optimizers import Optimizer, Linesearch


class ConjugateGradient(Optimizer):
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
        linesearch: Linesearch,
        betatype: str='FR'
    ) -> None:
        self.linesearch = linesearch
        self.betatype = betatype

    def solve(
        self,
        problem,
        initial_point: np.ndarray,
        max_iter: int=1000
    ) -> list[float]:
        
        manifold = problem.manifold
        point = initial_point

        history = []

        rgrad = problem.gradient(point)
        descent_direction = -rgrad

        for _ in range(max_iter):
            rgrad_norm = manifold.norm(point, rgrad)
            history.append(rgrad_norm)
            if rgrad_norm <= 1e-6:
                break
            
            step = self.linesearch.search(problem, point, descent_direction)

            point_next = manifold.retraction(point, step * descent_direction)
            rgrad_next = problem.gradient(point_next)

            beta = None
            try:
                if self.betatype == 'FR':
                    beta = _compute_FR(manifold, descent_direction, step, point, rgrad, point_next, rgrad_next)
                elif self.betatype == 'DY':
                    beta = _compute_DY(manifold, descent_direction, step, point, rgrad, point_next, rgrad_next)
                elif self.betatype == 'PRP':
                    beta = _compute_PRP(manifold, descent_direction, step, point, rgrad, point_next, rgrad_next)
                elif self.betatype == 'PRP+':
                    beta_PRP = _compute_PRP(manifold, descent_direction, step, point, rgrad, point_next, rgrad_next)
                    beta = max(0., beta_PRP)
                elif self.betatype == 'HS':
                    beta = _compute_HS(manifold, descent_direction, step, point, rgrad, point_next, rgrad_next)
                elif self.betatype == 'HS+':
                    beta_HS = _compute_HS(manifold, descent_direction, step, point, rgrad, point_next, rgrad_next)
                    beta = max(0., beta_HS)
                elif self.betatype == 'Hybrid1':
                    beta_DY = _compute_DY(manifold, descent_direction, step, point, rgrad, point_next, rgrad_next)
                    beta_HS = _compute_HS(manifold, descent_direction, step, point, rgrad, point_next, rgrad_next)
                    beta = max(0., min(beta_DY, beta_HS))
                elif self.betatype == 'Hybrid2':
                    beta_FR = _compute_FR(manifold, descent_direction, step, point, rgrad, point_next, rgrad_next)
                    beta_PRP = _compute_PRP(manifold, descent_direction, step, point, rgrad, point_next, rgrad_next)
                    beta = max(0., min(beta_FR, beta_PRP))
                else:
                    raise Exception(f'Exception: Unknown beta type: {self.betype}')
            except Exception as e:
                print(e)

            descent_direction = -rgrad_next + beta * manifold.vector_transport(point, step * descent_direction, descent_direction)
            
            point = point_next
            rgrad = rgrad_next

        return history


def _compute_FR(
    manifold,
    descent_direction,
    step,
    point,
    rgrad,
    point_next,
    rgrad_next
) -> float:
    
    _numer = manifold.inner_product(point_next, rgrad_next, rgrad_next)
    _denom = manifold.inner_product(point, rgrad, rgrad)
    return _numer / _denom


def _compute_DY(
    manifold,
    descent_direction,
    step,
    point,
    rgrad,
    point_next,
    rgrad_next
) -> float:
    
    _numer = manifold.inner_product(point_next, rgrad_next, rgrad_next)
    _transported_d = manifold.vector_transport(point, step * descent_direction, descent_direction)
    _derphi: float = manifold.inner_product(point, rgrad, descent_direction)
    _denom: float = manifold.inner_product(point_next, rgrad_next, _transported_d) - _derphi
    return _numer / _denom


def _compute_PRP(
    manifold,
    descent_direction,
    step,
    point,
    rgrad,
    point_next,
    rgrad_next
) -> float:
    _rgrad_sub = rgrad_next - manifold.vector_transport(point, step * descent_direction, rgrad)
    _numer = manifold.inner_product(point_next, rgrad_next, _rgrad_sub)
    _denom = manifold.inner_product(point, rgrad, rgrad)
    return _numer / _denom


def _compute_HS(
    manifold,
    descent_direction,
    step,
    point,
    rgrad,
    point_next,
    rgrad_next
) -> float:
    _rgrad_sub = rgrad_next - manifold.vector_transport(point, step * descent_direction, rgrad)
    _numer = manifold.inner_product(point_next, rgrad_next, _rgrad_sub)
    _transported_d = manifold.vector_transport(point, step * descent_direction, descent_direction)
    _derphi: float = manifold.inner_product(point, rgrad, descent_direction)
    _denom: float = manifold.inner_product(point_next, rgrad_next, _transported_d) - _derphi
    return _numer / _denom

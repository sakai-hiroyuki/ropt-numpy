from abc import ABC, abstractmethod
import numpy as np


class Linesearch(ABC):
    @abstractmethod
    def search(self, problem, xk: np.ndarray, d: np.ndarray) -> float:
        ...


class LinesearchArmijo(Linesearch):
    '''
    Armijo Line Search

    ----------
    Parameters
    ----------

    c1 : float=1e-4
        Constant in Armijo rule. Must be 0 < c1 < 1.
    '''
    def __init__(self, c1: float=1e-4) -> None:
        if not 0.0 < c1 < 1.0:
            raise ValueError(f'Invalid value: c1 = {c1}.')
        self.c1 = c1
    
    def __str__(self) -> str:
        return f'Armijo (c1={self.c1})'

    def __call__(self, problem, xk: np.ndarray, d: np.ndarray, initial: float=1., shrinkage: float=0.5) -> float:
        return self.search(problem, xk, d, initial=initial, shrinkage=shrinkage)

    def search(self, problem, xk: np.ndarray, d: np.ndarray, initial: float=1., shrinkage: float=0.5) -> float:
        '''
        Return the step size which satisfies the Armijo condition.

        ----------
        Parameters
        ----------
        problem : Problem
        xk : np.ndarray
            Current point.
        d : np.ndarray
            Search direction.
        upper : float=0.5
            Maximum step size.
        lower : float=1e-6
            Minimum step size. If no suitable step size is found, return this value.
        shrinkage : float=0.5
            Reduction rate (< 1).
        '''
        def phi(alpha) -> float:
            return problem.loss(problem.manifold.retraction(xk, alpha * d))

        def derphi(alpha) -> float: 
            xknew = problem.manifold.retraction(xk, alpha * d)
            dnew = problem.manifold.transport(xk, alpha * d, d)
            return problem.manifold.metric(xknew, problem.gradient(xknew), dnew)

        step = initial
        while phi(step) > phi(0.0) + self.c1 * step * derphi(0.0) and step > 1e-6:
            step *= shrinkage

        return step


class LinesearchWolfe(Linesearch):
    '''
    Strong Wolfe Line Search

    ----------
    Parameters
    ----------

    c1 : float=1e-4
        Constant in Armijo rule. Must be 0 < c1 < 1.
    c2 : float=0.9
        Constant in curvature condition. Must be 0 < c1 < c2 < 1.
    '''
    def __init__(self, c1: float=1e-4, c2: float=0.9):
        if not 0.0 < c1 < c2 < 1.0:
            raise ValueError(f'Invalid values: (c1, c2) = ({c1}, {c2}).')
        self.c1 = c1
        self.c2 = c2

    def __str__(self) -> str:
        return f'strong Wolfe (c1={self.c1}, c2={self.c2})'

    def __call__(self, problem, xk: np.ndarray, d: np.ndarray, maxiter: int=10) -> float:
        return self.search(problem, xk, d, maxiter=maxiter)

    def search(self, problem, xk: np.ndarray, d: np.ndarray, maxiter: int=10) -> float:
        '''
        Returns the step size that satisfies the strong Wolfe condition.
        Scipy.optimize.line_search in SciPy v1.4.1 modified to Riemannian manifold.

        ----------
        Parameters
        ----------
        problem : Problem
        xk : np.ndarray
            Current point.
        d : np.ndarray
            Search direction.
        maxiter : int=10
            Maximum number of iterations to perform (see [1]).
        exsize: float=1e-6
            Returned if the strong Wolfe step sizes are not found.

        ----------
        References
        ----------
        [1] SciPy v1.4.1 Reference Guide, https://docs.scipy.org/
        '''
        fc = [0]
        gc = [0]
        gval = [None]
        gval_alpha = [None]

        def phi(alpha):
            fc[0] += 1
            return problem.loss(problem.manifold.retraction(xk, alpha * d))

        def derphi(alpha):
            xknew = problem.manifold.retraction(xk, alpha * d)
            dnew = problem.manifold.transport(xk, alpha * d, d)
            gc[0] += 1
            gval[0] = problem.gradient(xknew)  # store for later use
            gval_alpha[0] = alpha
            return problem.manifold.metric(xknew, gval[0], dnew)

        gfk = problem.gradient(xk)
        derphi0 = problem.manifold.metric(xk, gfk, d)

        step = _scalar_search_wolfe(phi, derphi, self.c1, self.c2, maxiter=maxiter)
        if step is None:
            step = 1e-6
            # print('The Line Search Algorithm did not converge.')
        return step


def _scalar_search_wolfe(phi, derphi, c1=1e-4, c2=0.9, maxiter=10):
    phi0 = phi(0.)
    derphi0 = derphi(0.)
    alpha0 = 0
    alpha1 = 1.0
    phi_a1 = phi(alpha1)
    phi_a0 = phi0
    derphi_a0 = derphi0
    for i in range(maxiter):
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, phi_star, derphi_star = _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi, derphi, phi0, derphi0, c1, c2)
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= c2 * abs(derphi0)):
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi, derphi, phi0, derphi0, c1, c2)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1
    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None

    return alpha_star


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0, derphi0, c1, c2):
    """
    Part of the optimization algorithm in `_scalar_search_wolfe`.
    """
    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi
        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha
        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= c2 * abs(derphi0):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin

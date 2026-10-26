"""Gradient descent with a constant step for SPD linear systems."""

from linalg import Matrix, Vector, matrix_norm_inf
from solvers.common import IterationState, Options, Result, check_positive_definite


def gradient_descent(a: Matrix, f: Vector, options: Options) -> Result:
    """For an SPD matrix: Phi(x) = (Ax,x)/2 - (f,x), grad Phi = Ax-f.

    The default tau=1/||A||_inf ensures 0<tau<2/lambda_max for an SPD matrix.
    Positive definiteness is checked before execution.
    """
    inapplicable = check_positive_definite("gradient", a)
    if inapplicable is not None:
        return inapplicable
    try:
        state = IterationState("gradient", a, f, options)
        initial = state.initial_result()
        if initial is not None:
            return initial
        step = options.gradient_step if options.gradient_step is not None else 1 / matrix_norm_inf(a)
        r = f[:]
        for _ in range(options.max_iterations):
            x = [xi + step * ri for xi, ri in zip(state.x, r)]
            r, status = state.advance(x)
            if status:
                return state.finish(status, f"Constant step tau={step:.12g}.")
        return state.finish("iteration_limit")
    except (ArithmeticError, ValueError) as error:
        return Result("gradient", "breakdown", detail=f"Numerical arithmetic failed: {error}")

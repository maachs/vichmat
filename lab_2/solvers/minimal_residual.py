"""The one-step minimal residual method with an optimal residual step."""

from linalg import Matrix, Vector, dot, matvec
from solvers.common import IterationState, Options, Result


def minimal_residual(a: Matrix, f: Vector, options: Options) -> Result:
    """One-step iteration: x+=tau*r, tau=(Ar,r)/(Ar,Ar).

    This implements the assignment's one-step minimal residual method.
    Stagnation on an arbitrary matrix is reported to the caller.
    """
    try:
        state = IterationState("minres", a, f, options)
        initial = state.initial_result()
        if initial is not None:
            return initial
        r = f[:]
        for _ in range(options.max_iterations):
            ar = matvec(a, r)
            denominator = dot(ar, ar)
            if denominator == 0.0:
                return state.finish("breakdown", "Ar=0 with a nonzero residual.")
            step = dot(ar, r) / denominator
            if step == 0.0:
                return state.finish("breakdown", "Stagnation: (Ar,r)=0 with a nonzero residual.")
            x = [xi + step * ri for xi, ri in zip(state.x, r)]
            r, status = state.advance(x)
            if status:
                return state.finish(status)
        return state.finish("iteration_limit")
    except (ArithmeticError, ValueError) as error:
        return Result("minres", "breakdown", detail=f"Numerical arithmetic failed: {error}")

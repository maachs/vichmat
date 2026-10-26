"""Unpreconditioned conjugate gradient with true residual monitoring."""

from linalg import Matrix, Vector, dot, matvec, norm
from solvers.common import IterationState, Options, Result, check_positive_definite


def conjugate_gradient(a: Matrix, f: Vector, options: Options) -> Result:
    """Standard CG with a positive definiteness check before iteration."""
    inapplicable = check_positive_definite("cg", a)
    if inapplicable is not None:
        return inapplicable
    try:
        state = IterationState("cg", a, f, options)
        initial = state.initial_result()
        if initial is not None:
            return initial
        r, p = f[:], f[:]
        rr = dot(r, r)
        for _ in range(options.max_iterations):
            ap = matvec(a, p)
            denominator = dot(p, ap)
            if denominator <= 0.0:
                return state.finish("breakdown", "Nonpositive (p,Ap): CG preconditions failed or numerical accuracy was lost.")
            alpha = rr / denominator
            x = [xi + alpha * pi for xi, pi in zip(state.x, p)]
            actual_r, status = state.advance(x)
            if status:
                return state.finish(status)
            new_r = [ri - alpha * api for ri, api in zip(r, ap)]
            if norm(new_r) <= state.threshold:
                # Restart CG if the recursive residual is small but the true residual is not.
                r = actual_r
                p = r[:]
                rr = dot(r, r)
                continue
            new_rr = dot(new_r, new_r)
            beta = new_rr / rr
            p = [ri + beta * pi for ri, pi in zip(new_r, p)]
            r, rr = new_r, new_rr
        return state.finish("iteration_limit")
    except (ArithmeticError, ValueError) as error:
        return Result("cg", "breakdown", detail=f"Numerical arithmetic failed: {error}")

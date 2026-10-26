"""Shared iteration control for stationary coordinate methods."""

from collections.abc import Callable

from linalg import Matrix, Vector
from solvers.common import IterationState, Options, Result


def run_stationary(
    method: str, a: Matrix, f: Vector, options: Options,
    next_iterate: Callable[[Matrix, Vector, Vector, Options], Vector],
) -> Result:
    if any(a[i][i] == 0.0 for i in range(len(f))):
        return Result(
            method, "not_applicable",
            detail="A zero diagonal entry requires a transformation of the system.",
        )
    try:
        state = IterationState(method, a, f, options)
        initial = state.initial_result()
        if initial is not None:
            return initial
        for _ in range(options.max_iterations):
            new = next_iterate(a, f, state.x, options)
            _, status = state.advance(new)
            if status:
                return state.finish(status)
        return state.finish("iteration_limit")
    except (ArithmeticError, ValueError) as error:
        return Result(method, "breakdown", detail=f"Numerical arithmetic failed: {error}")

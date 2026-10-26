"""Doolittle LU factorization followed by two triangular solves."""

from math import fsum

from linalg import (
    Matrix, Vector, backward_substitution, forward_substitution, is_small_pivot,
)
from solvers.base import LinearSolver
from solvers.common import Options, Result, direct_result


def lu_decompose(a: Matrix) -> tuple[Matrix, Matrix]:
    """Doolittle factorization: A = LU with a unit diagonal in L."""
    n = len(a)
    lower, upper = [[0.0] * n for _ in range(n)], [[0.0] * n for _ in range(n)]
    for k in range(n):
        lower[k][k] = 1.0
        for j in range(k, n):
            upper[k][j] = a[k][j] - fsum(lower[k][s] * upper[s][j] for s in range(k))
        if is_small_pivot(upper[k][k], upper[k][k:]):
            raise ArithmeticError(f"Step {k + 1}: LU without row permutations is unavailable or numerically unstable.")
        for i in range(k + 1, n):
            lower[i][k] = (
                a[i][k] - fsum(lower[i][s] * upper[s][k] for s in range(k))
            ) / upper[k][k]
    return lower, upper


class LUFactorizationSolver(LinearSolver):
    method = "lu"
    is_iterative = False

    def _solve(self, a: Matrix, f: Vector, options: Options) -> Result:
        try:
            lower, upper = lu_decompose(a)
        except ArithmeticError as error:
            return Result("lu", "not_applicable", detail=str(error))
        y = forward_substitution(lower, f)
        return direct_result("lu", a, f, backward_substitution(upper, y), options)

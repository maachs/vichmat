"""Gaussian elimination with partial pivoting by column."""

from linalg import Matrix, Vector, backward_substitution, is_small_pivot
from solvers.base import LinearSolver
from solvers.common import Options, Result, direct_result


class GaussianEliminationSolver(LinearSolver):
    method = "gauss"
    is_iterative = False

    def _solve(self, a: Matrix, f: Vector, options: Options) -> Result:
        upper, rhs = [row[:] for row in a], f[:]
        n = len(f)
        for k in range(n):
            pivot_row = max(range(k, n), key=lambda i: abs(upper[i][k]))
            if is_small_pivot(upper[pivot_row][k], upper[pivot_row][k:]):
                return Result("gauss", "breakdown", detail=f"Step {k + 1}: the pivot is zero or too small.")
            upper[k], upper[pivot_row] = upper[pivot_row], upper[k]
            rhs[k], rhs[pivot_row] = rhs[pivot_row], rhs[k]
            for i in range(k + 1, n):
                multiplier = upper[i][k] / upper[k][k]
                upper[i][k] = 0.0
                for j in range(k + 1, n):
                    upper[i][j] -= multiplier * upper[k][j]
                rhs[i] -= multiplier * rhs[k]
        return direct_result("gauss", a, f, backward_substitution(upper, rhs), options)

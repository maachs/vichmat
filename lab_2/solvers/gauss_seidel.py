"""Gauss-Seidel updates that immediately use newly computed components."""

from math import fsum

from linalg import Matrix, Vector
from solvers.common import Options
from solvers.stationary import StationarySolver


class GaussSeidelSolver(StationarySolver):
    method = "seidel"

    def _next_iterate(self, a: Matrix, f: Vector, old: Vector, options: Options) -> Vector:
        new = old[:]
        for i in range(len(f)):
            lower_sum = fsum(a[i][j] * new[j] for j in range(i))
            upper_sum = fsum(a[i][j] * old[j] for j in range(i + 1, len(f)))
            total = lower_sum + upper_sum
            new[i] = (f[i] - total) / a[i][i]
        return new

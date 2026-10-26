"""Jacobi updates computed entirely from the previous iterate."""

from math import fsum

from linalg import Matrix, Vector
from solvers.common import Options
from solvers.stationary import StationarySolver


class JacobiSolver(StationarySolver):
    method = "jacobi"

    def _next_iterate(self, a: Matrix, f: Vector, old: Vector, options: Options) -> Vector:
        new = [0.0] * len(f)
        for i in range(len(f)):
            total = fsum(a[i][j] * old[j] for j in range(len(f)) if j != i)
            new[i] = (f[i] - total) / a[i][i]
        return new

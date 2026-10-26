"""Jacobi updates computed entirely from the previous iterate."""

from math import fsum

from linalg import Matrix, Vector
from solvers.common import Options, Result
from solvers.stationary import run_stationary


def jacobi(a: Matrix, f: Vector, options: Options) -> Result:
    return run_stationary("jacobi", a, f, options, _next_iterate)


def _next_iterate(a: Matrix, f: Vector, old: Vector, options: Options) -> Vector:
    new = [0.0] * len(f)
    for i in range(len(f)):
        total = fsum(a[i][j] * old[j] for j in range(len(f)) if j != i)
        new[i] = (f[i] - total) / a[i][i]
    return new

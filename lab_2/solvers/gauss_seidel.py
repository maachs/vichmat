"""Gauss-Seidel updates that immediately use newly computed components."""

from math import fsum

from linalg import Matrix, Vector
from solvers.common import Options, Result
from solvers.stationary import run_stationary


def gauss_seidel(a: Matrix, f: Vector, options: Options) -> Result:
    return run_stationary("seidel", a, f, options, _next_iterate)


def _next_iterate(a: Matrix, f: Vector, old: Vector, options: Options) -> Vector:
    new = old[:]
    for i in range(len(f)):
        lower_sum = fsum(a[i][j] * new[j] for j in range(i))
        upper_sum = fsum(a[i][j] * old[j] for j in range(i + 1, len(f)))
        total = lower_sum + upper_sum
        new[i] = (f[i] - total) / a[i][i]
    return new

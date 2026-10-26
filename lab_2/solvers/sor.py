"""Successive over-relaxation using already relaxed lower components."""

from math import fsum

from linalg import Matrix, Vector
from solvers.common import Options, Result
from solvers.stationary import run_stationary


def sor(a: Matrix, f: Vector, options: Options) -> Result:
    return run_stationary("sor", a, f, options, _next_iterate)


def _next_iterate(a: Matrix, f: Vector, old: Vector, options: Options) -> Vector:
    new = old[:]
    omega = options.omega
    for i in range(len(f)):
        lower_sum = fsum(a[i][j] * new[j] for j in range(i))
        upper_sum = fsum(a[i][j] * old[j] for j in range(i + 1, len(f)))
        total = lower_sum + upper_sum
        coordinate_value = (f[i] - total) / a[i][i]
        new[i] = (1 - omega) * old[i] + omega * coordinate_value
    return new

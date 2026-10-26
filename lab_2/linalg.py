"""Basic operations implemented without third-party linear algebra solvers."""

from dataclasses import dataclass
from math import fsum, hypot, isfinite
from sys import float_info

Vector = list[float]
Matrix = list[Vector]

def is_small_pivot(pivot: float, row: Vector) -> bool:
    """Use a relative pivot threshold so rescaling A preserves the check."""
    return abs(pivot) <= 16 * float_info.epsilon * max(abs(value) for value in row)


def dot(x: Vector, y: Vector) -> float:
    return fsum(xi * yi for xi, yi in zip(x, y))


def norm(x: Vector) -> float:
    return hypot(*x)


def matvec(a: Matrix, x: Vector) -> Vector:
    return [dot(row, x) for row in a]


def residual(a: Matrix, x: Vector, f: Vector) -> Vector:
    return [fi - axi for fi, axi in zip(f, matvec(a, x))]


def matrix_norm_inf(a: Matrix) -> float:
    return max(fsum(abs(value) for value in row) for row in a)


def is_symmetric(a: Matrix) -> bool:
    scale = max(abs(value) for row in a for value in row)
    return all(
        abs(a[i][j] - a[j][i]) <= 1e-12 * scale
        for i in range(len(a)) for j in range(i)
    )


def is_positive_definite(a: Matrix) -> bool:
    """Check positive definiteness using a hand-written Cholesky factorization."""
    if not is_symmetric(a):
        return False
    n = len(a)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = a[i][j] - fsum(lower[i][k] * lower[j][k] for k in range(j))
            if not isfinite(value):
                return False
            if i == j:
                if value <= 0.0:
                    return False
                lower[i][j] = value ** 0.5
            else:
                lower[i][j] = value / lower[j][j]
    return True


def has_strict_diagonal_dominance(a: Matrix) -> bool:
    return all(
        abs(row[i]) > fsum(abs(value) for j, value in enumerate(row) if j != i)
        for i, row in enumerate(a)
    )


def forward_substitution(lower: Matrix, f: Vector) -> Vector:
    x = [0.0] * len(f)
    for i in range(len(f)):
        if lower[i][i] == 0.0:
            raise ArithmeticError("The lower triangular matrix has a zero diagonal entry.")
        x[i] = (f[i] - fsum(lower[i][j] * x[j] for j in range(i))) / lower[i][i]
    return x


def backward_substitution(upper: Matrix, f: Vector) -> Vector:
    x = [0.0] * len(f)
    for i in range(len(f) - 1, -1, -1):
        if upper[i][i] == 0.0:
            raise ArithmeticError("The upper triangular matrix has a zero diagonal entry.")
        x[i] = (
            f[i] - fsum(upper[i][j] * x[j] for j in range(i + 1, len(f)))
        ) / upper[i][i]
    return x


@dataclass(frozen=True)
class MatrixProperties:
    symmetric: bool
    positive_definite: bool
    diagonal_dominance: bool
    nonzero_diagonal: bool

    @classmethod
    def analyze(cls, matrix: Matrix) -> "MatrixProperties":
        return cls(
            symmetric=is_symmetric(matrix),
            positive_definite=is_positive_definite(matrix),
            diagonal_dominance=has_strict_diagonal_dominance(matrix),
            nonzero_diagonal=all(row[i] != 0.0 for i, row in enumerate(matrix)),
        )

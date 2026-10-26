"""Build assignment variant m and read arbitrary square linear systems."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from linalg import Matrix, Vector, matrix_norm_inf, norm

# Store the preset as coefficients rather than a manually entered 20-by-20 matrix.
VARIANT_M_SIZE = 20
VARIANT_M_MAIN = 10.0
VARIANT_M_FIRST = 1.0
VARIANT_M_SECOND = 0.1

@dataclass
class LinearSystem:
    name: str
    a: Matrix
    f: Vector

    def __post_init__(self) -> None:
        n = len(self.a)
        if n == 0 or len(self.f) != n or any(len(row) != n for row in self.a):
            raise ValueError("A must be a nonempty square matrix, and f must have the same size.")
        if not all(isfinite(value) for row in self.a for value in row):
            raise ValueError("Matrix coefficients must be finite numbers.")
        if not all(isfinite(value) for value in self.f):
            raise ValueError("The right-hand side must contain finite numbers.")
        try:
            representable = isfinite(norm(self.f)) and isfinite(matrix_norm_inf(self.a))
        except OverflowError:
            representable = False
        if not representable:
            raise ValueError("The numbers are too large; rescale the system before solving it.")


def variant_m() -> LinearSystem:
    n = VARIANT_M_SIZE
    diagonals = {0: VARIANT_M_MAIN, 1: VARIANT_M_FIRST, 2: VARIANT_M_SECOND}
    a = [[diagonals.get(abs(i - j), 0.0) for j in range(n)] for i in range(n)]
    return LinearSystem("Variant m", a, [float(i) for i in range(1, n + 1)])


def read_system(path: str | Path) -> LinearSystem:
    """Read n followed by n rows of the augmented matrix [A | f]."""
    path = Path(path)
    lines = []
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        content = raw.split("#", 1)[0].strip()
        if content:
            lines.append((number, content.split()))
    if not lines:
        raise ValueError("The system file is empty.")
    if len(lines[0][1]) != 1:
        raise ValueError("The first nonempty line must contain only the size n.")
    try:
        n = int(lines[0][1][0])
    except ValueError as error:
        raise ValueError("The size n must be a positive integer.") from error
    if n <= 0:
        raise ValueError("The size n must be a positive integer.")
    if len(lines) != n + 1:
        raise ValueError(f"Expected exactly {n} equation rows after n={n}.")
    a, f = [], []
    for number, tokens in lines[1:]:
        if len(tokens) != n + 1:
            raise ValueError(f"Line {number}: expected {n} coefficients followed by the right-hand side.")
        try:
            values = [float(token) for token in tokens]
        except ValueError as error:
            raise ValueError(f"Line {number}: expected numbers with a decimal point.") from error
        a.append(values[:-1])
        f.append(values[-1])
    return LinearSystem(path.stem, a, f)

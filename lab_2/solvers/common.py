"""Shared stopping criteria and true residual histories."""

from dataclasses import dataclass, field
from math import inf, isfinite

from linalg import Matrix, Vector, is_positive_definite, norm, residual

METHOD_NAMES = {
    "gauss": "Gaussian elimination (partial pivoting)",
    "lu": "LU factorization",
    "jacobi": "Jacobi",
    "seidel": "Gauss-Seidel",
    "sor": "Successive over-relaxation (SOR)",
    "gradient": "Gradient descent",
    "minres": "Minimal residual",
    "cg": "Conjugate gradient (CG)",
    "bicgstab": "BiCGSTAB",
}
STATUS_NAMES = {
    "converged": "converged",
    "not_applicable": "not applicable",
    "breakdown": "numerical breakdown",
    "iteration_limit": "iteration limit reached",
    "diverged": "divergence or overflow",
    "inaccurate": "tolerance not reached",
}


@dataclass(frozen=True)
class Options:
    rtol: float = 1e-10
    atol: float = 1e-12
    max_iterations: int = 10000
    omega: float = 1.05
    gradient_step: float | None = None

    def __post_init__(self) -> None:
        if not all(isfinite(v) and v >= 0 for v in (self.rtol, self.atol)):
            raise ValueError("Tolerances must be finite nonnegative numbers.")
        if self.rtol == self.atol == 0:
            raise ValueError("At least one tolerance must be positive.")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("The iteration limit must be a positive integer.")
        if not isfinite(self.omega) or not 1 <= self.omega < 2:
            raise ValueError("Successive over-relaxation requires 1 <= omega < 2.")
        if self.gradient_step is not None and (
            not isfinite(self.gradient_step) or self.gradient_step <= 0
        ):
            raise ValueError("The gradient descent step must be finite and positive.")


@dataclass
class Result:
    method: str
    status: str
    x: Vector | None = None
    iterations: int = 0
    residual_norm: float | None = None
    relative_residual: float | None = None
    history: list[float] = field(default_factory=list)
    detail: str = ""
    is_iterative: bool = True


def check_positive_definite(method: str, a: Matrix) -> Result | None:
    if not is_positive_definite(a):
        return Result(
            method, "not_applicable",
            detail="This method requires a symmetric positive definite matrix.",
        )
    return None


class IterationState:
    def __init__(self, method: str, a: Matrix, f: Vector, options: Options):
        self._method = method
        self._a = a
        self._f = f
        self._options = options
        self._x = [0.0] * len(f)
        self._f_norm = norm(f)
        self._scale = self._f_norm if self._f_norm else 1.0
        self._threshold = options.atol + options.rtol * self._f_norm
        self._history = [self._f_norm]

    @property
    def x(self) -> Vector:
        return self._x[:]

    @property
    def threshold(self) -> float:
        return self._threshold

    def initial_result(self) -> Result | None:
        if self._history[0] <= self.threshold:
            return self.finish("converged")
        return None

    def advance(self, x: Vector) -> tuple[Vector, str | None]:
        """Record one completed iteration; initialization records iteration zero."""
        self._x = x[:]
        try:
            r = residual(self._a, x, self._f)
            value = norm(r)
        except (OverflowError, ValueError):
            r, value = [inf] * len(x), inf
        self._history.append(value)
        if not all(isfinite(v) for v in x) or not isfinite(value):
            return r, "diverged"
        if value <= self.threshold:
            return r, "converged"
        if value > 1e12 * max(self._f_norm, self.threshold):
            return r, "diverged"
        return r, None

    def finish(self, status: str, detail: str = "") -> Result:
        if status == "iteration_limit" and not detail:
            detail = f"Tolerance was not reached after {self._options.max_iterations} iterations."
        if status == "diverged" and not detail:
            detail = "The residual grew by a factor above 10^12, or arithmetic overflow occurred."
        return Result(
            method=self._method,
            status=status,
            x=self.x,
            iterations=len(self._history) - 1,
            residual_norm=self._history[-1],
            relative_residual=self._history[-1] / self._scale,
            history=self._history[:],
            detail=detail,
        )


def direct_result(method: str, a: Matrix, f: Vector, x: Vector, options: Options) -> Result:
    value = norm(residual(a, x, f))
    f_norm = norm(f)
    if not all(isfinite(v) for v in x) or not isfinite(value):
        status = "diverged"
    else:
        status = "converged" if value <= options.atol + options.rtol * f_norm else "inaccurate"
    return Result(
        method=method,
        status=status,
        x=x,
        residual_norm=value,
        relative_residual=value / (f_norm if f_norm else 1.0),
        is_iterative=False,
    )

"""A common solver contract with centralized applicability checks."""

from abc import ABC, abstractmethod

from linalg import Matrix, MatrixProperties, Vector
from solvers.common import Options, Result
from systems import LinearSystem


class LinearSolver(ABC):
    """Implement one numerical method behind the same public solve interface."""

    is_iterative = True
    requires_positive_definite = False
    requires_nonzero_diagonal = False

    @property
    @abstractmethod
    def method(self) -> str:
        """Return the stable method identifier used in saved results."""

    def solve(
        self, system: LinearSystem, options: Options,
        properties: MatrixProperties | None = None,
    ) -> Result:
        if properties is None:
            properties = MatrixProperties.analyze(system.a)
        if self.requires_positive_definite and not properties.positive_definite:
            return Result(
                self.method, "not_applicable", is_iterative=self.is_iterative,
                detail="This method requires a symmetric positive definite matrix.",
            )
        if self.requires_nonzero_diagonal and not properties.nonzero_diagonal:
            return Result(
                self.method, "not_applicable", is_iterative=self.is_iterative,
                detail="A zero diagonal entry requires a transformation of the system.",
            )
        try:
            result = self._solve(system.a, system.f, options)
        except (ArithmeticError, ValueError) as error:
            result = Result(self.method, "breakdown", detail=f"Numerical arithmetic failed: {error}")
        result.is_iterative = self.is_iterative
        return result

    @abstractmethod
    def _solve(self, a: Matrix, f: Vector, options: Options) -> Result:
        """Run the numerical algorithm after its preconditions have been checked."""

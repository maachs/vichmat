"""Shared iteration control for stationary coordinate methods."""

from abc import abstractmethod

from linalg import Matrix, Vector
from solvers.base import LinearSolver
from solvers.common import IterationState, Options, Result


class StationarySolver(LinearSolver):
    """Delegate update formulas to each method and share convergence control."""

    requires_nonzero_diagonal = True

    def _solve(self, a: Matrix, f: Vector, options: Options) -> Result:
        state = IterationState(self.method, a, f, options)
        initial = state.initial_result()
        if initial:
            return initial
        for _ in range(options.max_iterations):
            new = self._next_iterate(a, f, state.x, options)
            _, status = state.advance(new)
            if status:
                return state.finish(status)
        return state.finish("iteration_limit")

    @abstractmethod
    def _next_iterate(self, a: Matrix, f: Vector, old: Vector, options: Options) -> Vector:
        """Compute one iterate using the concrete method's coordinate formulas."""

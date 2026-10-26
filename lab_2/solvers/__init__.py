"""Hand-written linear system solvers behind a common interface."""

from solvers.base import LinearSolver
from solvers.bicgstab import BiCGSTABSolver
from solvers.conjugate_gradient import ConjugateGradientSolver
from solvers.gauss_seidel import GaussSeidelSolver
from solvers.gaussian_elimination import GaussianEliminationSolver
from solvers.gradient_descent import GradientDescentSolver
from solvers.jacobi import JacobiSolver
from solvers.lu_factorization import LUFactorizationSolver
from solvers.minimal_residual import MinimalResidualSolver
from solvers.sor import SORSolver


def default_solvers() -> tuple[LinearSolver, ...]:
    return (
        GaussianEliminationSolver(), LUFactorizationSolver(), JacobiSolver(),
        GaussSeidelSolver(), SORSolver(), GradientDescentSolver(),
        MinimalResidualSolver(), ConjugateGradientSolver(), BiCGSTABSolver(),
    )

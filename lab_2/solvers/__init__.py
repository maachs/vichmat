"""Hand-written linear system solvers with the interface (a, f, options)."""

from solvers.bicgstab import bicgstab
from solvers.conjugate_gradient import conjugate_gradient
from solvers.gauss_seidel import gauss_seidel
from solvers.gaussian_elimination import gaussian_elimination
from solvers.gradient_descent import gradient_descent
from solvers.jacobi import jacobi
from solvers.lu_factorization import lu_factorization
from solvers.minimal_residual import minimal_residual
from solvers.sor import sor

SOLVERS = (
    gaussian_elimination, lu_factorization, jacobi, gauss_seidel, sor,
    gradient_descent, minimal_residual, conjugate_gradient, bicgstab,
)

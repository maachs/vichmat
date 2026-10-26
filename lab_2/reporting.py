"""Print solutions and save numerical diagnostics separately."""

import csv
from pathlib import Path

from linalg import MatrixProperties, matrix_norm_inf
from solvers.common import METHOD_NAMES, STATUS_NAMES, Options, Result
from systems import LinearSystem


def print_solutions(results: list[Result]) -> None:
    for result in results:
        print(f"\n{METHOD_NAMES[result.method]}:")
        if result.status == "converged" and result.x is not None:
            for index, value in enumerate(result.x, 1):
                print(f"  x{index} = {value:.12g}")
        else:
            print(f"  No converged solution ({STATUS_NAMES[result.status]}). See results/report.txt.")


def save_report(
    system: LinearSystem, options: Options, properties: MatrixProperties,
    results: list[Result], output: Path,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "report.txt"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"System: {system.name}; unknowns: {len(system.f)}\n")
        stream.write(f"Symmetric: {properties.symmetric}\n")
        stream.write(f"Positive definite: {properties.positive_definite}\n")
        stream.write(f"Strict diagonal dominance: {properties.diagonal_dominance}\n")
        stream.write(f"Nonzero diagonal: {properties.nonzero_diagonal}\n")
        stream.write(f"Stopping rule: ||f-Ax||_2 <= {options.atol:g} + {options.rtol:g} * ||f||_2\n")
        stream.write(f"Initial guess: zero; iteration limit: {options.max_iterations}\n")
        stream.write(f"SOR omega: {options.omega:g}\n")
        if properties.positive_definite:
            step = options.gradient_step if options.gradient_step is not None else 1 / matrix_norm_inf(system.a)
            stream.write(f"Constant gradient descent step: {step:.12g}\n")
        stream.write("\nAugmented matrix [A | f]:\n")
        for row, rhs in zip(system.a, system.f):
            stream.write(" ".join(str(value) for value in row) + f" | {rhs}\n")
        for result in results:
            stream.write(f"\n{METHOD_NAMES[result.method]}: {STATUS_NAMES[result.status]}\n")
            if result.is_iterative:
                stream.write(f"Iterations: {result.iterations}\n")
            stream.write(f"Residual norm: {result.residual_norm}\n")
            stream.write(f"Relative residual: {result.relative_residual}\n")
            if result.detail:
                stream.write(f"Details: {result.detail}\n")
        stream.write("\nFor f=0, the relative residual field contains the absolute residual.\n")
        stream.write("solutions.csv contains solutions, or the last approximation when convergence failed.\n")
    return path


def save_solutions(system: LinearSystem, results: list[Result], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "solutions.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "status", "iterations", "residual_norm", "relative_residual", "detail"]
                        + [f"x{i + 1}" for i in range(len(system.f))])
        for result in results:
            solution = result.x if result.x is not None else [None] * len(system.f)
            iterations = result.iterations if result.is_iterative else ""
            writer.writerow([
                result.method, result.status, iterations,
                result.residual_norm, result.relative_residual, result.detail,
            ] + solution)
    return path

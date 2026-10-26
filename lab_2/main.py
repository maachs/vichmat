import errno
import sys
from pathlib import Path

from linalg import MatrixProperties
from reporting import print_solutions, save_report, save_solutions
from solvers import default_solvers
from solvers.common import Options
from systems import LinearSystem, read_system, variant_m


def describe_file_error(error: OSError) -> str:
    descriptions = {
        errno.ENOENT: "File or directory not found",
        errno.EACCES: "Permission denied",
        errno.EPERM: "Operation not permitted",
        errno.EISDIR: "Expected a file, but found a directory",
        errno.ENOTDIR: "A path component is not a directory",
        errno.ENOSPC: "Not enough free disk space",
    }
    description = descriptions.get(error.errno, f"File operation failed (OS error {error.errno})")
    return f"{description}: {error.filename}" if error.filename else description


def choose_system() -> LinearSystem | None:
    while True:
        print("\nChoose a linear system:")
        print("  1 — solve assignment variant m)")
        print("  2 — load a system from a text file")
        print("  0 — exit")
        choice = input("Your choice: ").strip() or "1"
        if choice == "0":
            return None
        if choice == "1":
            return variant_m()
        if choice == "2":
            text = input("TXT file path (for example, examples/system.txt): ").strip().strip('"')
            if not text:
                print("No file path was provided.")
                continue
            try:
                return read_system(text)
            except OSError as error:
                print(f"Could not read the system: {describe_file_error(error)}")
            except ValueError as error:
                print(f"Could not read the system: {error}")
        else:
            print("Enter 1, 2, or 0.")


def main() -> int:
    try:
        system = choose_system()
        if system is None:
            return 0
        options = Options()
        output = Path(__file__).resolve().parent / "results"
        properties = MatrixProperties.analyze(system.a)
        results = [solver.solve(system, options, properties) for solver in default_solvers()]
        print_solutions(results)
        save_solutions(system, results, output)
        report = save_report(system, options, properties, results, output)
        # Remove the previous plot so it cannot be mistaken for this run's result.
        (output / "convergence.png").unlink(missing_ok=True)
        try:
            from plotting import save_plot
            save_plot(system, options, results, output)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            with report.open("a", encoding="utf-8") as stream:
                stream.write(f"\nPlot creation failed: {error}\n")
                stream.write("On Ubuntu, install Matplotlib with: sudo apt install python3-matplotlib\n")
            print("\nCould not save the plot. See results/report.txt.")
            return 1
        return 0 if any(result.status == "converged" for result in results) and all(
            result.status in ("converged", "not_applicable") for result in results
        ) else 2
    except OSError as error:
        print(f"Error: {describe_file_error(error)}")
        return 1
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    except EOFError:
        print("\nInput ended before a system was selected.")
        return 1
    except KeyboardInterrupt:
        print("\nComputation interrupted by the user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

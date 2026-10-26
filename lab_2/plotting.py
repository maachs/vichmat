"""Use Matplotlib only to plot previously computed residual histories."""

from math import isfinite, nan
from pathlib import Path

from linalg import norm
from solvers.common import METHOD_NAMES, STATUS_NAMES, Options, Result
from systems import LinearSystem


def save_plot(
    system: LinearSystem, options: Options, results: list[Result], output: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    iterative = [result for result in results if result.is_iterative and result.history]
    output.mkdir(parents=True, exist_ok=True)
    f_norm = norm(system.f)
    scale = f_norm or 1.0
    tolerance = (options.atol + options.rtol * f_norm) / scale
    floor = max(min(tolerance * 0.01, 1e-16), 1e-300)
    ylabel = "Relative residual ‖f − Ax⁽ᵏ⁾‖₂ / ‖f‖₂" if f_norm else "Absolute residual ‖f − Ax⁽ᵏ⁾‖₂"

    def values(result: Result) -> list[float]:
        # A logarithmic axis needs a positive plotting floor for exact zeros.
        # Keep the recorded history intact; only plotting values are clipped.
        return [max(value / scale, floor) if isfinite(value) else nan for value in result.history]

    def configure(ax, title: str, zero_present: bool) -> None:
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Iteration k")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if tolerance > 0:
            ax.axhline(tolerance, color="black", linestyle="--", linewidth=1, label="Tolerance")
        ax.grid(True, which="both", alpha=0.3)
        if zero_present:
            ax.text(0, -0.18, "Exact zeros are shown at a positive plotting floor.",
                    transform=ax.transAxes, fontsize=8)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    configure(ax, f"{system.name}: iterative method comparison",
              any(value == 0 for result in iterative for value in result.history))
    for result in iterative:
        label = METHOD_NAMES.get(result.method, result.method)
        if result.status != "converged":
            label += f" ({STATUS_NAMES[result.status]})"
        ax.plot(range(len(result.history)), values(result), marker=".", markersize=4, label=label)
    if not iterative:
        ax.text(0.5, 0.5, "No iterative histories: check method statuses in solutions.csv.",
                ha="center", transform=ax.transAxes)
    ax.legend(fontsize=9)
    path = output / "convergence.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path

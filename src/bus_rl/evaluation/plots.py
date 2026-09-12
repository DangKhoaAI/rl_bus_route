"""Headless plots for evaluation reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    _plt().close(fig)


def plot_waiting_cdf(frame: pd.DataFrame, output: Path) -> None:
    fig, axis = _plt().subplots(figsize=(7, 4))
    plotted = False
    for method, group in frame.groupby("method"):
        waits = group["mean_wait"].dropna().to_numpy(dtype=float)
        if waits.size == 0:
            continue
        ordered = np.sort(waits)
        axis.plot(ordered, np.linspace(0, 1, ordered.size), label=str(method))
        plotted = True
    axis.set_xlabel("Mean observed wait (minutes)")
    axis.set_ylabel("CDF across days")
    axis.set_title("Waiting time CDF (includes censored/abandoned days)")
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "no waiting observations", ha="center", va="center")
    _save(fig, output / "waiting_cdf.png")


def plot_cost_bars(frame: pd.DataFrame, output: Path) -> None:
    fig, axis = _plt().subplots(figsize=(7, 4))
    summary = frame.groupby("method", dropna=False)["total_cost_core"].mean()
    axis.bar([str(index) for index in summary.index], summary.to_numpy())
    axis.set_ylabel("Mean core cost (passenger-minute equivalent)")
    axis.set_title("Core-weighted cost by method")
    _save(fig, output / "cost_bars.png")


def plot_learning_curves(history: pd.DataFrame, output: Path) -> None:
    fig, axis = _plt().subplots(figsize=(7, 4))
    if history.empty:
        axis.text(0.5, 0.5, "no learning curve", ha="center", va="center")
    else:
        for seed, group in history.groupby("seed"):
            axis.plot(group["timesteps"], group["val_cost"], label=f"seed {seed}")
        axis.legend()
    axis.set_xlabel("Transitions")
    axis.set_ylabel("Validation cost")
    axis.set_title("Learning curves")
    _save(fig, output / "learning_curves.png")


def plot_queue_heatmap(traces: list[dict], output: Path) -> None:
    fig, axis = _plt().subplots(figsize=(8, 3))
    if not traces:
        axis.text(0.5, 0.5, "no trace", ha="center", va="center")
        _save(fig, output / "queue_heatmap.png")
        return
    queues = np.array([row["queues"] for row in traces], dtype=float)
    image = axis.imshow(queues.T, aspect="auto", origin="lower")
    axis.set_xlabel("Control step")
    axis.set_ylabel("Route")
    axis.set_title("Queue heatmap (passengers)")
    fig.colorbar(image, ax=axis, label="waiting passengers")
    _save(fig, output / "queue_heatmap.png")


def plot_fleet_gantt(traces: list[dict], output: Path) -> None:
    fig, axis = _plt().subplots(figsize=(8, 4))
    if not traces:
        axis.text(0.5, 0.5, "no trace", ha="center", va="center")
        _save(fig, output / "fleet_gantt.png")
        return
    colors = {
        "DEPOT_IDLE": "#bbbbbb",
        "DEADHEAD": "#d62728",
        "TERMINAL_IDLE": "#17becf",
        "SERVICE_MOVING": "#2ca02c",
        "SERVICE_DWELL": "#1f77b4",
        "LAYOVER": "#ff7f0e",
    }
    n_buses = len(traces[0]["buses"])
    times = [row["time_s"] / 60 for row in traces]
    width = (times[1] - times[0]) if len(times) > 1 else 2
    for bus_id in range(n_buses):
        for time, row in zip(times, traces):
            phase = row["buses"][bus_id]["phase"]
            axis.barh(bus_id, width, left=time, color=colors.get(phase, "#333333"), height=0.6)
    axis.set_xlabel("Time (minutes)")
    axis.set_ylabel("Bus ID")
    axis.set_title("Bus mission Gantt (deadhead in red)")
    _save(fig, output / "fleet_gantt.png")


def plot_headways(traces: list[dict], output: Path) -> None:
    fig, axis = _plt().subplots(figsize=(7, 4))
    if not traces:
        axis.text(0.5, 0.5, "no trace", ha="center", va="center")
        _save(fig, output / "headway.png")
        return
    times = [row["time_s"] / 60 for row in traces]
    targets = np.array([row["headway_targets"] for row in traces], dtype=float) / 60
    for route in range(targets.shape[1]):
        axis.plot(times, targets[:, route], label=f"target route {route}")
    axis.set_xlabel("Time (minutes)")
    axis.set_ylabel("Headway target (minutes)")
    axis.set_title("Headway targets over the M3 horizon")
    axis.legend()
    _save(fig, output / "headway.png")


def render_plots(
    frame: pd.DataFrame,
    output: Path,
    traces: list[dict] | None = None,
    history: pd.DataFrame | None = None,
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plot_waiting_cdf(frame, output)
    plot_cost_bars(frame, output)
    plot_learning_curves(history if history is not None else pd.DataFrame(), output)
    plot_queue_heatmap(traces or [], output)
    plot_fleet_gantt(traces or [], output)
    plot_headways(traces or [], output)

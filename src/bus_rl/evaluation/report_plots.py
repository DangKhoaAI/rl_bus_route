"""Presentation figures: direct labels, honest zoomed point axes and paired intervals."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd

from bus_rl.evaluation.plots import _plt
from bus_rl.evaluation.report_data import METHOD_NAMES, SPLIT_NAMES, selected_checkpoint

RL_COLOR = "#087f8c"
BASE_COLOR = "#737e8c"
LOSS_COLOR = "#bf4b32"


def _name(method: str) -> str:
    return METHOD_NAMES.get(method, method)


def _style(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#e5e9ed", linewidth=0.8)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#bec6cc")
    axis.tick_params(axis="both", labelsize=10)


def _save(fig, output: Path, name: str, footer: str) -> None:
    fig.text(
        0.02, 0.025, fill(footer, width=int(fig.get_figwidth() * 13)), fontsize=10, color="#495563"
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.88), w_pad=3)
    fig.savefig(output / name, dpi=180, facecolor="white")
    _plt().close(fig)


def plot_main_result(
    summary: pd.DataFrame, comparisons: pd.DataFrame, output: Path, rl_method: str
) -> None:
    splits = list(summary.split.unique())
    fig, axes = _plt().subplots(
        1, len(splits), figsize=(max(7, 5.7 * len(splits)), 5.6), squeeze=False
    )
    fig.suptitle("Operational Cost", fontsize=21, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.89,
        "Mean evaluation cost by operating condition · Lower is better",
        ha="center",
        fontsize=12,
    )
    for axis, split in zip(axes[0], splits):
        group = summary[summary.split == split].copy()
        group["is_rl"] = group.method == rl_method
        group = group.sort_values(["is_rl", "method"], ascending=[False, True])
        x = group.total_cost_core.to_numpy() / 1000
        span = max(float(np.ptp(x)), float(np.max(np.abs(x))) * 0.06, 0.1)
        for y, (_, row) in enumerate(group.iterrows()):
            color = RL_COLOR if row.method == rl_method else BASE_COLOR
            axis.scatter(
                row.total_cost_core / 1000,
                y,
                s=125 if row.method == rl_method else 75,
                color=color,
                zorder=3,
            )
            axis.text(
                row.total_cost_core / 1000 + span * 0.05,
                y,
                f"{row.total_cost_core:,.0f}",
                va="center",
                fontsize=11,
                color=color,
                fontweight="bold" if row.method == rl_method else "normal",
            )
        axis.set_yticks(range(len(group)), [_name(method) for method in group.method])
        axis.set_ylim(len(group) - 0.4, -0.7)
        axis.set_xlim(max(0, x.min() - span * 0.15), x.max() + span * 0.45)
        axis.set_xlabel("Cost (thousand units; data-scaled axis)", fontsize=11)
        axis.set_title(SPLIT_NAMES.get(split, split), fontsize=13, fontweight="bold", pad=24)
        note = "Baseline comparison unavailable"
        note_color = BASE_COLOR
        if not comparisons.empty:
            candidates = (
                comparisons[(comparisons.split == split) & comparisons.baseline_cost.notna()]
                if "baseline_cost" in comparisons
                else pd.DataFrame()
            )
            if not candidates.empty:
                best = candidates.loc[candidates.baseline_cost.idxmin()]
                if pd.notna(best.cost_saved_pct):
                    direction = "lower" if best.cost_saved_pct >= 0 else "higher"
                    note = f"RL cost: {abs(best.cost_saved_pct):.1f}% {direction} vs. best baseline"
                    note_color = RL_COLOR if best.cost_saved_pct >= 0 else LOSS_COLOR
        axis.text(0, 1.025, note, transform=axis.transAxes, fontsize=10, color=note_color)
        _style(axis)
    _save(
        fig,
        output,
        "01_main_result.png",
        "Points show mean cost; axes are scaled to the observed range. Cost is a synthetic operational objective, in passenger-minute equivalents.",
    )


def plot_savings(comparisons: pd.DataFrame, output: Path) -> None:
    splits = list(comparisons.split.unique()) if not comparisons.empty else ["unspecified"]
    fig, axes = _plt().subplots(
        1, len(splits), figsize=(max(8, 5.7 * len(splits)), 5.5), squeeze=False
    )
    fig.suptitle("Cost Reduction", fontsize=20, fontweight="bold", y=0.97)
    fig.text(
        0.5, 0.89, "Relative to each baseline · Positive values favor RL", ha="center", fontsize=12
    )
    for axis, split in zip(axes[0], splits):
        group = comparisons[comparisons.split == split] if not comparisons.empty else pd.DataFrame()
        valid = group[group.cost_saved_pct.notna()] if "cost_saved_pct" in group else pd.DataFrame()
        if valid.empty:
            axis.text(
                0.5,
                0.5,
                "Matched RL and baseline evaluations required",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            bounds = [0.0]
            labels = []
            for y, (_, row) in enumerate(valid.iterrows()):
                value = row.cost_saved_pct
                low, high = row.cost_saved_pct_ci95_low, row.cost_saved_pct_ci95_high
                color = (
                    RL_COLOR
                    if row.status == "rl_lower_cost"
                    else LOSS_COLOR
                    if row.status == "rl_higher_cost"
                    else BASE_COLOR
                )
                if pd.notna(low) and pd.notna(high):
                    axis.plot([low, high], [y, y], color=color, lw=3)
                    bounds.extend([low, high])
                else:
                    bounds.append(value)
                axis.scatter(value, y, color=color, s=95, zorder=3)
                label = f"{value:+.1f}%"
                if row.status in {"inconclusive", "insufficient_days"}:
                    label += " · inconclusive"
                axis.annotate(
                    label,
                    (value, y),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    color=color,
                    fontweight="bold",
                )
                labels.append(_name(row.baseline))
            margin = max(np.ptp(bounds) * 0.2, 3)
            axis.set_xlim(min(bounds) - margin, max(bounds) + margin)
            axis.set_ylim(len(valid) - 0.5, -0.7)
            axis.set_yticks(range(len(valid)), labels)
        axis.axvline(0, color="#283747", lw=1, ls="--")
        axis.set_title(SPLIT_NAMES.get(split, split), fontsize=13, fontweight="bold", pad=16)
        axis.set_xlabel("Cost reduction vs. baseline (%)", fontsize=11)
        _style(axis)
    _save(
        fig,
        output,
        "02_baseline_savings.png",
        "Intervals: 95% paired day bootstrap, with model seeds averaged within each day. Positive values indicate lower RL cost.",
    )


def plot_training(history: pd.DataFrame, training: pd.DataFrame, output: Path) -> None:
    fig, axis = _plt().subplots(figsize=(12, 6.75))
    fig.suptitle("Training Progress", fontsize=21, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.89,
        "Validation cost across training · Stars mark selected checkpoints",
        ha="center",
        fontsize=12,
    )
    if history.empty:
        axis.text(
            0.5,
            0.5,
            "Training history unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        colors = [RL_COLOR, "#5967a0", "#c78125", "#966289"]
        for index, (run_id, group) in enumerate(history.groupby("run_id", sort=False)):
            group = group.sort_values("timesteps")
            best = selected_checkpoint(group)
            encoder = str(group.iloc[0].encoder).upper()
            if encoder == "UNKNOWN":
                encoder = "Policy"
            seed = group.iloc[0].seed
            color = colors[index % len(colors)]
            axis.plot(
                group.timesteps / 1000,
                group.val_cost / 1000,
                color=color,
                alpha=0.8,
                lw=1.7,
                label=f"{encoder} · seed {seed}",
            )
            axis.scatter(
                best.timesteps / 1000,
                best.val_cost / 1000,
                s=180,
                marker="*",
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
        axis.legend(frameon=False, fontsize=10)
    axis.set_xlabel("Training transitions (thousands)", fontsize=12)
    axis.set_ylabel("Validation cost (thousand units)", fontsize=12)
    _style(axis)
    axis.grid(axis="y", color="#e5e9ed", linewidth=0.8)
    note = "Each line represents a training run; observations are unsmoothed. Cost axes are scaled to the observed range."
    if not training.empty:
        if (training.reference_kind == "untrained").all():
            note += " Step zero measures the untrained policy."
        else:
            note += " Historical runs without step zero use their first recorded evaluation as reference."
    _save(fig, output, "03_training_progress.png", note)


def plot_service(summary: pd.DataFrame, output: Path, rl_method: str) -> None:
    splits = list(summary.split.unique())
    metrics = [
        ("mean_wait", "Mean wait (min) · Lower is better"),
        ("p95_wait", "P95 wait (min) · Lower is better"),
    ]
    fig, axes = _plt().subplots(
        2, len(splits), figsize=(max(8, 5.7 * len(splits)), 8), squeeze=False
    )
    fig.suptitle("Service Quality", fontsize=20, fontweight="bold", y=0.97)
    for col, split in enumerate(splits):
        group = summary[summary.split == split].copy()
        group["is_rl"] = group.method == rl_method
        group = group.sort_values(["is_rl", "method"], ascending=[False, True])
        for row_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, col]
            if metric in group and group[metric].notna().any():
                values = group[metric]
                span = max(float(values.max() - values.min()), float(values.max()) * 0.1, 1)
                for y, (_, row) in enumerate(group.iterrows()):
                    if pd.isna(row[metric]):
                        continue
                    color = RL_COLOR if row.method == rl_method else BASE_COLOR
                    axis.scatter(row[metric], y, s=80, color=color, zorder=3)
                    axis.text(
                        row[metric] + span * 0.05,
                        y,
                        f"{row[metric]:.2f}",
                        va="center",
                        color=color,
                        fontsize=10,
                    )
                axis.set_xlim(max(0, values.min() - span * 0.15), values.max() + span * 0.4)
            else:
                axis.text(0.5, 0.5, "Metric unavailable", transform=axis.transAxes, ha="center")
            axis.set_yticks(range(len(group)), [_name(method) for method in group.method])
            axis.set_ylim(len(group) - 0.4, -0.7)
            axis.set_title(
                SPLIT_NAMES.get(split, split) if row_index == 0 else "",
                fontweight="bold",
                fontsize=13,
            )
            axis.set_xlabel(title, fontsize=11)
            _style(axis)
    _save(
        fig,
        output,
        "04_service_quality.png",
        "Points show mean daily metrics on data-scaled axes. P95 wait is the average of daily passenger-level 95th percentiles.",
    )


def render_presentation(summary, comparisons, history, training, output, rl_method):
    plot_main_result(summary, comparisons, output, rl_method)
    plot_savings(comparisons, output)
    plot_training(history, training, output)
    plot_service(summary, output, rl_method)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot paired Newton scores and animate model-guided cable interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib  # noqa: TID253 -- standalone experiment visualization
import numpy as np

matplotlib.use("Agg")

from matplotlib import animation  # noqa: TID253
from matplotlib import pyplot as plt  # noqa: TID253

_METHODS = ("sampling", "wall_matched_sampling", "gradient")
_LABELS = ("Sobol\nmatched queries", "Sobol\nmatched time", "Model gradients")
_COLORS = ("#7766ad", "#b85830", "#146f6c")
_NEUTRAL = "#a6b0b2"
_TARGET = "#17364a"
_PLANNED = "#b85830"
_REPLAY = "#146f6c"


def _confidence(values):
    rng = np.random.default_rng(661)
    bootstrap = rng.choice(values, (10_000, len(values)), replace=True).mean(1)
    return np.percentile(bootstrap, (2.5, 97.5)).tolist()


def _method_comparison(cases, reference):
    gradient = np.array([case["gradient"]["newton_response_error_pct"] for case in cases])
    comparator = np.array([case[reference]["newton_response_error_pct"] for case in cases])
    difference = gradient - comparator
    return {
        "delta_mean_percentage_points": float(difference.mean()),
        "delta_95_pct_case_bootstrap": _confidence(difference),
    }


def _scores(run, output):
    methods, labels, colors = _METHODS, _LABELS, _COLORS
    if "exact_response_solve" in run["results"]:
        methods = (*methods, "exact_response_solve")
        labels = (*labels, "Exact linear solve")
        colors = (*colors, "#526d29")
    panels = [
        (
            "cases",
            f"Training solver settings · {run.get('primary_solver_substeps', 4)} substeps, "
            f"{run.get('primary_solver_sweeps', 8)} sweeps",
        )
    ]
    if run["refined_cases"]:
        panels.append(("refined_cases", "Finer solver settings · 8 substeps, 16 sweeps"))
    figure, axes = plt.subplots(
        1, len(panels), figsize=(5.2 * len(panels), 4.4), constrained_layout=True, sharey=True, squeeze=False
    )
    for axis, (tag, title) in zip(axes[0], panels, strict=True):
        cases = run[tag]
        matrix = np.array([[case[method]["newton_response_error_pct"] for method in methods] for case in cases])
        axis.axhline(100, color="#485967", ls="--", lw=1)
        axis.text(len(methods) - 0.75, 100, "No new force", fontsize=8, va="bottom", ha="right", color="#485967")
        for row in matrix:
            axis.plot(range(len(methods)), row, color=_NEUTRAL, alpha=0.5, lw=0.8, zorder=1)
        for index, (values, color) in enumerate(zip(matrix.T, colors, strict=True)):
            mean = values.mean()
            low, high = _confidence(values)
            axis.scatter(np.full(len(values), index), values, color=color, alpha=0.45, s=15, zorder=2)
            axis.errorbar(
                index,
                mean,
                yerr=[[mean - low], [high - mean]],
                ms=6,
                lw=2,
                marker="D",
                capsize=4,
                color=color,
                zorder=3,
            )
        axis.set_title(f"{title}\n{len(cases)} moving cables", loc="left", fontsize=10)
        axis.set_xticks(range(len(methods)), labels, fontsize=9)
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", alpha=0.22, lw=0.8)
        axis.margins(x=0.14, y=0.1)
    axes[0, 0].set_ylabel("Newton replay error / target response (%)")
    figure.suptitle("Do gradients find better cable controls?", x=0.1, ha="left", fontweight="bold")
    figure.savefig(output / "planning-comparison.png", dpi=180)
    plt.close(figure)


def _animate(traces, case, method, output):
    time = traces["time"]
    data = {key: traces[key] for key in ("nominal", "target")}
    data["replay"] = traces[method]
    changes = {key: data[key] - data["nominal"] for key in ("target", "replay")}
    learned = "model_exact" if method == "exact_response_solve" else "model_optimized"
    if learned in traces:
        changes["model_optimized"] = traces[learned] - traces["model_nominal"]
    else:
        data["model_forecast"] = traces["model_target"]
        changes["model_forecast"] = traces["model_target"] - traces["model_nominal"]
    component = int(np.square(changes["target"]).sum(axis=(0, 1)).argmax())
    xyz = "XYZ"[component]
    figure, (overall, response) = plt.subplots(1, 2, figsize=(9.8, 4.35), constrained_layout=True)
    figure.suptitle(f"Moving cable · held-out scene {case['seed']}", fontweight="bold")
    traces_label = {
        "nominal": "Newton nominal",
        "target": "Newton goal",
        "replay": "Newton replay",
        "model_optimized": "Model-planned response",
        "model_forecast": "Model under target force",
    }
    color = {
        "nominal": _NEUTRAL,
        "target": _TARGET,
        "replay": _REPLAY,
        "model_optimized": _PLANNED,
        "model_forecast": _PLANNED,
    }
    marks = {}
    for key in data:
        (marks[key],) = overall.plot([], [], lw=2, color=color[key], label=traces_label[key])
    overall.plot(data["nominal"][0, 0, 0], data["nominal"][0, 0, 2], "s", ms=7, color=_TARGET)
    overall.set_title("Cable centerline in Newton", loc="left", fontsize=11)
    overall.set_xlabel("World X (m)")
    overall.set_ylabel("World Z (m)")
    overall.set_aspect("equal", adjustable="box")
    all_points = np.concatenate([value.reshape(-1, 3) for value in data.values()])
    minimum, maximum = all_points.min(0), all_points.max(0)
    overall.set_xlim(minimum[0] - 0.03, maximum[0] + 0.03)
    overall.set_ylim(minimum[2] - 0.03, maximum[2] + 0.03)

    shifts = {}
    for key in changes:
        (shifts[key],) = response.plot([], [], label=traces_label[key], color=color[key], lw=2)
    response.axhline(0, color=_NEUTRAL, lw=1)
    response.set_xlim(0, 1)
    scale = 1.1e3 * max(np.abs(delta[..., component]).max() for delta in changes.values())
    response.set_ylim(-scale, scale)
    response.set_title(f"Force-induced motion along world {xyz}", loc="left", fontsize=11)
    response.set_xlabel("Fraction of cable length from clamp")
    response.set_ylabel("Change from nominal (mm)")
    for axis in (overall, response):
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(alpha=0.18)
        axis.legend(loc="best", fontsize=8)
    caption = overall.text(0.02, 0.96, "", transform=overall.transAxes, va="top", color="#485967", fontsize=9)

    def frame(index):
        for key, artist in marks.items():
            artist.set_data(data[key][index, :, 0], data[key][index, :, 2])
        for key, artist in shifts.items():
            delta = changes[key][index, :, component]
            artist.set_data(np.linspace(0, 1, len(delta)), 1000 * delta)
        caption.set_text(f"{time[index]:.2f} s from observation")
        return (*marks.values(), *shifts.values(), caption)

    representative = np.square(changes["target"]).sum(axis=(1, 2)).argmax()
    frame(int(representative))
    figure.savefig(output / "replay-frame.png", dpi=180)
    movie = animation.FuncAnimation(figure, frame, frames=len(time), interval=125)
    movie.save(output / "moving-cable-replay.gif", writer=animation.PillowWriter(fps=8), dpi=100)
    plt.close(figure)


def main():
    """Publish reproducible images and paired bootstrap intervals beside experiment evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    run = json.loads((args.directory / "results.json").read_text())
    _scores(run, args.directory)
    method = "exact_response_solve" if "exact_response_solve" in run["results"] else "gradient"
    ordered = sorted(run["cases"], key=lambda case: case[method]["newton_response_error_pct"])
    case = ordered[len(ordered) // 2]
    with np.load(args.directory / f"case-{case['seed']}.npz", allow_pickle=False) as traces:
        _animate(traces, case, method, args.directory)
    receipt = {
        "representative_seed": case["seed"],
        "representative_selection": "Upper median held-out native Newton replay error",
        "representative_method": method,
        "primary_paired_bootstrap": {method: _method_comparison(run["cases"], method) for method in _METHODS[:2]},
        "refined_paired_bootstrap": {
            method: _method_comparison(run["refined_cases"], method) for method in _METHODS[:2]
        }
        if run["refined_cases"]
        else {},
    }
    (args.directory / "figure-summary.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

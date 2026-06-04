#!/usr/bin/env python3
"""Plot validation errors across policies from Critical Interval MSE result files.

Usage:
    python plot_validation_results.py result1.txt result2.txt ...
    python plot_validation_results.py results/validation/*/val_data_results.txt
    python plot_validation_results.py --dir results/validation
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_result_txt(path: Path) -> dict:
    """Parse a validation result TXT file into a dict of metrics."""
    text = path.read_text(encoding="utf-8")
    metrics = {}

    # Mean error
    m = re.search(r"Mean error:\s+([\d.]+)", text)
    if m:
        metrics["mean"] = float(m.group(1))

    # Median error
    m = re.search(r"Median error:\s+([\d.]+)", text)
    if m:
        metrics["median"] = float(m.group(1))

    # Q1 (1%)
    m = re.search(r"Q1 \(1%\):\s+([\d.]+)", text)
    if m:
        metrics["q1"] = float(m.group(1))

    # Q99 (99%)
    m = re.search(r"Q99 \(99%\):\s+([\d.]+)", text)
    if m:
        metrics["q99"] = float(m.group(1))

    # Mean (1-99%)
    m = re.search(r"Mean \(1-99%\):\s+([\d.]+)", text)
    if m:
        metrics["mean_1_99"] = float(m.group(1))

    # Per-interval mean errors
    m = re.search(r"Per-interval mean errors:\s+\[([\d.,\s]+)\]", text)
    if m:
        vals = [float(x.strip()) for x in m.group(1).split(",")]
        metrics["per_interval"] = vals

    # Total timesteps
    m = re.search(r"Total timesteps evaluated:\s+(\d+)", text)
    if m:
        metrics["n_timesteps"] = int(m.group(1))

    return metrics


def load_errors_from_hdf5(path: Path) -> np.ndarray | None:
    """Load error array from HDF5 file if it exists (same stem, .hdf5 extension)."""
    hdf5_path = path.with_suffix(".hdf5")
    if not hdf5_path.exists():
        return None
    try:
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            return f["errors"][:]
    except Exception:
        return None


def policy_name_from_path(path: Path) -> str:
    """Derive a short policy label from the result file path."""
    # e.g. results/validation/x-vla-lbm-relative-80p-test/val_data_results.txt
    # -> x-vla-lbm-relative-80p-test
    parent = path.parent.name
    if parent and parent != ".":
        return parent
    return path.stem


def plot_bar_comparison(
    policies: list[str],
    metrics_list: list[dict],
    metric_keys: list[str],
    output_path: Path,
    title: str = "Policy Error Comparison",
):
    """Bar chart comparing selected metrics across policies."""
    n_policies = len(policies)
    n_metrics = len(metric_keys)
    x = np.arange(n_policies)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, n_policies * 1.2), 5))

    labels = {
        "mean": "Mean",
        "median": "Median",
        "q1": "Q1 (1%)",
        "q99": "Q99 (99%)",
        "mean_1_99": "Mean (1-99%)",
    }

    for i, key in enumerate(metric_keys):
        vals = []
        for m in metrics_list:
            v = m.get(key)
            vals.append(v if v is not None else np.nan)
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=labels.get(key, key),
        )

    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_mean_only(
    policies: list[str],
    metrics_list: list[dict],
    output_path: Path,
    title: str = "Mean Error by Policy",
):
    """Simple bar chart of mean error only."""
    means = [m.get("mean", np.nan) for m in metrics_list]
    fig, ax = plt.subplots(figsize=(max(8, len(policies) * 1.2), 5))
    x = np.arange(len(policies))
    bars = ax.bar(x, means, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_ylabel("Mean Error")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_interval(
    policies: list[str],
    metrics_list: list[dict],
    output_path: Path,
    title: str = "Per-Interval Mean Error by Policy",
):
    """Grouped bar chart of per-interval mean errors."""
    max_intervals = max(
        len(m.get("per_interval", [])) for m in metrics_list
    )
    if max_intervals == 0:
        print("No per-interval data to plot.")
        return

    n_policies = len(policies)
    x = np.arange(n_policies)
    width = 0.8 / max_intervals

    fig, ax = plt.subplots(figsize=(max(8, n_policies * 1.2), 5))

    colors = plt.cm.tab10(np.linspace(0, 1, max_intervals))
    for i in range(max_intervals):
        vals = []
        for m in metrics_list:
            pi = m.get("per_interval", [])
            vals.append(pi[i] if i < len(pi) else np.nan)
        offset = (i - max_intervals / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=f"Interval {i + 1}",
            color=colors[i],
        )

    ax.set_ylabel("Mean Error")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_error_distributions(
    policies: list[str],
    errors_list: list[np.ndarray],
    output_path: Path,
    title: str = "Error Distribution by Policy",
):
    """Box plot of error distributions (requires HDF5 data)."""
    fig, ax = plt.subplots(figsize=(max(8, len(policies) * 1.2), 5))
    bp = ax.boxplot(
        errors_list,
        tick_labels=policies,
        patch_artist=True,
        showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot validation errors across policies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results",
        nargs="+",
        type=str,
        help="Paths to validation result TXT files (e.g. .../val_data_results.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="validation_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Policy Error Comparison",
        help="Title for the main comparison plot",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not plot anything, only parse the results",
    )
    args = parser.parse_args()

    # Resolve paths and collect TXT files
    paths: list[Path] = []
    for p in args.results:
        path = Path(p)
        if path.is_dir():
            # Find all *results*.txt in directory
            paths.extend(path.glob("*results*.txt"))
        elif path.exists():
            if path.suffix.lower() == ".txt":
                paths.append(path)
            else:
                # Assume it's a result dir
                txt = path / "val_data_results.txt"
                if txt.exists():
                    paths.append(txt)
                else:
                    paths.append(path)
        else:
            print(f"Warning: path not found: {path}")

    if not paths:
        print("No result files found.")
        return

    paths = sorted(set(paths))

    # Parse each file
    policies = []
    metrics_list = []
    errors_list = []

    for p in paths:
        try:
            metrics = parse_result_txt(p)
            if not metrics:
                print(f"Warning: no metrics parsed from {p}")
                continue
            policies.append(policy_name_from_path(p))
            metrics_list.append(metrics)
            errs = load_errors_from_hdf5(p)
            if errs is not None:
                errors_list.append(errs)
        except Exception as e:
            print(f"Error parsing {p}: {e}")

    if not policies:
        print("No valid results to plot.")
        return

    if args.no_plot:
        # Only print the results
        for p, m in zip(policies, metrics_list):
            print(f"{p}: {m}")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = f".{args.format}"

    # Main bar chart: mean, median, Q1, Q99, mean (1-99%)
    plot_bar_comparison(
        policies,
        metrics_list,
        metric_keys=["mean", "median", "mean_1_99"],
        output_path=out_dir / f"comparison{ext}",
        title=args.title,
    )

    # Simple mean-only bar chart
    plot_mean_only(
        policies,
        metrics_list,
        output_path=out_dir / f"mean_error{ext}",
        title=args.title,
    )

    # Per-interval comparison
    if any(m.get("per_interval") for m in metrics_list):
        plot_per_interval(
            policies,
            metrics_list,
            output_path=out_dir / f"per_interval{ext}",
            title="Per-Interval Mean Error by Policy",
        )

    # Error distribution box plot (if HDF5 available)
    if len(errors_list) == len(policies):
        plot_error_distributions(
            policies,
            errors_list,
            output_path=out_dir / f"distribution{ext}",
            title="Error Distribution by Policy",
        )
    elif errors_list:
        print("Skipping distribution plot: HDF5 available for only some policies.")


if __name__ == "__main__":
    main()

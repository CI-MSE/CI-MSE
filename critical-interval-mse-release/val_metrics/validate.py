#!/usr/bin/env python3
"""CLI validation script for Critical Interval MSE evaluation.

Reads policy predictions and ground-truth targets from an HDF5 file,
optionally loads labelled critical intervals from the JSON produced by
the labelling tool, and writes per-timestep errors to an output HDF5
file together with a human-readable TXT summary.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from dataclasses import asdict

try:
    from .critical_interval_mse import CriticalIntervalMSE, CriticalIntervalMSEConfig
    from .pose_conversion import convert_to_absolute
except ImportError:
    from critical_interval_mse import CriticalIntervalMSE, CriticalIntervalMSEConfig
    from pose_conversion import convert_to_absolute


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------

def load_input_data(path, load_states=False):
    with h5py.File(path, "r") as f:
        predictions = f["predictions"][:]
        targets = f["targets"][:]
        episode_ends = f["episode_ends"][:]
        states = None
        if load_states:
            if "proprio" not in f:
                raise KeyError(
                    "HDF5 file has no 'proprio' field, which is required "
                    "for relative-to-absolute pose conversion."
                )
            states = f["proprio"][:]
    return predictions, targets, episode_ends, states


def load_critical_intervals(path, num_episodes):
    """Convert the annotator JSON into a list of (start, end) tuples per episode."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intervals = [[] for _ in range(num_episodes)]
    for entry in data:
        ep_idx = entry["episode"]
        if ep_idx >= num_episodes:
            continue
        for iv in entry.get("intervals", []):
            if iv.get("start") is not None and iv.get("end") is not None:
                if isinstance(iv["start"], float):
                    iv["start"] = round(iv["start"] * 10)
                    iv["end"] = round(iv["end"] * 10)
                intervals[ep_idx].append((iv["start"], iv["end"]))

    for ep in intervals:
        ep.sort(key=lambda x: x[0])

    return intervals


def save_results(output_path, errors, episode_ends, interval_ends):
    with h5py.File(output_path, "w") as f:
        f.create_dataset("errors", data=errors)
        f.create_dataset("episode_ends", data=episode_ends)
        f.create_dataset("interval_ends", data=interval_ends)


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def per_interval_means(errors, episode_ends, interval_ends):
    """Average each interval slot across episodes."""
    num_episodes = len(episode_ends)
    n_intervals = interval_ends.shape[1] if interval_ends.ndim == 2 else 0

    buckets = [[] for _ in range(n_intervals)]
    for i in range(num_episodes):
        ep_start = 0 if i == 0 else int(episode_ends[i - 1])
        ep_errors = errors[ep_start : int(episode_ends[i])]
        for j in range(n_intervals):
            if interval_ends[i, j] == -1:
                continue
            int_start = 0 if j == 0 else int(interval_ends[i, j - 1])
            int_end = int(interval_ends[i, j])
            if int_end > int_start:
                buckets[j].append(float(np.mean(ep_errors[int_start:int_end])))

    return [float(np.mean(b)) if b else float("nan") for b in buckets]


def generate_summary(
    config,
    errors,
    episode_ends,
    interval_ends,
    output_path,
    action_type="absolute",
    inputs=None,
):
    lines = []
    lines.append("=== Critical Interval MSE Validation Report ===")
    lines.append("")
    lines.append("Config:")
    lines.append(f"  action_type: {action_type}")
    if inputs:
        input_names = [Path(p).name for p in inputs]
        lines.append(f"  n_inputs: {len(inputs)}")
        lines.append(f"  inputs: {', '.join(input_names)}")
    for key, value in asdict(config).items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Results:")
    lines.append(f"  Total timesteps evaluated: {len(errors)}")

    if len(errors) > 0:
        mean = np.mean(errors)
        median = np.median(errors)
        q1 = np.percentile(errors, 1)
        q99 = np.percentile(errors, 99)
        mask = (errors >= q1) & (errors <= q99)
        trimmed_mean = np.mean(errors[mask]) if mask.any() else mean

        lines.append(f"  Mean error:    {mean:.6f}")
        lines.append(f"  Median error:  {median:.6f}")
        lines.append(f"  Q1 (1%):       {q1:.6f}")
        lines.append(f"  Q99 (99%):     {q99:.6f}")
        lines.append(f"  Mean (1-99%):  {trimmed_mean:.6f}")

        iv_means = per_interval_means(errors, episode_ends, interval_ends)
        if iv_means:
            formatted = [f"{v:.6f}" for v in iv_means]
            lines.append(f"  Per-interval mean errors: [{', '.join(formatted)}]")
    else:
        lines.append("  No errors computed.")

    summary = "\n".join(lines)

    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(summary)
    return txt_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Critical Interval MSE Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "inputs",
        nargs="+",
        type=str,
        help="Input HDF5 file path(s). Multiple inputs will be concatenated into a single joint result.",
    )
    p.add_argument(
        "--intervals",
        type=str,
        default=None,
        nargs="+",
        help="Critical intervals JSON file(s) (from the labelling tool). Must match the number of inputs when critical interval filtering is enabled.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 file path (default: <first_input>_results.hdf5)",
    )
    p.add_argument(
        "--action-type",
        type=str,
        default="absolute",
        choices=["absolute", "xvla_relative", "openpi_relative", "umi_relative"],
        help="Action representation type. 'absolute' skips conversion; "
             "'xvla_relative' converts to absolute pose using states from the "
             "HDF5 file before computing errors.",
    )

    # Feature toggles (default: all enabled)
    g = p.add_argument_group("feature toggles")
    g.add_argument(
        "--no-critical-intervals",
        action="store_true",
        help="Disable critical interval filtering (evaluate full episodes)",
    )
    g.add_argument(
        "--no-horizon-truncation",
        action="store_true",
        help="Disable horizon truncation",
    )
    g.add_argument(
        "--no-temporal-ensemble",
        action="store_true",
        help="Disable temporal ensemble",
    )
    g.add_argument(
        "--no-dtw",
        action="store_true",
        help="Use plain MSE instead of DTW distance",
    )

    # Hyperparameters
    h = p.add_argument_group("hyperparameters")
    h.add_argument("--horizon-start", type=int, default=0,
                    help="Horizon truncation start index")
    h.add_argument("--horizon-end", type=int, default=-1,
                    help="Horizon truncation end index (-1 = full horizon)")
    h.add_argument("--ensemble-horizon", type=int, default=1,
                    help="Temporal ensemble window size H_ens (1 = no ensemble)")
    h.add_argument("--interpolation-factor", type=int, default=5,
                    help="Interpolation factor for DTW (1 = no interpolation)")
    h.add_argument(
        "--dtw-window-size",
        type=int,
        default=None,
        metavar="W",
        help="DTW Sakoe-Chiba window size on the original horizon axis "
             "(scaled by interpolation_factor internally). Omit for no window.",
    )
    return p


def main():
    args = build_parser().parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for ip in input_paths:
        if not ip.exists():
            raise FileNotFoundError(f"Input file not found: {ip}")

    output_path = Path(args.output) if args.output else input_paths[0].with_name(
        input_paths[0].stem + "_results.hdf5"
    )

    action_type = args.action_type
    needs_states = action_type != "absolute"

    # ---- Build config ---------------------------------------------
    config = CriticalIntervalMSEConfig(
        use_critical_intervals=not args.no_critical_intervals,
        use_horizon_truncation=not args.no_horizon_truncation,
        use_temporal_ensemble=not args.no_temporal_ensemble,
        use_dtw=not args.no_dtw,
        horizon_start=args.horizon_start,
        horizon_end=args.horizon_end,
        ensemble_horizon=args.ensemble_horizon,
        interpolation_factor=args.interpolation_factor,
        dtw_window_size=args.dtw_window_size,
    )

    # ---- Validate intervals input --------------------------------
    intervals_paths = None
    if config.use_critical_intervals:
        if not args.intervals:
            raise ValueError(
                "--intervals must be provided for each input when critical interval "
                "filtering is enabled. Pass --no-critical-intervals to evaluate full episodes."
            )
        intervals_paths = [Path(p) for p in args.intervals]
        if len(intervals_paths) != len(input_paths):
            raise ValueError(
                f"--intervals count ({len(intervals_paths)}) must match --inputs count ({len(input_paths)})."
            )
        for ip in intervals_paths:
            if not ip.exists():
                raise FileNotFoundError(f"Intervals file not found: {ip}")

    # ---- Compute --------------------------------------------------
    print("Loading and concatenating inputs ...")
    joint_predictions_list = []
    joint_targets_list = []
    joint_episode_ends_list = []
    critical_intervals_joint: list[list[tuple]] = []

    t_offset = 0
    for i, input_path in enumerate(input_paths):
        print(f"[{i+1}/{len(input_paths)}] Loading input data from {input_path} ...")
        predictions_i, targets_i, episode_ends_i, states_i = load_input_data(
            input_path, load_states=needs_states
        )
        print(f"  Predictions shape: {predictions_i.shape}")
        print(f"  Targets shape:     {targets_i.shape}")
        print(f"  Episodes:          {len(episode_ends_i)}")

        # ---- Pose conversion --------------------------------------
        if needs_states:
            print(f"  Converting actions from '{action_type}' to absolute pose ...")
            predictions_i, targets_i = convert_to_absolute(
                predictions_i, targets_i, states_i, action_type
            )
            print("  Conversion done.")

        # ---- Load critical intervals ------------------------------
        if config.use_critical_intervals:
            intervals_path = intervals_paths[i]
            print(f"  Loading critical intervals from {intervals_path} ...")
            critical_intervals_i = load_critical_intervals(
                intervals_path, len(episode_ends_i)
            )
            labelled = sum(1 for ivs in critical_intervals_i if ivs)
            print(f"    {labelled}/{len(episode_ends_i)} episodes have labelled intervals")
        else:
            critical_intervals_i = [[] for _ in range(len(episode_ends_i))]

        # ---- Append into joint arrays ----------------------------
        joint_predictions_list.append(predictions_i)
        joint_targets_list.append(targets_i)
        joint_episode_ends_list.append(episode_ends_i + t_offset)
        critical_intervals_joint.extend(critical_intervals_i)
        t_offset += predictions_i.shape[0]

    predictions = np.concatenate(joint_predictions_list, axis=0)
    targets = np.concatenate(joint_targets_list, axis=0)
    episode_ends = np.concatenate(joint_episode_ends_list, axis=0)

    print("Computing Critical Interval MSE ...")
    evaluator = CriticalIntervalMSE(config)
    errors, out_episode_ends, out_interval_ends = evaluator.compute(
        predictions, targets, episode_ends, critical_intervals_joint
    )

    # ---- Save -----------------------------------------------------
    print(f"Saving results to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(output_path, errors, out_episode_ends, out_interval_ends)

    txt_path = generate_summary(
        config, errors, out_episode_ends, out_interval_ends, output_path,
        action_type=action_type,
        inputs=input_paths,
    )
    print(f"Summary saved to {txt_path}")


if __name__ == "__main__":
    main()

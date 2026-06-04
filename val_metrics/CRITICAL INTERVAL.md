# Critical Interval MSE

`val_metrics` evaluates policy prediction errors on labelled critical
intervals. It can optionally apply horizon truncation, temporal ensembling, and
DTW alignment before reporting timestep-level errors.

## Install

```bash
pip install ".[metrics]"
```

## Run

```bash
python -m val_metrics.validate examples/predictions_3episodes.h5 \
  --intervals examples/critical_intervals_3episodes.json \
  --output results/ci_mse_demo_results.hdf5
```

Multiple prediction files can be evaluated jointly. Provide one interval JSON
for each input:

```bash
python -m val_metrics.validate run_a.h5 run_b.h5 \
  --intervals run_a_intervals.json run_b_intervals.json \
  --output results/joint_ci_mse.hdf5
```

## Input HDF5

Required datasets:

- `predictions`: `[T_total, H, D]` predicted action chunks.
- `targets`: `[T_total, H, D]` ground-truth action chunks.
- `episode_ends`: cumulative episode end indices.

Optional dataset:

- `proprio`: robot state at each timestep. Required for relative action types.

## Critical interval JSON

```json
[
  {
    "episode": 0,
    "intervals": [
      { "label": "grasp", "start": 12, "end": 24 }
    ]
  }
]
```

`start` and `end` are timestep indices. Float values are treated as seconds and
converted using the 10 Hz convention used by the example files.

## Main options

| Argument | Default | Description |
|---|---:|---|
| `inputs` | required | One or more HDF5 files. |
| `--intervals` | `None` | One or more critical interval JSON files. Required unless `--no-critical-intervals` is set. |
| `--output` | `<first_input>_results.hdf5` | Output HDF5 path. |
| `--action-type` | `absolute` | `absolute`, `xvla_relative`, `openpi_relative`, or `umi_relative`. |
| `--horizon-start` | `0` | Horizon truncation start index. |
| `--horizon-end` | `-1` | Horizon truncation end index. `-1` means full horizon. |
| `--ensemble-horizon` | `1` | Temporal ensemble window. `1` disables averaging. |
| `--interpolation-factor` | `5` | DTW interpolation factor. |
| `--dtw-window-size` | `None` | Optional Sakoe-Chiba window size before interpolation scaling. |

## Feature toggles

All features are enabled by default unless disabled:

| Flag | Effect |
|---|---|
| `--no-critical-intervals` | Evaluate full episodes. |
| `--no-horizon-truncation` | Use the full action horizon. |
| `--no-temporal-ensemble` | Skip temporal ensembling. |
| `--no-dtw` | Use plain MSE instead of DTW distance. |

## Output

The HDF5 output contains:

- `errors`: selected scalar timestep errors.
- `episode_ends`: cumulative episode boundaries in `errors`.
- `interval_ends`: per-episode cumulative interval boundaries.

A `.txt` file with the same stem is also written. It reports total evaluated
timesteps, mean, median, 1st and 99th percentiles, trimmed mean, and per-interval
mean errors.

## Plotting

```bash
python -m val_metrics.plot_validation_results \
  results/ci_mse_demo_results.txt \
  --output results/plots
```

# critical interval mse

Tools for annotating critical intervals in robot demonstrations and evaluating
policy predictions with Critical Interval MSE (CI-MSE).

> Paper(coming soon)  
> [Project page](https://ci-mse.github.io/)

![Teaser](docs/assets/teaser.jpg)

## What is included

- `.agents/skills/critical-interval-annotator`: zero-shot local annotation of
  critical intervals in robot manipulation videos.
- `vlm_annotator`: deprecated Gemini and Vertex AI annotator.
- `val_metrics`: computes CI-MSE from model prediction HDF5 files and critical
  interval annotations.
- `examples`: small validation inputs for a runnable CI-MSE demo.

## Installation

Create a fresh Python environment, then install the subset you need:

```bash
pip install ".[metrics]"
pip install ".[anno]"
pip install ".[all]"
```

Use `.[metrics]` for CI-MSE evaluation only. Use `.[anno]` when you need the
local annotation dependencies (NumPy, OpenCV, and zarr). Use `.[all]` when you
need both.

Local annotation also requires FFmpeg on `PATH`. The annotator skill
materializes one episode at a time and inspects that video locally.

LeRobotDataset v2.1 data format is supported by default. 

## CI-MSE demo

The repository includes a tiny example prediction file and critical interval
annotation file:

```bash
python -m val_metrics.validate examples/predictions_3episodes.h5 \
  --intervals examples/critical_intervals_3episodes.json \
  --output results/ci_mse_demo_results.hdf5
```

This writes:

- `results/ci_mse_demo_results.hdf5`: timestep-level error arrays.
- `results/ci_mse_demo_results.txt`: human-readable summary metrics.

Optional plotting:

```bash
python -m val_metrics.plot_validation_results \
  results/ci_mse_demo_results.txt \
  --output results/plots
```

## Training-time LeRobot validation

For LeRobotDataset v2.1 validation during training, use the importable metric
API. The first version is LeRobot-only and expects your policy callback to
return predicted action chunks shaped `[B, H, D]`.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from val_metrics import (
    CriticalIntervalMSEConfig,
    LeRobotCriticalIntervalMSE,
    LeRobotCriticalIntervalMSEConfig,
)

val_dataset = LeRobotDataset(
    repo_id="/path/to/lerobot-v2.1-dataset",
    delta_timestamps={"action": [i / 10 for i in range(30)]},
)   # delta_timestamps depends on actual fps and action chunk length

metric = LeRobotCriticalIntervalMSE(
    dataset=val_dataset,
    intervals_path="results/critical_intervals.json",
    config=LeRobotCriticalIntervalMSEConfig(
        ci_mse_config=CriticalIntervalMSEConfig(
            ensemble_horizon=4,
            use_dtw=True,
            interpolation_factor=5,
        ),
        target_key="action",
        batch_size=64,
        device="cuda",
    ),
)

result = metric.evaluate(lambda batch: policy.select_action(batch))
print(result["mean"])
```

The metric computes the LeRobot data indices required by the critical intervals
and `ensemble_horizon`, runs only those samples through `predict_fn`, then calls
the core CI-MSE evaluator.

For distributed training, call `evaluate_distributed()` on every rank:

```python
result = metric.evaluate_distributed(lambda batch: policy.select_action(batch))
```

Each rank predicts a shard of the required indices. The metric gathers
`(global_index, prediction, target)` payloads with `torch.distributed.all_gather_object`,
then rank 0 restores the episode timeline and computes CI-MSE. By default,
scalar summary metrics are broadcast back to every rank; raw arrays remain on
rank 0.

## Zero-shot critical-interval annotation

Annotate outcome-critical object-contact, precision, and high-risk intervals
with the local
[critical-interval-annotator](.agents/skills/critical-interval-annotator/SKILL.md)
skill. Prompt configs and labeled examples are not required. Each episode is
inspected locally by a fresh agent; the skill does not upload video or call an
annotation API.

In Cursor, ask the agent to use the skill:

```text
Use the critical-interval-annotator skill to annotate episodes 0 through 10 of
/path/to/lerobot-v2.1-dataset.
Task instruction: Place the cup on the coaster.
```

Supported local inputs are video files, LeRobotDataset v2.1 directories, and
zarr archives. Remote dataset downloads are disabled. The task instruction is
optional. When supplied, it helps identify the intended objects and target
state without changing the critical-interval criteria.

The skill writes a new result directory and one `episode-<id>.json` per
episode, then validates and merges those files:

```bash
python .agents/skills/critical-interval-annotator/scripts/validate_and_merge.py \
  --input-dir <result-dir> \
  --expected-episodes 0 1 2 3 4 5 6 7 8 9 10 \
  --output <result-dir>/annotations.json \
  --seconds-output <result-dir>/annotations_seconds.json
```

That produces:

- `annotations_seconds.json`: auditable second-based boundaries.
- `annotations.json`: pipeline-compatible entries whose `start` and `end` are
  frame indices. Pass this file to `val_metrics`.

Dataset episode indices are preserved. For a dataset episode, the assigned
agent materializes only that episode with
`.agents/skills/critical-interval-annotator/scripts/materialize_episode.py`
before inspecting frames.

The Gemini and Vertex AI annotator in `vlm_annotator/` is deprecated. Its
previous workflow is recorded in the
[deprecated VLM annotator guide](vlm_annotator/VLM_LABELLER_GUIDE.md).

## Data formats

### Prediction HDF5

`val_metrics` expects:

- `predictions`: array shaped `[T_total, H, D]`.
- `targets`: array shaped `[T_total, H, D]`.
- `episode_ends`: cumulative episode end indices.
- `proprio`: required only when `--action-type` is a relative action format.

Supported action types:

- `absolute`
- `xvla_relative`
- `openpi_relative`
- `umi_relative`

### Critical interval JSON

The annotator and evaluator use a list of episode entries:

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

`start` and `end` are timestep indices. If a JSON file uses seconds as floats,
`val_metrics` converts them with the default 10 Hz convention used by the
release examples.

### Validation output HDF5

`val_metrics` writes:

- `errors`: scalar timestep errors selected by the CI-MSE pipeline.
- `episode_ends`: cumulative episode end indices in the output error array.
- `interval_ends`: per-episode cumulative interval end indices.

## More documentation

- [Critical interval annotator skill](.agents/skills/critical-interval-annotator/SKILL.md)
- [Deprecated VLM annotator guide](vlm_annotator/VLM_LABELLER_GUIDE.md)
- [CI-MSE guide](val_metrics/CRITICAL%20INTERVAL.md)

## License

MIT. See [LICENSE](LICENSE).

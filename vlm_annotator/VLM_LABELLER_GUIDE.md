# VLM Annotator Guide (deprecated)

> **Deprecated.** Use the
> [critical-interval-annotator](../.agents/skills/critical-interval-annotator/SKILL.md)
> skill for zero-shot critical-interval labeling. Ask the agent to annotate
> local video files or dataset episodes; see the
> [README annotation section](../README.md#zero-shot-critical-interval-annotation).
> The skill writes `annotations.json` with frame-index boundaries for
> `val_metrics`, plus `annotations_seconds.json`. The Gemini and Vertex AI
> workflow below is kept for reference and is no longer the supported
> annotation path.

The deprecated VLM annotator converts robot demonstration episodes to videos,
sends them to Gemini or Vertex AI, and stores critical interval annotations as
JSON.

## Install

```bash
pip install ".[anno]"
```

FFmpeg must also be installed and available on `PATH`.

This release targets LeRobotDataset v2.1 datasets via `lerobot>=0.3.2,<0.4`.
For zarr/UMI data, install the same `anno` extra and use `--demo_type zarr`.

## Quick start

Vertex AI backend:

```bash
export GOOGLE_CLOUD_PROJECT="your-google-cloud-project"
export GOOGLE_CLOUD_LOCATION="us-central1"

python -m vlm_annotator.gemini_annotator \
  --demo_path /path/to/lerobot-v2.1-dataset \
  --demo_type lerobot \
  --prompt_config vlm_annotator/prompts/PlaceCupByCoaster.json \
  --episode_idx "[0, 0]" \
  --output_dir results/vlm_annotations
```

Gemini API backend:

```bash
export GEMINI_API_KEY="your-gemini-api-key"

python -m vlm_annotator.gemini_annotator \
  --backend gemini \
  --demo_path /path/to/demo.zarr.zip \
  --demo_type zarr \
  --prompt_config vlm_annotator/prompts/PlaceCupByCoaster.json \
  --episode_idx "[0, 0]" \
  --output_dir results/vlm_annotations
```

## CLI options

| Option | Default | Description |
|---|---:|---|
| `--demo_path` | required | Path or repo id for a LeRobot v2.1 dataset, or path to a zarr archive. |
| `--demo_type` | `zarr` | `zarr` or `lerobot`. |
| `--prompt_config` | `vlm_annotator/prompts/PlaceCupByCoaster.json` | Prompt JSON with task and interval descriptions. |
| `--episode_idx` | `[0, 0]` | Inclusive episode range written as two integers. |
| `--downsample` | `3` | Video downsample factor. |
| `--model` | `gemini-3-pro-preview` | Gemini model name. |
| `--few_shot` | off | Use prompt examples as few-shot demonstrations. |
| `--backend` | `vertex` | `vertex` or `gemini`. |
| `--vertex_project` | env | Google Cloud project for Vertex AI. |
| `--vertex_location` | env or `us-central1` | Vertex AI region. |
| `--example_source` | same as `--demo_path` | Optional dataset path for few-shot examples. |
| `--output_dir` | `results` | Directory for JSON and TXT output. |
| `--verbose` | off | Print the constructed text prompt. |

## Prompt config

Prompt files live in `vlm_annotator/prompts/`. Each file defines one task and
the semantic intervals to label:

```json
{
  "task": "Pick up the cup from the table and place it by the coaster.",
  "intervals": [
    {
      "start": "The gripper has not yet grabbed the cup.",
      "end": "The gripper has just fully gripped the cup."
    }
  ],
  "examples": [
    {
      "episode": 0,
      "intervals": [
        { "label": "grasp", "start": 1.2, "end": 2.4 }
      ]
    }
  ]
}
```

`examples` are used only with `--few_shot`. Example interval times are seconds.
The saved output converts model-returned seconds to timestep indices using the
episode stream FPS.

## Output

The annotator writes:

- `test_zero_shot_<model>.txt` or `test_few_shot_<model>.txt`: raw model output.
- `test_zero_shot_<model>.json` or `test_few_shot_<model>.json`: parsed entries:

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

## Batch annotation

`vlm_annotator/run_vlm_labeller.sh` is a template batch runner. Set `DEMO_PATH`
and update the task episode ranges before using it:

```bash
DEMO_PATH=/path/to/lerobot-v2.1-dataset \
  bash vlm_annotator/run_vlm_labeller.sh
```

## Merge results

```bash
python -m vlm_annotator.merge_results \
  --results_dir results/vlm_annotations \
  --out results/critical_intervals.json
```

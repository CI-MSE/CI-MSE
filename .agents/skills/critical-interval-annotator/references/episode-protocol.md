# Episode annotation protocol

This protocol is for the fresh subagent assigned to exactly one episode. Do not inspect any other episode and do not call an external API or upload the video.

## Inputs

The orchestration task provides:

- one video path, or one dataset path plus episode index;
- a unique non-negative integer episode identifier;
- an output JSON path;
- an optional natural-language task instruction.

For a local zarr or LeRobot v2.1 dataset, materialize only the assigned episode with `scripts/materialize_episode.py`. Pass the dataset type, existing local dataset path, episode index, temporary output path, and desired downsample factor. The self-contained helper reads zarr arrays or LeRobot metadata/video files directly; it does not import repository modules or use a dataset hub. It reports original FPS, frame count, and duration; retain the original FPS and frame count for output conversion. Never edit the source dataset.

## Inspect locally

Use local tools such as `ffprobe` and `ffmpeg` to determine duration and extract frames or short clips into a temporary directory. Inspect those derived images yourself with the local image-viewing tool.

Treat source time as authoritative: frame `i` is at `i / fps` seconds. Downsampling does not change this mapping. For a downsampled inspection MP4, ignore any encoded tail beyond `total_frames / fps`; it is an artifact of the final sampled frame.

First inspect the full timeline at a coarse cadence sufficient to find contacts, target approaches, retries, and distinct subgoals. Then inspect denser frames around every candidate boundary; use approximately 0.1--0.25 second spacing when the source supports it. Scan the entire episode before applying the retry rule. Do not label from filenames, metadata, task text, or a single still image alone.

The main agent is prohibited from seeing the video, frames, contact sheets, or extracted clips. Keep all visual inspection and temporary visual artifacts inside this episode task. Store temporary media outside the result directory and remove it when practical; never remove source data.

## Critical interval definition

Identify the shortest intervals that most directly determine whether the manipulation succeeds or fails.

An interval is critical when it contains at least one of these:

1. **Object interaction.** The robot makes an imminent purposeful grasp, push, pull, or other contact. Start approximately 5 cm before contact, or at the final deliberate approach when distance is not visually reliable. End when the intended contact is established or completed.
2. **Precision near a target.** The robot performs fine positioning, orientation, placement, insertion, or alignment near a required target. Start approximately 10 cm from the target, or at the visible transition from coarse transport to careful alignment. End at a stable intended pose, achieved insertion/alignment, or clear failure.
3. **High difficulty or risk.** The robot manipulates a fragile, deformable, unstable, or easily dropped object; traverses a narrow constraint; maintains delicate contact; overcomes resistance; or avoids a close obstacle. Include only the visibly risky portion.

Do not annotate long unconstrained transit, idle stabilization after success, homing/reset, exploratory or random positioning unrelated to the goal, ordinary non-contact motion, or post-action retreat unless retreat is required for success.

## Boundaries and retries

- One interval represents one coherent critical action.
- Merge adjacent or overlapping phases of one continuous action, such as final approach plus grasp closure.
- Separate distinct actions when non-critical motion occurs between them.
- Exclude unnecessary lead-in and follow-through. Intervals must not overlap and must be chronological.
- For repeated attempts at the same immediate subgoal, keep only the final attempt. A new attempt begins after disengagement, backing away, reset, or a clearly renewed approach.
- For multiple distinct required subgoals, keep the final attempt for each subgoal.
- If success is unclear, use the last visible attempt.
- If no critical interval is visible, return an empty list.

Use the optional task instruction only to infer the intended goal, objects, and target state. Apply the criteria independently; do not label the full instructed task by default.

## Output

Estimate boundaries from visible motion, contact state, and object motion. Round to the nearest 0.1 seconds and ensure `0.0 <= start < end <= duration_seconds`.

Write one UTF-8 JSON object atomically to the assigned output path:

```json
{
  "episode": 0,
  "source": "<provided source path or dataset-path#episode-index>",
  "duration_seconds": 12.3,
  "fps": 30.0,
  "total_frames": 369,
  "intervals": [
    {
      "label": "secure grasp of the cup",
      "start": 2.3,
      "end": 4.7
    }
  ]
}
```

Use only these top-level and interval fields. `episode` must be the provided integer identifier. `fps` and `total_frames` describe the original episode, not the downsampled inspection encoding. Use concise, action-specific labels. JSON numbers do not preserve display precision, but write one decimal digit in the serialized text when possible. Do not include reasoning, confidence, frame paths, free-form inspection notes, or task instructions in the JSON; the concise action label is permitted and required.

Before finishing, parse the file locally and verify its schema, finite timestamps, duration bounds, chronological order, and lack of overlap. Return only a short completion status and the JSON path to the main agent.

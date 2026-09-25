"""Zero-shot Gemini annotation for generic critical manipulation intervals.

Unlike :mod:`vlm_annotator.gemini_annotator`, this entry point does not require
a task-specific prompt configuration or labeled examples.  It asks the model to
discover every outcome-critical interval directly from the video.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from .gemini_annotator import (
        _create_client,
        _get_generate_config,
        _intervals_sec_to_index,
        _video_part,
        create_video_stream,
        episode_to_mp4,
        load_demo_data,
        parse_episode_range,
        transcode_to_h264,
    )
except ImportError:  # Support ``python vlm_annotator/gemini_zero_shot_annotator.py``.
    from gemini_annotator import (  # type: ignore[no-redef]
        _create_client,
        _get_generate_config,
        _intervals_sec_to_index,
        _video_part,
        create_video_stream,
        episode_to_mp4,
        load_demo_data,
        parse_episode_range,
        transcode_to_h264,
    )


ZERO_SHOT_PROMPT = """You are given a short video of a robot performing a manipulation task.

Your task is to identify and timestamp the critical time intervals that most directly determine whether the task succeeds or fails.

Timestamp rules:
- Use seconds relative to the start of the video.
- Round every timestamp to the nearest 0.1 seconds.
- Ensure 0.0 <= start < end <= video duration.
- Base interval boundaries on visible evidence.

An interval is critical if it contains at least one of the following:

1. Object interaction
   - The robot approaches an object for an imminent grasp, push, pull, or other purposeful contact.
   - Start shortly before contact becomes imminent: approximately 5 cm before direct contact, or at the beginning of the final deliberate approach when distance cannot be estimated reliably.
   - End when the intended contact operation is established or completed, such as when the object is securely grasped, the push or pull is complete, or the required contact ends.

2. Precision near a task target
   - The robot performs fine positioning, orientation, placement, insertion, or alignment near a required target.
   - Start approximately 10 cm from the target, or when motion visibly changes from coarse transport to careful alignment.
   - End when the object reaches a stable intended pose, insertion or alignment is achieved, or the attempt clearly fails.

3. High-difficulty or high-risk manipulation
   - The robot performs an action with a high risk of failure, such as manipulating a fragile, deformable, unstable, or easily dropped object; moving through a narrow or constrained path; maintaining delicate contact; overcoming resistance; or avoiding a nearby obstacle.
   - Include only the portion during which the difficulty or risk is visibly present.

Do NOT annotate:
- Long, unconstrained transit motions.
- Idle periods or stabilization after an action has already succeeded.
- Robot homing or reset motions.
- Exploratory or random positioning unrelated to the task goal.
- Ordinary motion that is neither near contact nor precision-sensitive.
- A final retreat after the critical action is complete, unless retreat itself is necessary for task success.

Interval construction rules:
- Each interval must correspond to one coherent critical action.
- Merge adjacent or overlapping critical phases when they form one continuous action, such as a final grasp approach followed immediately by grasp closure.
- Keep distinct actions separate when non-critical motion occurs between them, such as grasping an object and later precisely placing it.
- Use the shortest interval that fully captures the critical action; exclude unnecessary lead-in and follow-through.
- List intervals in chronological order.
- Do not return duplicate or overlapping intervals.

Retries and multiple attempts:
- An attempt is a continuous effort toward the same immediate subgoal.
- If the robot fails and retries the same subgoal, annotate only the final attempt.
- A new attempt begins when the robot disengages, backs away, resets, or clearly starts a new approach.
- If the video contains multiple distinct required subgoals, annotate the final attempt for each subgoal.
- If success is unclear, annotate the last visible attempt.

Uncertainty:
- Infer the task goal from the video.
- If exact distances or boundaries cannot be determined, use visible changes in motion, contact state, object motion, and robot behavior to make the best estimate.
- If no critical interval is visible, return an empty "intervals" array.

Output requirements:
- Return ONLY one valid JSON object, without explanations, comments, Markdown, or code fences.
- Use exactly one digit after the decimal point for every timestamp, including whole seconds (for example, 2.0).
- Use concise, action-specific labels.
- Do not include fields other than "intervals", "label", "start", and "end".

The output must follow this structure:
{
  "intervals": [
    {
      "label": "<concise description of the critical robot action>",
      "start": 2.3,
      "end": 4.7
    }
  ]
}

Now annotate the critical intervals in this video."""


def build_zero_shot_prompt(task_instruction: str = "") -> str:
    """Add optional task context without changing the annotation criteria."""
    task_instruction = task_instruction.strip()
    if not task_instruction:
        return ZERO_SHOT_PROMPT

    task_context = (
        "\n\nTask instruction:\n"
        f"{task_instruction}\n\n"
        "Use this instruction to infer the intended task goal, required objects, "
        "and target state. Apply the critical-interval rules below independently; "
        "do not assume that the entire instructed task is critical."
    )
    insertion_point = "\n\nYour task is to identify"
    return ZERO_SHOT_PROMPT.replace(
        insertion_point,
        task_context + insertion_point,
        1,
    )


def parse_annotation_response(response: str, video_duration_sec: float) -> dict[str, Any]:
    """Parse and validate a model response before converting seconds to frames."""
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {"intervals"}:
        raise ValueError('Response must be an object containing only "intervals"')
    if not isinstance(result["intervals"], list):
        raise ValueError('"intervals" must be a list')

    previous_end = -math.inf
    normalized = []
    for i, interval in enumerate(result["intervals"]):
        if not isinstance(interval, dict) or set(interval) != {"label", "start", "end"}:
            raise ValueError(f"Interval {i} must contain only label, start, and end")
        label = interval["label"]
        start = interval["start"]
        end = interval["end"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Interval {i} has an empty or invalid label")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (start, end)):
            raise ValueError(f"Interval {i} timestamps must be numbers")
        start = round(float(start), 1)
        end = round(float(end), 1)
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"Interval {i} timestamps must be finite")
        if not (0.0 <= start < end <= video_duration_sec + 0.05):
            raise ValueError(
                f"Interval {i} [{start}, {end}] is outside video duration "
                f"[0.0, {video_duration_sec:.1f}]"
            )
        if start < previous_end:
            raise ValueError(f"Interval {i} overlaps or is out of chronological order")
        normalized.append({"label": label.strip(), "start": start, "end": end})
        previous_end = end

    return {"intervals": normalized}


def generate_annotation(
    *,
    client,
    backend: str,
    model: str,
    h264_path: str,
    prompt: str = ZERO_SHOT_PROMPT,
    max_retries: int = 5,
) -> str:
    """Send one video and the fixed zero-shot prompt to Gemini."""
    video_part = _video_part(client, backend, h264_path)
    config = _get_generate_config(model)
    config.response_mime_type = "application/json"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[video_part, prompt],
                config=config,
            )
            print(f"Gemini Response (usage = [{response.usage_metadata}]):")
            print(response.text)
            return response.text
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_sec = min(2 ** attempt, 60)
            print(
                f"[WARN] generate_content failed on attempt {attempt}/{max_retries}: "
                f"{exc}. Retrying in {wait_sec}s..."
            )
            time.sleep(wait_sec)
    raise RuntimeError("Unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate generic critical manipulation intervals with Gemini, without examples."
    )
    parser.add_argument(
        "--demo_path",
        required=True,
        help="Path to a zarr archive or a LeRobot v2.1 dataset directory/repo_id.",
    )
    parser.add_argument("--demo_type", choices=["zarr", "lerobot"], default="zarr")
    parser.add_argument(
        "--episode_idx",
        default="[0, 0]",
        help='Inclusive episode range, for example "[0, 10]".',
    )
    parser.add_argument("--downsample", type=int, default=3)
    parser.add_argument("--model", default="gemini-3-pro-preview")
    parser.add_argument("--backend", choices=["gemini", "vertex"], default="vertex")
    parser.add_argument("--vertex_project", default="")
    parser.add_argument("--vertex_location", default="")
    parser.add_argument(
        "--task_instruction",
        default="",
        help=(
            "Optional natural-language description of the intended manipulation "
            "task, objects, or target state."
        ),
    )
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--verbose", action="store_true", help="Print the zero-shot prompt.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.downsample < 1:
        raise ValueError("--downsample must be at least 1")

    epi_start, epi_end = parse_episode_range(args.episode_idx)
    prompt = build_zero_shot_prompt(args.task_instruction)
    if args.verbose:
        print("[VERBOSE] Zero-shot prompt sent to the model:")
        print("-" * 60)
        print(prompt)
        print("-" * 60)

    zarr_group, dataset = load_demo_data(args.demo_type, args.demo_path)
    if zarr_group is not None:
        total_episodes = len(zarr_group["meta/episode_ends"])
    else:
        total_episodes = dataset.num_episodes
    if epi_end >= total_episodes:
        raise ValueError(
            f"Episode range ends at {epi_end}, but dataset has {total_episodes} episodes"
        )
    print(f"[INFO] Loaded {args.demo_type} with {total_episodes} episodes")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if args.backend == "gemini" and not api_key:
        raise ValueError("Gemini backend requires the GEMINI_API_KEY environment variable")
    client = _create_client(
        args.backend,
        api_key=api_key,
        project=args.vertex_project,
        location=args.vertex_location,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_txt = output_dir / f"test_generic_zero_shot_{args.model}.txt"
    results_json = output_dir / f"test_generic_zero_shot_{args.model}.json"

    results: list[dict[str, Any]] = []
    completed_episodes: set[int] = set()
    if results_json.exists():
        try:
            existing = json.loads(results_json.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                raise ValueError("top-level value is not a list")
            results = existing
            completed_episodes = {
                item["episode"]
                for item in existing
                if isinstance(item, dict) and isinstance(item.get("episode"), int)
            }
            print(f"[INFO] Resuming with {len(completed_episodes)} completed episodes")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Cannot resume from {results_json}: {exc}") from exc

    try:
        for idx in range(epi_start, epi_end + 1):
            if idx in completed_episodes:
                print(f"[INFO] Episode {idx} already completed, skipping")
                continue

            with tempfile.TemporaryDirectory(prefix="ci_zero_shot_") as tmp_dir:
                mp4_path = Path(tmp_dir) / f"episode_{idx}.mp4"
                h264_path = Path(tmp_dir) / f"episode_{idx}_h264.mp4"
                stream = create_video_stream(
                    args.demo_type,
                    args.demo_path,
                    idx,
                    zarr_group=zarr_group,
                    dataset=dataset,
                )
                try:
                    episode_fps = stream.fps
                    video_duration_sec = stream.total_frames / stream.fps
                    episode_to_mp4(stream, str(mp4_path), downsample=args.downsample)
                finally:
                    stream.close()
                transcode_to_h264(str(mp4_path), str(h264_path))
                response = generate_annotation(
                    client=client,
                    backend=args.backend,
                    model=args.model,
                    h264_path=str(h264_path),
                    prompt=prompt,
                )

            with results_txt.open("a", encoding="utf-8") as file:
                file.write(f"--- Episode {idx} ---\n{response}\n\n")

            try:
                parsed = parse_annotation_response(response, video_duration_sec)
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"[ERROR] Episode {idx} returned an invalid annotation: {exc}")
                continue

            results.append(
                {
                    "episode": idx,
                    "intervals": _intervals_sec_to_index(
                        parsed["intervals"], episode_fps
                    ),
                }
            )
            completed_episodes.add(idx)
            results_json.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[INFO] Results saved to {results_json}")
    finally:
        if zarr_group is not None and hasattr(zarr_group, "store"):
            zarr_group.store.close()


if __name__ == "__main__":
    main()

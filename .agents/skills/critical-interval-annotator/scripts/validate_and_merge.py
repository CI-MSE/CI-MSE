"""Validate and merge per-episode critical-interval annotations."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


TOP_LEVEL_FIELDS = {
    "episode",
    "source",
    "duration_seconds",
    "fps",
    "total_frames",
    "intervals",
}
INTERVAL_FIELDS = {"label", "start", "end"}


def validate_episode(data: Any, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != TOP_LEVEL_FIELDS:
        raise ValueError(f"{path}: expected exactly {sorted(TOP_LEVEL_FIELDS)}")

    episode = data["episode"]
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        raise ValueError(f"{path}: episode must be a non-negative integer")
    if not isinstance(data["source"], str) or not data["source"].strip():
        raise ValueError(f"{path}: source must be a non-empty string")

    duration = data["duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError(f"{path}: duration_seconds must be numeric")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{path}: duration_seconds must be finite and positive")

    fps = data["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError(f"{path}: fps must be numeric")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: fps must be finite and positive")

    total_frames = data["total_frames"]
    if (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames <= 0
    ):
        raise ValueError(f"{path}: total_frames must be a positive integer")
    source_duration = total_frames / fps
    if abs(duration - source_duration) > 0.11:
        raise ValueError(
            f"{path}: duration_seconds does not match total_frames / fps "
            f"({source_duration:.3f})"
        )

    intervals = data["intervals"]
    if not isinstance(intervals, list):
        raise ValueError(f"{path}: intervals must be a list")

    normalized_intervals = []
    previous_end = -math.inf
    for index, interval in enumerate(intervals):
        prefix = f"{path}: interval {index}"
        if not isinstance(interval, dict) or set(interval) != INTERVAL_FIELDS:
            raise ValueError(f"{prefix} must contain exactly {sorted(INTERVAL_FIELDS)}")
        label = interval["label"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{prefix} label must be a non-empty string")
        start, end = interval["start"], interval["end"]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (start, end)
        ):
            raise ValueError(f"{prefix} timestamps must be numeric")
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"{prefix} timestamps must be finite")
        start, end = round(start, 1), round(end, 1)
        if not (0.0 <= start < end <= duration + 0.05):
            raise ValueError(f"{prefix} timestamps are outside [0.0, {duration:.1f}]")
        if start < previous_end:
            raise ValueError(f"{prefix} overlaps or is out of chronological order")
        normalized_intervals.append(
            {"label": label.strip(), "start": start, "end": end}
        )
        previous_end = end

    return {
        "episode": episode,
        "source": data["source"].strip(),
        "duration_seconds": round(duration, 1),
        "fps": fps,
        "total_frames": total_frames,
        "intervals": normalized_intervals,
    }


def to_pipeline_episode(item: dict[str, Any], path: Path) -> dict[str, Any]:
    intervals = []
    for index, interval in enumerate(item["intervals"]):
        start = round(interval["start"] * item["fps"])
        end = round(interval["end"] * item["fps"])
        source_duration = item["total_frames"] / item["fps"]
        if (
            end > item["total_frames"]
            and interval["end"] <= source_duration + 0.0500001
        ):
            # A true tail time such as 12.266 s correctly rounds to 12.3 s,
            # which can round back to one frame beyond the half-open source.
            end = item["total_frames"]
        if start >= end:
            raise ValueError(
                f"{path}: interval {index} collapses after conversion to frame indices"
            )
        if start < 0 or end > item["total_frames"]:
            raise ValueError(
                f"{path}: interval {index} converts outside source frame range "
                f"[0, {item['total_frames']}]"
            )
        intervals.append({"label": interval["label"], "start": start, "end": end})
    return {"episode": item["episode"], "intervals": intervals}


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds-output", type=Path)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output = args.output.resolve()
    seconds_output = (
        args.seconds_output.resolve()
        if args.seconds_output
        else output.with_name(f"{output.stem}_seconds{output.suffix}")
    )
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    expected_episodes = set(args.expected_episodes)
    if len(expected_episodes) != len(args.expected_episodes) or any(
        episode < 0 for episode in expected_episodes
    ):
        raise ValueError("--expected-episodes must contain unique non-negative integers")

    episodes = []
    episode_sources: dict[str, Path] = {}
    for path in sorted(input_dir.glob("episode-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
        episode = validate_episode(data, path)
        expected_name = f"episode-{episode['episode']}.json"
        if path.name != expected_name:
            raise ValueError(
                f"{path}: declared episode requires filename {expected_name!r}"
            )
        identity = f"{type(episode['episode']).__name__}:{episode['episode']}"
        if identity in episode_sources:
            raise ValueError(
                f"Duplicate episode {episode['episode']!r} in "
                f"{episode_sources[identity]} and {path}"
            )
        episode_sources[identity] = path
        episodes.append(episode)
        atomic_write_json(path, episode)

    if not episodes:
        raise ValueError(f"No per-episode JSON files found in {input_dir}")
    actual_episodes = {item["episode"] for item in episodes}
    if actual_episodes != expected_episodes:
        missing = sorted(expected_episodes - actual_episodes)
        unexpected = sorted(actual_episodes - expected_episodes)
        raise ValueError(
            f"Episode set mismatch; missing={missing}, unexpected={unexpected}"
        )

    episodes.sort(key=lambda item: item["episode"])
    pipeline_episodes = [
        to_pipeline_episode(item, episode_sources[f"int:{item['episode']}"])
        for item in episodes
    ]
    atomic_write_json(seconds_output, {"episodes": episodes})
    atomic_write_json(output, pipeline_episodes)
    print(
        f"Validated and merged {len(episodes)} episode(s) into {output} "
        f"and {seconds_output}"
    )


if __name__ == "__main__":
    main()

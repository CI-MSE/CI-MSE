"""Materialize one local zarr or LeRobot episode as an H.264 MP4.

The implementation is self-contained: it reads local dataset metadata and
media directly and never imports repository code or calls a network service.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LEROBOT_VIDEO_KEY = "observation.images.scene_left_0"


@dataclass(frozen=True)
class EpisodeSource:
    video_path: Path
    fps: float
    total_frames: int
    video_key: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required metadata file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _read_lerobot_episode(dataset_path: Path, episode: int, video_key: str) -> EpisodeSource:
    info = _read_json(dataset_path / "meta" / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            "Only local LeRobot v2.1 datasets are supported; found "
            f"{info.get('codebase_version')!r}"
        )

    total_episodes = info.get("total_episodes")
    if not isinstance(total_episodes, int) or episode >= total_episodes:
        raise ValueError(
            f"Episode {episode} is outside dataset with {total_episodes!r} episodes"
        )
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("meta/info.json must contain a positive numeric fps")

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json must contain a features object")
    video_features = [
        key
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    if video_key not in video_features:
        available = ", ".join(sorted(video_features)) or "none"
        raise ValueError(f"Video key {video_key!r} is unavailable; choices: {available}")

    episode_record = None
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    try:
        with episodes_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {episodes_path}:{line_number}: {exc}"
                    ) from exc
                if record.get("episode_index") == episode:
                    episode_record = record
                    break
    except FileNotFoundError as exc:
        raise ValueError(f"Required metadata file does not exist: {episodes_path}") from exc
    if episode_record is None:
        raise ValueError(f"Episode {episode} is missing from {episodes_path}")
    total_frames = episode_record.get("length")
    if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames <= 0:
        raise ValueError(f"Episode {episode} has invalid length {total_frames!r}")

    chunks_size = info.get("chunks_size")
    video_template = info.get("video_path")
    if isinstance(chunks_size, bool) or not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError("meta/info.json must contain a positive integer chunks_size")
    if not isinstance(video_template, str) or not video_template:
        raise ValueError("meta/info.json must contain video_path")
    try:
        relative_video = video_template.format(
            episode_chunk=episode // chunks_size,
            video_key=video_key,
            episode_index=episode,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported LeRobot video_path template: {video_template}") from exc
    video_path = (dataset_path / relative_video).resolve()
    try:
        video_path.relative_to(dataset_path)
    except ValueError as exc:
        raise ValueError(f"Resolved video path escapes dataset root: {video_path}") from exc
    if not video_path.is_file():
        raise ValueError(f"Episode video does not exist locally: {video_path}")
    return EpisodeSource(video_path, float(fps), total_frames, video_key)


def _run_ffmpeg(input_path: Path, output_path: Path, fps: float, downsample: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_fps = fps / downsample
    command = ["ffmpeg", "-y", "-i", str(input_path)]
    if downsample > 1:
        command.extend(["-vf", f"fps={sampled_fps:.12g}"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ]
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=output_path.parent,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-20:])
        raise RuntimeError(f"ffmpeg failed with exit code {completed.returncode}:\n{tail}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create a usable output: {output_path}")


def _materialize_zarr(
    dataset_path: Path,
    episode: int,
    output_path: Path,
    fps: float,
    downsample: int,
) -> EpisodeSource:
    try:
        import cv2
        import numpy as np
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr materialization requires zarr, numpy, and opencv-python") from exc

    group = zarr.open(str(dataset_path), mode="r")
    try:
        episode_ends = group["meta/episode_ends"]
        if episode >= len(episode_ends):
            raise ValueError(
                f"Episode {episode} is outside zarr dataset with {len(episode_ends)} episodes"
            )
        start = 0 if episode == 0 else int(episode_ends[episode - 1])
        end = int(episode_ends[episode])
        total_frames = end - start
        if total_frames <= 0:
            raise ValueError(f"Episode {episode} contains no frames")
        frames = group["data/camera0_rgb"]
        first = np.asarray(frames[start])
        if first.ndim != 3 or first.shape[2] != 3:
            raise ValueError(f"Expected RGB frames, found shape {first.shape}")
        height, width = first.shape[:2]

        with tempfile.TemporaryDirectory(prefix="ci_zarr_") as temp_dir:
            intermediate = Path(temp_dir) / "episode.mp4"
            writer = cv2.VideoWriter(
                str(intermediate),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps / downsample,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {intermediate}")
            try:
                for frame_index in range(start, end, downsample):
                    frame = np.asarray(frames[frame_index])
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            _run_ffmpeg(intermediate, output_path, fps / downsample, 1)
    finally:
        if hasattr(group, "store") and hasattr(group.store, "close"):
            group.store.close()
    return EpisodeSource(output_path, fps, total_frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-type", choices=["zarr", "lerobot"], required=True)
    parser.add_argument("--demo-path", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--downsample", type=int, default=3)
    parser.add_argument(
        "--video-key",
        default=DEFAULT_LEROBOT_VIDEO_KEY,
        help="LeRobot video feature to inspect.",
    )
    parser.add_argument(
        "--zarr-fps",
        type=float,
        default=60.0,
        help="Source FPS for zarr datasets (default matches the original annotator).",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episode < 0:
        raise ValueError("--episode must be non-negative")
    if args.downsample < 1:
        raise ValueError("--downsample must be at least 1")
    if args.zarr_fps <= 0:
        raise ValueError("--zarr-fps must be positive")
    dataset_path = args.demo_path.resolve()
    if not dataset_path.exists():
        raise ValueError(
            "--demo-path must be an existing local path; remote dataset downloads are disabled"
        )
    output_path = args.output.resolve()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if args.demo_type == "lerobot":
        source = _read_lerobot_episode(dataset_path, args.episode, args.video_key)
        _run_ffmpeg(source.video_path, output_path, source.fps, args.downsample)
    else:
        source = _materialize_zarr(
            dataset_path,
            args.episode,
            output_path,
            args.zarr_fps,
            args.downsample,
        )

    print(
        json.dumps(
            {
                "video": str(output_path),
                "episode": args.episode,
                "fps": source.fps,
                "total_frames": source.total_frames,
                "duration_seconds": source.total_frames / source.fps,
                "video_key": source.video_key,
            }
        )
    )


if __name__ == "__main__":
    main()

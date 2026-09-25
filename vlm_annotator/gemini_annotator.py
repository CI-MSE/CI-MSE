"""
Usage:
    # Vertex AI (default; no file upload, uses inline video bytes):
    export GOOGLE_CLOUD_PROJECT="your-project-id"
    export GOOGLE_CLOUD_LOCATION="us-central1"   # optional
    python gemini_annotator.py --demo_path path/to/demo --demo_type lerobot ...

    # Gemini API (API key + file upload):
    export GEMINI_API_KEY="your_key_here"
    python gemini_annotator.py --backend gemini --demo_path path/to/demo.zarr.zip ...
"""

import argparse
import ast
import copy
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------- Episode -> MP4 (generic VideoStream) ----------

def episode_to_mp4(stream, video_out: str, downsample: int = 1) -> None:
    """Write episode frames from a VideoStream to an MP4 file."""
    import cv2

    ok, frame0 = stream.get_frame(0)
    if not ok:
        raise RuntimeError("Failed to read first frame from VideoStream")
    h, w, _ = frame0.shape
    fps = stream.fps / downsample
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))
    n = stream.total_frames
    for i in range(0, n, downsample):
        ok, frame = stream.get_frame(i)
        if not ok:
            break
        writer.write(frame)
    writer.release()


def transcode_to_h264(in_path: str, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        out_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"[INFO] Transcoded to H.264: {out_path}")


# ---------- Prompt config: load JSON and build instruction ----------

def load_prompt_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_substitution_map(prompt_data: Dict[str, Any]) -> Dict[str, str]:
    """Build placeholder -> value map for template substitution. Supports variable N intervals."""
    intervals = prompt_data["intervals"]
    N = len(intervals)
    mapping = {
        "task": prompt_data["task"],
        "num_intervals": str(N),
    }
    for i in range(N):
        mapping[f"interval_{i}_start"] = intervals[i]["start"]
        mapping[f"interval_{i}_end"] = intervals[i]["end"]
    return mapping


def get_default_template(N: int) -> str:
    """Build default prompt template for N intervals (variable)."""
    blocks = [
        "You are given a short robot manipulation video.",
        f"Your task is to annotate {N} specific time intervals in the video with timestamps in seconds, accurate to one decimal place.",
        "",
        f"The {N} intervals are defined semantically as:",
        "",
    ]
    for i in range(N):
        blocks.append(f"{i + 1}. Interval {i + 1}:")
        blocks.append("   - Start: \"{interval_" + str(i) + "_start}\"")
        blocks.append("   - End: \"{interval_" + str(i) + "_end}\"")
        blocks.append("")
    blocks.extend([
        "Requirements:",
        f"- Return exactly {N} intervals in a JSON object.",
        "- Each interval must have:",
        "  - \"label\": a short description.",
        "  - \"start\": start time in seconds, with exactly one decimal place (e.g., 2.3).",
        "  - \"end\": end time in seconds, with exactly one decimal place.",
        "- Timestamps must be within the video duration.",
        "- If you are uncertain, make your best estimate based on the visual evidence, but still output valid timestamps.",
        "- If there are retries, only return the last attempt's timestamps."
        "",
        "Output format:",
        "Return ONLY a valid JSON object, with no extra text, no explanations, and no Markdown code fences.",
        "",
        "The JSON format must be exactly:",
        "",
        "{\n  \"intervals\": [\n    {\n      \"label\": \"<description text>\",\n      \"start\": <time in seconds, 1 decimal place>,\n      \"end\": <time in seconds, 1 decimal place>\n    },\n    ... one object per interval above ...\n  ]\n}",
        "",
    ])
    return "\n".join(blocks)


def apply_template(template: str, mapping: Dict[str, str]) -> str:
    """Replace placeholders in template with values from mapping."""
    out = template
    for key, value in mapping.items():
        out = out.replace("{" + key + "}", value)
    return out


def build_instruction(prompt_data: Dict[str, Any]) -> str:
    """Build final instruction from prompt config (variable N intervals)."""
    mapping = build_substitution_map(prompt_data)
    if "template" in prompt_data:
        template = prompt_data["template"]
    else:
        N = len(prompt_data["intervals"])
        template = get_default_template(N)
    return apply_template(template, mapping)


def few_shot_example_message(example_intervals: List[Dict[str, float]]) -> str:
    """Build the 'example response' text for one few-shot example (variable N intervals)."""
    intervals_json = json.dumps([
        {"label": f"interval_{i+1}", "start": iv["start"], "end": iv["end"]}
        for i, iv in enumerate(example_intervals)
    ], indent=2)
    return (
        "\nHere is an example of how to label the intervals, and corresponding video in attachment.\n"
        "Example response:\n"
        f'{{"intervals": {intervals_json}}}\n'
    )


# ---------- Data loading: zarr vs lerobot ----------

def create_video_stream(demo_type: str, demo_path: str, episode_idx: int, zarr_group=None, dataset=None):
    """Create a VideoStream for the given demo_type and episode. Caller must pass either zarr_group (for zarr) or dataset (for lerobot)."""
    try:
        from .video_stream import VIDEO_STREAM_REGISTRY
    except ImportError:
        from video_stream import VIDEO_STREAM_REGISTRY

    if demo_type == "zarr":
        if zarr_group is None:
            raise ValueError("zarr_group required for demo_type zarr")
        return VIDEO_STREAM_REGISTRY["zarr"](zarr_group, episode_idx)
    elif demo_type == "lerobot":
        if dataset is None:
            raise ValueError("dataset required for demo_type lerobot")
        return VIDEO_STREAM_REGISTRY["lerobot"](dataset, episode_idx)
    else:
        raise ValueError(f"Unknown demo_type: {demo_type}")


def load_demo_data(demo_type: str, demo_path: str):
    """Load zarr group or lerobot dataset. Returns (zarr_group or None, dataset or None)."""
    if demo_type == "zarr":
        import zarr

        zarr_group = zarr.open(demo_path, mode="r")
        return zarr_group, None
    elif demo_type == "lerobot":
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=demo_path)
        return None, dataset
    else:
        raise ValueError(f"Unknown demo_type: {demo_type}")


def parse_episode_range(value: str) -> tuple[int, int]:
    """Parse an inclusive episode range like ``[0, 5]`` without executing code."""
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("--episode_idx must be a two-integer list, e.g. [0, 5]") from exc

    if (
        not isinstance(parsed, (list, tuple))
        or len(parsed) != 2
        or not all(isinstance(v, int) for v in parsed)
    ):
        raise ValueError("--episode_idx must be a two-integer list, e.g. [0, 5]")

    start, end = parsed
    if start < 0 or end < start:
        raise ValueError("--episode_idx must satisfy 0 <= start <= end")
    return start, end


# ---------- Gemini Interval Annotator ----------

def _get_generate_config(model: str):
    from google.genai import types

    if model == "gemini-3-pro-preview":
        return types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="high"))
    return types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=-1))


def _create_client(backend: str, api_key: str = "", project: str = "", location: str = ""):
    """Create genai Client. backend: 'gemini' (API key + file upload) or 'vertex' (inline video, no upload)."""
    from google import genai

    if backend == "vertex":
        proj = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        loc = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not proj:
            raise ValueError("Vertex AI requires project. Set --vertex_project or GOOGLE_CLOUD_PROJECT.")
        return genai.Client(vertexai=True, project=proj, location=loc)
    return genai.Client(api_key=api_key)


def _video_part(client, backend: str, h264_path: str):
    """
    Return a content part for the video at h264_path.
    - Gemini: upload file and return the file reference (after waiting for ACTIVE).
    - Vertex: read file as bytes and return Part.from_bytes (no upload).
    """
    from google.genai import types

    if backend == "vertex":
        with open(h264_path, "rb") as f:
            data = f.read()
        return types.Part.from_bytes(data=data, mime_type="video/mp4")
    # Gemini: upload and wait
    episode_video = client.files.upload(file=h264_path)
    while not episode_video.state or episode_video.state.name != "ACTIVE":
        print("[INFO] Processing file...")
        time.sleep(2)
        episode_video = client.files.get(name=episode_video.name)
    return episode_video


def _intervals_sec_to_index(intervals: List[Dict[str, float]], fps: int) -> List[Dict[str, int]]:
    rtn_interval = copy.deepcopy(intervals)
    for interval in rtn_interval:
        interval["start"] = round(interval["start"] * fps)
        interval["end"] = round(interval["end"] * fps)
    return rtn_interval

def _intervals_index_to_sec(intervals: List[Dict[str, int]], fps: int) -> List[Dict[str, float]]:
    rtn_interval = copy.deepcopy(intervals)
    for interval in rtn_interval:
        interval["start"] = interval["start"] / fps
        interval["end"] = interval["end"] / fps
    return rtn_interval


def _chat_send_with_retry(
    chat,
    message,
    config,
    max_retries: int = 10,
    initial_wait_sec: float = 2.0,
    backoff_factor: float = 2.0,
    max_wait_sec: float = 600.0,
):
    """Send chat message with retry/backoff for transient network issues."""
    for attempt in range(1, max_retries + 1):
        try:
            return chat.send_message(message=message, config=config)
        except Exception as e:
            if attempt == max_retries:
                raise
            wait_sec = min(initial_wait_sec * (backoff_factor ** (attempt - 1)), max_wait_sec)
            print(
                f"[WARN] chat.send_message failed on attempt {attempt}/{max_retries}: {e}. "
                f"Retrying in {wait_sec:.1f}s..."
            )
            time.sleep(wait_sec)


def gemini_annotator(
    api_key: str,
    model: str,
    instruction: str,
    h264_path: str,
    use_few_shot: bool = False,
    prompt_data: Optional[Dict[str, Any]] = None,
    demo_type: str = "zarr",
    demo_path: str = "",
    zarr_group=None,
    dataset=None,
    example_demo_path: str = "",
    example_zarr_group=None,
    example_dataset=None,
    downsample: int = 3,
    backend: str = "gemini",
    vertex_project: str = "",
    vertex_location: str = "",
) -> str:
    """
    Call Gemini to label critical intervals in the video at h264_path.
    If use_few_shot is True, prompt_data must contain "examples" and we send all examples + target in one request.
    backend: 'gemini' (API key + file upload) or 'vertex' (Vertex AI + inline video bytes).
    """
    client = _create_client(backend, api_key=api_key, project=vertex_project, location=vertex_location)
    config = _get_generate_config(model)

    if not use_few_shot or not prompt_data or not prompt_data.get("examples"):
        # Zero-shot: single video + instruction
        video_part = _video_part(client, backend, h264_path)
        response = client.models.generate_content(
            model=model,
            contents=[video_part, instruction],
            config=config,
        )
        print(f"Gemini Response (usage = [{response.usage_metadata}]):")
        print(response.text)
        return response.text

    # Few-shot: assemble all example videos + example responses + target video into one message.
    chat = client.chats.create(model=model)
    N = len(prompt_data["intervals"])
    one_shot_message = [instruction]

    ex_demo_path = example_demo_path or demo_path
    ex_zarr_group = example_zarr_group if example_zarr_group is not None else zarr_group
    ex_dataset = example_dataset if example_dataset is not None else dataset

    for ex in prompt_data["examples"]:
        ep_idx = ex["episode"]
        ex_intervals = ex["intervals"]
        if len(ex_intervals) != N:
            raise ValueError(f"Example episode {ep_idx} has {len(ex_intervals)} intervals, expected {N}")
        stream = create_video_stream(
            demo_type,
            ex_demo_path,
            ep_idx,
            zarr_group=ex_zarr_group,
            dataset=ex_dataset,
        )
        with tempfile.TemporaryDirectory(prefix="ci_mse_fewshot_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            try:
                fps = stream.fps
                example_mp4 = tmp_dir_path / "example_episode.mp4"
                example_h264 = tmp_dir_path / "example_episode_h264.mp4"
                episode_to_mp4(stream, str(example_mp4), downsample=downsample)
                transcode_to_h264(str(example_mp4), str(example_h264))
            finally:
                stream.close()
            example_video_part = _video_part(client, backend, str(example_h264))
        example_msg = few_shot_example_message(_intervals_index_to_sec(ex_intervals, fps))
        one_shot_message.extend([
            example_video_part,
            f"Example episode {ep_idx}.{example_msg}",
        ])
        print(f"[INFO] Prepared few-shot example episode {ep_idx}")

    episode_video_part = _video_part(client, backend, h264_path)
    follow_up = f"Please label the {N} critical intervals in this video as per the previous examples."
    one_shot_message.extend([episode_video_part, follow_up])
    response = _chat_send_with_retry(
        chat=chat,
        message=one_shot_message,
        config=config,
    )
    print(f"Gemini Response (usage = [{response.usage_metadata}]):")
    print(response.text)
    return response.text


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Convert episode to video and ask Gemini for critical interval annotations (supports zarr and lerobot)."
    )
    parser.add_argument("--demo_path", required=True, help="Path to a zarr.zip archive or LeRobot v2.1 dataset directory/repo_id.")
    parser.add_argument("--demo_type", choices=["zarr", "lerobot"], default="zarr", help="Data format: zarr (UMI) or lerobot.")
    parser.add_argument("--prompt_config", type=str, default="vlm_annotator/prompts/PlaceCupByCoaster.json", help="Path to prompt JSON (task, intervals, optional template and examples).")
    parser.add_argument("--episode_idx", type=str, default="[0, 0]", help="Episode range as Python list, e.g. [179, 237].")
    parser.add_argument("--downsample", type=int, default=3, help="Downsample factor for video.")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview", help="Gemini model to use.")
    parser.add_argument("--few_shot", action="store_true", help="Use few-shot prompting; read examples from prompt_config.")
    parser.add_argument("--backend", choices=["gemini", "vertex"], default="vertex", help="API backend: gemini (API key + file upload) or vertex (Vertex AI + inline video). Default: vertex.")
    parser.add_argument("--vertex_project", type=str, default="", help="Google Cloud project for Vertex AI (or set GOOGLE_CLOUD_PROJECT).")
    parser.add_argument("--vertex_location", type=str, default="", help="Vertex AI location (or set GOOGLE_CLOUD_LOCATION). Default: us-central1.")
    parser.add_argument("--example_source", type=str, default="", help="Optional data source path for few-shot examples. If not set, examples are read from --demo_path.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output JSON and TXT results. Default: results.")
    parser.add_argument("--verbose", action="store_true", help="Print the text prompt (instruction) sent to the model.")
    args = parser.parse_args()

    epi_start, epi_end = parse_episode_range(args.episode_idx)
    print("[INFO] Processing episodes from", epi_start, "to", epi_end)

    prompt_data = load_prompt_config(args.prompt_config)
    instruction = build_instruction(prompt_data)
    if args.verbose:
        print("[VERBOSE] Prompt (instruction) sent to the model:")
        print("-" * 60)
        print(instruction)
        print("-" * 60)
        if args.few_shot and prompt_data.get("examples"):
            N = len(prompt_data["intervals"])
            first_ex = prompt_data["examples"][0]
            example_msg = few_shot_example_message(first_ex["intervals"])
            print("[VERBOSE] Few-shot: text appended to instruction for each example (below: first example):")
            print("-" * 60)
            print(example_msg)
            print("-" * 60)
            follow_up = f"Please label the {N} critical intervals in this video as per the previous examples."
            print("[VERBOSE] Follow-up message sent with target video:")
            print("-" * 60)
            print(follow_up)
            print("-" * 60)

    zarr_group, dataset = load_demo_data(args.demo_type, args.demo_path)
    example_source = args.example_source.strip()
    example_zarr_group, example_dataset = None, None
    if args.few_shot and example_source:
        example_zarr_group, example_dataset = load_demo_data(args.demo_type, example_source)
        print(f"[INFO] Loaded few-shot example source from {example_source}")
    try:
        if zarr_group is not None:
            total_episodes = len(zarr_group["meta/episode_ends"])
        else:
            total_episodes = dataset.num_episodes
        print(f"[INFO] Loaded {args.demo_type} with {total_episodes} episodes")
    except Exception as e:
        print(f"[ERROR] Failed to load demo: {e}")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results_txt = Path(args.output_dir) / f"test_{'few_shot' if args.few_shot else 'zero_shot'}_{args.model}.txt"
    results_json = Path(args.output_dir) / f"test_{'few_shot' if args.few_shot else 'zero_shot'}_{args.model}.json"

    # Auto-resume: load existing JSON results and skip finished episodes.
    results = []
    completed_episodes = set()
    if results_json.exists():
        try:
            with open(results_json, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            if isinstance(existing_results, list):
                results = existing_results
                completed_episodes = {
                    item["episode"]
                    for item in existing_results
                    if isinstance(item, dict) and isinstance(item.get("episode"), int)
                }
                print(f"[INFO] Resume mode: found {len(completed_episodes)} completed episodes in {results_json}")
            else:
                print(f"[WARN] Existing results file is not a list: {results_json}. Start from scratch in memory.")
        except Exception as e:
            print(f"[WARN] Failed to load existing results from {results_json}: {e}. Start from scratch in memory.")

    for idx in range(epi_start, epi_end + 1):
        if idx in completed_episodes:
            print(f"[INFO] Episode {idx} already completed, skipping.")
            continue

        with tempfile.TemporaryDirectory(prefix="ci_mse_episode_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            mp4_path = tmp_dir_path / f"episode_{idx}.mp4"
            h264_path = mp4_path.with_name(f"{mp4_path.stem}_h264.mp4")

            stream = create_video_stream(args.demo_type, args.demo_path, idx, zarr_group=zarr_group, dataset=dataset)
            try:
                episode_fps = stream.fps
                episode_to_mp4(stream, str(mp4_path), downsample=args.downsample)
            finally:
                stream.close()

            transcode_to_h264(str(mp4_path), str(h264_path))

            gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
            response = gemini_annotator(
                api_key=gemini_api_key,
                model=args.model,
                instruction=instruction,
                h264_path=str(h264_path),
                use_few_shot=args.few_shot,
                prompt_data=prompt_data if args.few_shot else None,
                demo_type=args.demo_type,
                demo_path=args.demo_path,
                zarr_group=zarr_group,
                dataset=dataset,
                example_demo_path=example_source,
                example_zarr_group=example_zarr_group,
                example_dataset=example_dataset,
                downsample=args.downsample,
                backend=args.backend,
                vertex_project=args.vertex_project,
                vertex_location=args.vertex_location,
            )

        with open(results_txt, "a", encoding="utf-8") as f:
            f.write(f"--- Episode {idx} ---\n")
            f.write(response + "\n\n")
        print(f"[INFO] Episodes {epi_start} to {idx} have been processed by Gemini.")

        try:
            resp_clean = response.strip()
            if resp_clean.endswith("```"):
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp_clean, re.DOTALL)
                if m:
                    resp_clean = m.group(1)
            result = json.loads(resp_clean)
        except json.JSONDecodeError as e:
            print(f"[ERROR] episode {idx}: Failed to decode JSON:", e)
            print("Raw response:", response)
            continue
        
        # convert start and end timestamp to index
        result_intervals = _intervals_sec_to_index(result.get("intervals", []), episode_fps)
        
        results.append({
            "episode": idx,
            "intervals": result_intervals,
        })
        completed_episodes.add(idx)
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Results saved to {results_json}")

    if zarr_group is not None:
        if hasattr(zarr_group, "store"):
            zarr_group.store.close()
    if example_zarr_group is not None:
        if hasattr(example_zarr_group, "store"):
            example_zarr_group.store.close()
    if dataset is not None:
        dataset = None
    if example_dataset is not None:
        example_dataset = None


if __name__ == "__main__":
    main()

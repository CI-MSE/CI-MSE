---
name: critical-interval-annotator
description: Annotate outcome-critical time intervals in one or more robot manipulation video episodes using local visual inspection. Use for zero-shot critical-interval labeling from video files or dataset episodes; do not use for metadata-only labeling or API-based VLM annotation.
---

# Critical Interval Annotator

Orchestrate local, zero-shot annotation while keeping video content out of the main agent's context.

## Non-negotiable isolation

- Treat each video file or requested dataset episode as one episode.
- Spawn a **fresh subagent for every episode**. Never assign two episodes to one agent and never reuse an episode agent for a later episode.
- The main agent must not open, play, sample, transcode, screenshot, extract frames from, or visually inspect any video. It may inspect filenames, paths, episode indices, and non-visual dataset metadata solely to enumerate work, and it may read the final structured labels and timestamps.
- Only the assigned episode agent may inspect that episode and its derived frames. Pass it the episode source, identifier, optional task instruction, output path, and [episode protocol](references/episode-protocol.md).
- Do not upload video or call Gemini, Vertex AI, OpenAI API, Hugging Face downloads, web services, or any other external annotation API. Dataset sources must already exist locally. Local commands and local image inspection are allowed inside the episode agent.
- Keep the skill self-contained: its scripts must not import `vlm_annotator` or other modules from the host repository. They may use installed local runtime libraries such as `zarr`, NumPy, OpenCV, and `ffmpeg`.
- If subagent delegation is unavailable, stop and explain that the isolation requirement cannot be satisfied; do not inspect the video in the main thread.

## Orchestration

1. Resolve the episodes without examining their visual content. For direct inputs, each video is an episode. For a dataset, use the user's requested episode range; if none is given, use only an unambiguous single episode or ask for the range when processing all episodes would be unexpectedly large.
2. Create a new, run-specific result directory outside the source data. Assign every episode a unique non-negative integer identifier, preserving dataset indices when available. Do not reuse a result directory or modify source videos or datasets.
3. Dispatch as many fresh episode agents concurrently as available slots permit, then continue in waves. Each task must tell the agent to:
   - read `references/episode-protocol.md` in this skill;
   - inspect only its assigned episode using local tools;
   - incorporate the user's optional task instruction as context;
   - write its result atomically to `<result-dir>/episode-<episode-id>.json` using the protocol schema;
   - return a short status without embedding contact sheets or video frames in its final response.
4. Wait for every episode agent. A user-input request from one agent concerns only that episode; do not let another agent inspect it as a shortcut. A retry after an agent failure must use a fresh replacement agent assigned only to that same episode.
5. The main agent may read the final structured JSON labels and timestamps. It must not read intermediate images, contact sheets, extracted clips, or free-form visual inspection notes.
6. Validate and merge results with:

   ```text
   python scripts/validate_and_merge.py --input-dir <result-dir> --expected-episodes <id> [<id> ...] --output <result-dir>/annotations.json --seconds-output <result-dir>/annotations_seconds.json
   ```

   Resolve script paths relative to this skill directory. Validation failure should go back to a fresh replacement episode agent; the main agent must not infer or repair timestamps from video.
   This canonicalizes each per-episode JSON, writes `annotations_seconds.json` with auditable second-based boundaries, and writes pipeline-compatible `annotations.json` with boundaries converted to frame indices.
7. Return both merged JSON paths, episode count, and any failed episode identifiers. For a single episode, the canonicalized per-episode JSON is also a direct seconds-based annotation artifact.

Keep an optional task instruction verbatim when handing off. It helps infer the intended objects and target state but does not make the entire task critical.

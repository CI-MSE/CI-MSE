#!/usr/bin/env python3
"""
Merge all JSON result files under vlm_annotator/results into one JSON.
Output: a single list of all {episode, intervals} entries (no task names).
Usage:
    python merge_results.py [--results_dir results] [--out merged_results.json]
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge VLM annotation JSONs into one file.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory under vlm_annotator containing task subdirs with JSON files (default: results).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="merged_results.json",
        help="Output merged JSON path (default: merged_results.json in results_dir parent).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_root = (script_dir / args.results_dir).resolve()
    if not results_root.is_dir():
        raise SystemExit(f"Results directory not found: {results_root}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = script_dir / out_path

    merged: list = []

    for json_path in sorted(results_root.rglob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        merged.extend(data)

    merged.sort(key=lambda x: x.get("episode", 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(merged)} entries -> {out_path}")


if __name__ == "__main__":
    main()

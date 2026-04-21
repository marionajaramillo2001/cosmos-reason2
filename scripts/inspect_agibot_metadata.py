# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect AgiBotWorld metadata before building the planning manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agibot_planning_common import load_meta_file, read_json


def describe_value(value: Any, depth: int = 0) -> str:
    if depth > 1:
        return type(value).__name__
    if isinstance(value, dict):
        keys = list(value.keys())[:12]
        return "dict keys=" + ", ".join(map(str, keys))
    if isinstance(value, list):
        if not value:
            return "list len=0"
        return f"list len={len(value)} item={describe_value(value[0], depth + 1)}"
    return repr(value)[:160]


def print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agibot-root", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=3)
    args = parser.parse_args()

    root = args.agibot_root.expanduser()
    meta = root / "meta"
    if not meta.exists():
        raise SystemExit(f"Missing meta directory: {meta}")

    print(f"AgiBot root: {root}")

    print_section("Metadata Files")
    for path in sorted(meta.glob("*")):
        if path.suffix in {".json", ".jsonl"}:
            print(f"{path.relative_to(root)}")

    episodes = load_meta_file(root, "episodes.jsonl")
    tasks = load_meta_file(root, "tasks.jsonl")
    info = load_meta_file(root, "info.json")
    annotations = load_meta_file(root, "annotations.json")

    print_section("Core Counts")
    print(f"episodes.jsonl rows: {len(episodes) if isinstance(episodes, list) else 0}")
    print(f"tasks.jsonl rows: {len(tasks) if isinstance(tasks, list) else 0}")
    print(f"info.json: {describe_value(info)}")
    print(f"annotations.json: {describe_value(annotations)}")

    print_section("Episode Samples")
    for row in episodes[: args.max_samples]:
        print(describe_value(row))
        print(row)

    print_section("Task Samples")
    for row in tasks[: args.max_samples]:
        print(row)

    print_section("Info Keys")
    if isinstance(info, dict):
        for key, value in info.items():
            print(f"{key}: {describe_value(value)}")

    print_section("Annotation Keys")
    if isinstance(annotations, dict):
        for key, value in annotations.items():
            print(f"{key}: {describe_value(value)}")

    print_section("Video Cameras")
    video_dirs = sorted(p for p in (root / "videos").glob("**") if p.is_dir())
    camera_dirs = [p for p in video_dirs if "observation.images" in str(p)]
    for path in camera_dirs[:30]:
        count = len(list(path.glob("*.mp4")))
        print(f"{path.relative_to(root)}: {count} mp4 files")

    print_section("Extra JSON Shapes")
    for path in sorted(meta.glob("*.json")):
        if path.name in {"info.json", "annotations.json"}:
            continue
        try:
            value = read_json(path)
        except Exception as exc:  # pragma: no cover - inspection utility
            print(f"{path.name}: failed to parse: {exc}")
            continue
        print(f"{path.name}: {describe_value(value)}")


if __name__ == "__main__":
    main()

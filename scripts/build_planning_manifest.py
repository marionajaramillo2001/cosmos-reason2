# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a long-horizon planning manifest from AgiBotWorld metadata."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from agibot_planning_common import (
    extract_instruction_segments,
    extract_objects,
    extract_task_frames,
    find_episode_video,
    infer_task_text,
    load_meta_file,
    normalize_episode_id,
    plan_length_bucket,
    task_lookup,
    write_jsonl,
)


def load_task_goals(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    if yaml is None:
        raise SystemExit("PyYAML is required to load --task-goals. Install it with: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}


def task_key_from_root(root: Path) -> str:
    for part in reversed(root.parts):
        if re.match(r"task_\d+", part):
            return re.sub(r"_sample$", "", part)
    return re.sub(r"_sample$", "", root.name)


def episode_id_from_row(row: dict[str, Any], fallback_index: int) -> str:
    for key in ("episode_id", "episode", "episode_index", "index", "id"):
        if key in row:
            return normalize_episode_id(row[key])
    return normalize_episode_id(fallback_index)


def build_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.agibot_root.expanduser()
    task_goals = load_task_goals(args.task_goals)
    task_key = task_key_from_root(root)
    task_goal_override = task_goals.get(task_key, "")

    episodes = load_meta_file(root, "episodes.jsonl")
    tasks = load_meta_file(root, "tasks.jsonl")
    info = load_meta_file(root, "info.json")
    annotations = load_meta_file(root, "annotations.json")
    tasks_by_id = task_lookup(tasks if isinstance(tasks, list) else [])

    if not isinstance(episodes, list) or not episodes:
        raise SystemExit(f"No episode rows found at {root / 'meta' / 'episodes.jsonl'}")
    if not isinstance(info, dict):
        info = {}
    if not isinstance(annotations, dict):
        annotations = {}

    rows = []
    for idx, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            continue
        episode_id = episode_id_from_row(episode, idx)
        reference_subgoals = extract_task_frames(info, annotations, episode_id)
        reference_steps = extract_instruction_segments(info, annotations, episode_id)
        high_level_task = task_goal_override or (
            " ".join(reference_subgoals) if reference_subgoals else infer_task_text(episode, tasks_by_id)
        )
        video_path = find_episode_video(root, episode_id, args.camera)
        objects = extract_objects(episode, reference_subgoals, reference_steps)
        include = bool(high_level_task and video_path and (reference_steps or reference_subgoals))
        row = {
            "episode_id": episode_id,
            "task_id": episode.get("task_id") or episode.get("task_index") or "",
            "high_level_task": high_level_task,
            "video_path": video_path,
            "preferred_camera": args.camera,
            "reference_subgoals": reference_subgoals,
            "reference_steps": reference_steps,
            "objects": objects,
            "plan_length_bucket": plan_length_bucket(reference_steps, reference_subgoals),
            "include": include,
            "qc_notes": "" if include else "missing task text, video, or reference annotations",
        }
        if args.included_only and not include:
            continue
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agibot-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--camera", default="observation.images.top_head")
    parser.add_argument("--included-only", action="store_true")
    parser.add_argument("--task-goals", type=Path, default=None,
                        help="YAML file mapping task keys to hand-written high-level goals")
    args = parser.parse_args()

    rows = build_manifest(args)
    write_jsonl(args.out.expanduser(), rows)
    included = sum(1 for row in rows if row["include"])
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Included rows: {included}")
    print("Plan buckets:", {bucket: sum(1 for r in rows if r["plan_length_bucket"] == bucket) for bucket in ("short", "medium", "long")})


if __name__ == "__main__":
    main()

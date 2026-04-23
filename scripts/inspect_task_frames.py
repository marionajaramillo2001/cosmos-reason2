# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print the raw Task Frame annotations per episode for each AgiBot task.

Helps decide what a hand-written `high_level_task` should say, by showing
how the episode was temporally segmented and labeled by annotators.

For each task root, prints:
  - all Task Frame chunks in order (frame range + comment)
  - the unique comments after dedup (what the current code keeps)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def key_frame_entries(raw: dict | list) -> list[dict]:
    entries: list[dict] = []
    if isinstance(raw, dict):
        for group in ("single", "dual"):
            for entry in raw.get(group, []) or []:
                if isinstance(entry, dict):
                    entries.append(entry)
    elif isinstance(raw, list):
        entries = [e for e in raw if isinstance(e, dict)]
    return entries


def task_frame_text(entry: dict) -> str:
    fd = entry.get("frame_detail")
    if isinstance(fd, dict):
        for key in ("comment", "task", "description", "text"):
            if isinstance(fd.get(key), str) and fd[key].strip():
                return fd[key].strip()
    for key in ("comment", "description", "text"):
        if isinstance(entry.get(key), str) and entry[key].strip():
            return entry[key].strip()
    return ""


def inspect_task(root: Path, episode_filter: set[str] | None = None, markdown: bool = False) -> None:
    info_path = root / "data" / "meta" / "info.json"
    if not info_path.exists():
        info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"[skip] no info.json under {root}")
        return
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    key_frames = info.get("key_frame", {}) or {}

    if markdown:
        print(f"\n## {root.name}\n")
    else:
        print(f"\n=== {root.name} ===")

    for ep_key in sorted(key_frames.keys(), key=lambda k: int(k) if str(k).isdigit() else k):
        if episode_filter is not None and str(ep_key) not in episode_filter:
            continue
        entries = key_frame_entries(key_frames[ep_key])
        task_frames = [e for e in entries if "task" in str(e.get("frame_type_name", "")).lower()]
        if not task_frames:
            continue
        if markdown:
            print(f"### episode {ep_key} — {len(task_frames)} Task Frame chunk(s)\n")
            print("| # | frames | comment |")
            print("|---|--------|---------|")
            seen: list[str] = []
            for i, tf in enumerate(task_frames):
                start = tf.get("start", "?")
                end = tf.get("end", "?")
                text = (task_frame_text(tf) or "").replace("|", "\\|")
                print(f"| {i} | {start}–{end} | {text} |")
                if text and text not in seen:
                    seen.append(text)
            print(f"\n**Unique dedup ({len(seen)}):**\n")
            for t in seen:
                print(f"- {t}")
            print()
        else:
            print(f"\n  episode {ep_key} — {len(task_frames)} Task Frame chunk(s)")
            seen = []
            for i, tf in enumerate(task_frames):
                start = tf.get("start", "?")
                end = tf.get("end", "?")
                text = task_frame_text(tf)
                print(f"    [{i}] frames {start}-{end}: {text}")
                if text and text not in seen:
                    seen.append(text)
            print(f"  unique dedup: {len(seen)}")
            for t in seen:
                print(f"    - {t}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, help="Task root dir(s) or globs (e.g. AgiBotWorld2026/task_*_sample).")
    parser.add_argument("--episodes", nargs="+", default=None, help='Restrict to these episode keys (e.g. 0  or  episode_000000). Default: all.')
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown-formatted output (tables + headers).")
    args = parser.parse_args()
    episode_filter: set[str] | None = None
    if args.episodes:
        episode_filter = set()
        for e in args.episodes:
            episode_filter.add(str(e))
            episode_filter.add(str(e).lstrip("0") or "0")
            if e.startswith("episode_"):
                episode_filter.add(str(int(e.split("_")[-1])))
    resolved: list[Path] = []
    for pattern in args.roots:
        matches = sorted(glob.glob(pattern))
        resolved.extend(Path(m) for m in matches) if matches else resolved.append(Path(pattern))
    if args.markdown:
        print("# Task Frame annotations\n")
    for root in resolved:
        inspect_task(root, episode_filter, markdown=args.markdown)


if __name__ == "__main__":
    main()

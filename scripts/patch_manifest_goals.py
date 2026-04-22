# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replace `high_level_task` in manifests with hand-written task goals.

Reads a YAML file mapping task_name -> goal string, and rewrites every
manifest whose filename starts with that task_name, overwriting the
`high_level_task` field on every row. Empty goal strings are skipped.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import yaml

from agibot_planning_common import read_jsonl, write_jsonl


def task_name_from_manifest(path: Path) -> str:
    name = path.name
    for suffix in ("_with_clips.jsonl", ".jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goals", required=True, type=Path, help="YAML mapping task_name -> goal.")
    parser.add_argument("--manifests", nargs="+", required=True, help="Manifest JSONL paths or globs.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    args = parser.parse_args()

    with args.goals.expanduser().open("r", encoding="utf-8") as f:
        goals = yaml.safe_load(f) or {}
    goals = {k: v for k, v in goals.items() if isinstance(v, str) and v.strip()}

    manifest_paths: list[Path] = []
    for pattern in args.manifests:
        matches = sorted(glob.glob(pattern))
        manifest_paths.extend(Path(m) for m in matches) if matches else manifest_paths.append(Path(pattern))

    patched = 0
    skipped = 0
    for path in manifest_paths:
        task = task_name_from_manifest(path)
        goal = goals.get(task)
        if not goal:
            print(f"[skip] {path.name}: no goal for '{task}'")
            skipped += 1
            continue
        rows = list(read_jsonl(path))
        for row in rows:
            row["high_level_task"] = goal
            row["high_level_task_source"] = "hand_written"
        if args.dry_run:
            print(f"[dry] {path.name}: {len(rows)} rows -> {goal!r}")
        else:
            write_jsonl(path, rows)
            print(f"[ok]  {path.name}: {len(rows)} rows -> {goal!r}")
        patched += 1

    print(f"\nPatched {patched} manifests, skipped {skipped}.")


if __name__ == "__main__":
    main()

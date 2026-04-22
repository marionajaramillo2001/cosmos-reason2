# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download AgiBotWorld task archives for inter-task planning experiments.

The working Explorer path uses `uvx hf download` for large gated dataset files.
This script uses Hugging Face Hub only to list matching archive paths, then
delegates each download to the same CLI path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "agibot-world/AgiBotWorld2026"
DEFAULT_PREFIX = "ImitationLearning/CommercialSpaces"
TASK_RE = re.compile(r"/(task_\d+)/([^/]+\.tar\.gz)$")


def task_sort_key(path: str) -> tuple[int, str]:
    match = TASK_RE.search(path)
    if not match:
        return (10**12, path)
    number_match = re.search(r"\d+", match.group(1))
    number = int(number_match.group(0)) if number_match else 10**12
    return (number, path)


def discover_archives(repo_id: str, prefix: str) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Run with `uv run --with huggingface_hub "
            "scripts/download_agibot_task_archives.py ...` or install it in the venv."
        ) from exc

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    archives = [
        path
        for path in files
        if path.startswith(prefix.rstrip("/") + "/")
        and path.endswith(".tar.gz")
        and TASK_RE.search(path)
    ]
    return sorted(archives, key=task_sort_key)


def read_task_ids(paths: list[Path]) -> set[str]:
    task_ids = set()
    for path in paths:
        with path.expanduser().open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    task_ids.add(line)
    return task_ids


def dedupe_by_task(archives: list[str]) -> list[str]:
    selected = []
    seen = set()
    for archive in archives:
        task_id = task_id_from_archive(archive)
        if task_id in seen:
            continue
        seen.add(task_id)
        selected.append(archive)
    return selected


def select_archives(
    archives: list[str],
    task_ids: set[str],
    limit: int,
    download_all: bool,
    one_archive_per_task: bool,
) -> list[str]:
    if task_ids:
        archives = [
            archive
            for archive in archives
            if (match := TASK_RE.search(archive)) and match.group(1) in task_ids
        ]
    if one_archive_per_task:
        archives = dedupe_by_task(archives)
    if not download_all and limit:
        archives = archives[:limit]
    return archives


def task_id_from_archive(path: str) -> str:
    match = TASK_RE.search(path)
    if not match:
        raise ValueError(f"Could not infer task id from archive path: {path}")
    return match.group(1)


def archive_id_from_archive(path: str) -> str:
    match = TASK_RE.search(path)
    if not match:
        raise ValueError(f"Could not infer archive id from archive path: {path}")
    return match.group(2).removesuffix(".tar.gz")


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def download_archive(args: argparse.Namespace, archive: str) -> Path:
    local_path = args.local_dir / archive
    if local_path.exists() and not args.force_download:
        print(f"Archive already exists, skipping download: {local_path}")
        return local_path

    cmd = [
        "uvx",
        "hf",
        "download",
        args.repo_id,
        "--repo-type",
        "dataset",
        "--include",
        archive,
        "--local-dir",
        str(args.local_dir),
    ]
    if args.force_download:
        cmd.append("--force-download")
    run_command(cmd, args.dry_run)
    return local_path


def extract_archive(args: argparse.Namespace, archive_path: Path, task_id: str) -> Path:
    archive_id = archive_id_from_archive(str(archive_path))
    sample_dir = args.local_dir / f"{task_id}_{archive_id}_sample"
    sample_root = sample_dir / "data"
    if sample_root.exists() and not args.force_extract:
        print(f"Sample already exists, skipping extract: {sample_root}")
        return sample_root

    sample_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["tar", "-xzf", str(archive_path), "-C", str(sample_dir)]
    run_command(cmd, args.dry_run)
    return sample_root


def write_jsonl(path: Path, rows: list[dict[str, Any]], dry_run: bool) -> None:
    print(f"Writing inventory: {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_path_from_env(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback)).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=default_path_from_env("AGIBOT_ROOT", "AgiBotWorld2026"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_path_from_env("EXP_ROOT", "cosmos_agibot_planning") / "agibot_task_archives.jsonl",
    )
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--all", action="store_true", help="Download all matching task archives.")
    parser.add_argument(
        "--all-archives-per-task",
        action="store_true",
        help="Allow multiple archives from the same task. By default, one archive is selected per task.",
    )
    parser.add_argument("--task-id", action="append", default=[], help="Specific task id, e.g. task_3401. Can be repeated.")
    parser.add_argument("--task-id-file", action="append", type=Path, default=[])
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.local_dir = args.local_dir.expanduser()
    args.out = args.out.expanduser()
    task_ids = set(args.task_id) | read_task_ids(args.task_id_file)

    archives = discover_archives(args.repo_id, args.prefix)
    selected = select_archives(
        archives,
        task_ids,
        args.limit,
        args.all,
        one_archive_per_task=not args.all_archives_per_task,
    )
    if not selected:
        raise SystemExit("No matching AgiBot task archives found.")

    print(f"Discovered {len(archives)} matching task archives.")
    print(f"Selected {len(selected)} task archives.")

    rows = []
    for index, archive in enumerate(selected, start=1):
        task_id = task_id_from_archive(archive)
        print(f"\n[{index}/{len(selected)}] {task_id}: {archive}")
        archive_path = download_archive(args, archive)
        sample_root = extract_archive(args, archive_path, task_id) if args.extract else None
        rows.append(
            {
                "task_id": task_id,
                "repo_id": args.repo_id,
                "archive_path": archive,
                "local_archive_path": str(archive_path),
                "sample_root": str(sample_root) if sample_root else "",
            }
        )

    write_jsonl(args.out, rows, args.dry_run)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one within-task episode rank across tasks for the four planning methods."""

from __future__ import annotations

import argparse
import glob
import subprocess
from pathlib import Path


def expand_manifest_args(values: list[str]) -> list[Path]:
    paths = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    return sorted({path.expanduser() for path in paths})


def task_name_from_manifest(path: Path) -> str:
    return path.name.removesuffix("_with_clips.jsonl").removesuffix(".jsonl")


def run(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True, help="Manifest paths or glob patterns.")
    parser.add_argument("--intra-rag-dir", required=True, type=Path)
    parser.add_argument("--inter-rag-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--episode-rank", type=int, default=0)
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifests = expand_manifest_args(args.manifests)
    if args.limit_tasks:
        manifests = manifests[: args.limit_tasks]
    if not manifests:
        raise SystemExit("No manifests found.")

    args.out_dir.expanduser().mkdir(parents=True, exist_ok=True)
    intra_out_dir = args.out_dir.expanduser() / "same_task"
    inter_out_dir = args.out_dir.expanduser() / "inter_task"
    intra_out_dir.mkdir(parents=True, exist_ok=True)
    inter_out_dir.mkdir(parents=True, exist_ok=True)

    for manifest in manifests:
        name = task_name_from_manifest(manifest)
        intra_rag = args.intra_rag_dir.expanduser() / f"{name}.jsonl"
        inter_rag = args.inter_rag_dir.expanduser() / f"{name}_rank{args.episode_rank}_inter_task.jsonl"
        if not intra_rag.exists():
            raise SystemExit(f"Missing intra-task RAG index: {intra_rag}")
        if not inter_rag.exists():
            raise SystemExit(f"Missing rank inter-task RAG index: {inter_rag}")

        print(f"\n=== Running rank {args.episode_rank} comparison for {name} ===")
        run(
            [
                "python",
                "scripts/run_planning_prompts_transformers.py",
                "--manifest",
                str(manifest),
                "--rag-index",
                str(intra_rag),
                "--methods",
                "direct",
                "hierarchical",
                "intra_task_rag",
                "--model",
                args.model,
                "--fps",
                str(args.fps),
                "--episode-rank",
                str(args.episode_rank),
                "--out",
                str(intra_out_dir / f"{name}_rank{args.episode_rank}_same_task_predictions.jsonl"),
            ],
            args.dry_run,
        )
        run(
            [
                "python",
                "scripts/run_planning_prompts_transformers.py",
                "--manifest",
                str(manifest),
                "--rag-index",
                str(inter_rag),
                "--methods",
                "inter_task_rag",
                "--model",
                args.model,
                "--fps",
                str(args.fps),
                "--episode-rank",
                str(args.episode_rank),
                "--out",
                str(inter_out_dir / f"{name}_rank{args.episode_rank}_inter_task_predictions.jsonl"),
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge, reparse, and evaluate aligned episode-rank comparison predictions."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from agibot_planning_common import read_jsonl, write_jsonl
from reparse_generated_plans import numbered_steps
from run_planning_prompts import parse_plan


METRICS = ("semantic_step_coverage", "order_lcs", "object_f1", "verb_f1")


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


def prediction_paths(predictions_dir: Path, task_name: str, episode_rank: int) -> tuple[Path, Path]:
    same_task = predictions_dir / "same_task" / f"{task_name}_rank{episode_rank}_same_task_predictions.jsonl"
    inter_task = predictions_dir / "inter_task" / f"{task_name}_rank{episode_rank}_inter_task_predictions.jsonl"
    return same_task, inter_task


def reparse_rows(paths: list[Path], pretty_raw_response: bool) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            row["generated_plan"] = parse_plan(row.get("raw_response", ""))
            if pretty_raw_response:
                row["raw_response"] = numbered_steps(row["generated_plan"])
            rows.append(row)
    return rows


def run_evaluate(
    manifest: Path,
    predictions: Path,
    out_csv: Path,
    summary_json: Path,
    human_template: Path,
) -> None:
    subprocess.run(
        [
            "python",
            "scripts/evaluate_plans.py",
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--out-csv",
            str(out_csv),
            "--summary-json",
            str(summary_json),
            "--human-template-out",
            str(human_template),
        ],
        check=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def combined_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    summary = {}
    for method, method_rows in sorted(by_method.items()):
        summary[method] = {"n": len(method_rows)}
        for metric in METRICS:
            summary[method][metric] = sum(float(row[metric]) for row in method_rows) / len(method_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True, help="Manifest paths or glob patterns.")
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--episode-rank", type=int, default=0)
    parser.add_argument("--limit-tasks", type=int, default=0, help="Evaluate only the first N manifests.")
    parser.add_argument(
        "--no-pretty-raw-response",
        dest="pretty_raw_response",
        action="store_false",
        help="Keep raw_response unchanged instead of replacing it with the parsed numbered plan.",
    )
    parser.set_defaults(pretty_raw_response=True)
    args = parser.parse_args()

    manifests = expand_manifest_args(args.manifests)
    if args.limit_tasks:
        manifests = manifests[: args.limit_tasks]
    if not manifests:
        raise SystemExit("No manifests found.")

    predictions_dir = args.predictions_dir.expanduser()
    out_dir = args.out_dir.expanduser()
    reparsed_dir = out_dir / "reparsed"
    per_task_dir = out_dir / "per_task"
    reparsed_dir.mkdir(parents=True, exist_ok=True)
    per_task_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    for manifest in manifests:
        task_name = task_name_from_manifest(manifest)
        same_task, inter_task = prediction_paths(predictions_dir, task_name, args.episode_rank)
        missing = [path for path in (same_task, inter_task) if not path.exists()]
        if missing:
            raise SystemExit("Missing prediction file(s): " + ", ".join(str(path) for path in missing))

        print(f"\n=== Evaluating rank {args.episode_rank} comparison for {task_name} ===")
        reparsed = reparsed_dir / f"{task_name}_rank{args.episode_rank}_all_methods_reparsed.jsonl"
        rows = reparse_rows([same_task, inter_task], args.pretty_raw_response)
        write_jsonl(reparsed, rows)
        print(f"Wrote merged reparsed predictions to {reparsed}")

        out_csv = per_task_dir / f"{task_name}_rank{args.episode_rank}_auto_eval.csv"
        summary_json = per_task_dir / f"{task_name}_rank{args.episode_rank}_auto_eval_summary.json"
        human_template = per_task_dir / f"{task_name}_rank{args.episode_rank}_human_scores.csv"
        run_evaluate(manifest, reparsed, out_csv, summary_json, human_template)

        for row in read_csv(out_csv):
            row["task_name"] = task_name
            row["episode_rank"] = args.episode_rank
            combined_rows.append(row)

    combined_csv = out_dir / f"rank{args.episode_rank}_combined_auto_eval.csv"
    combined_summary_json = out_dir / f"rank{args.episode_rank}_combined_auto_eval_summary.json"
    write_csv(combined_csv, combined_rows)
    combined_summary_json.write_text(json.dumps(combined_summary(combined_rows), indent=2), encoding="utf-8")
    print(f"\nWrote combined CSV to {combined_csv}")
    print(f"Wrote combined summary JSON to {combined_summary_json}")


if __name__ == "__main__":
    main()

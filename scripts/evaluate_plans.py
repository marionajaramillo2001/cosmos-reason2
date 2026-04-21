# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate generated long-horizon plans against AgiBot reference plans."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from agibot_planning_common import read_jsonl, token_jaccard, token_set


VERBS = {
    "pick",
    "place",
    "move",
    "open",
    "close",
    "grasp",
    "lift",
    "put",
    "push",
    "pull",
    "pour",
    "insert",
    "remove",
    "arrange",
    "stack",
    "wipe",
    "turn",
}


def lcs_length(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, left in enumerate(a, start=1):
        for j, right in enumerate(b, start=1):
            if left == right:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def best_step_coverage(reference: list[str], generated: list[str]) -> float:
    if not reference:
        return 0.0
    scores = [max((token_jaccard(ref, gen) for gen in generated), default=0.0) for ref in reference]
    return sum(scores) / len(scores)


def order_score(reference: list[str], generated: list[str], threshold: float) -> float:
    if not reference:
        return 0.0
    matched = []
    for gen in generated:
        best_i = -1
        best_score = 0.0
        for i, ref in enumerate(reference):
            score = token_jaccard(ref, gen)
            if score > best_score:
                best_score = score
                best_i = i
        if best_score >= threshold:
            matched.append(str(best_i))
    return lcs_length([str(i) for i in range(len(reference))], matched) / len(reference)


def f1(reference: set[str], generated: set[str]) -> float:
    if not reference and not generated:
        return 1.0
    if not reference or not generated:
        return 0.0
    tp = len(reference & generated)
    precision = tp / len(generated) if generated else 0.0
    recall = tp / len(reference) if reference else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def verbs_in_steps(steps: list[str]) -> set[str]:
    tokens = set()
    for step in steps:
        tokens |= token_set(step)
    return tokens & VERBS


def generated_objects(objects: list[str], generated_text: str) -> set[str]:
    lower = generated_text.lower()
    return {obj for obj in objects if obj.lower() in lower}


def read_human_scores(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return {(row["episode_id"], row["method"]): row for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--human-scores", type=Path, default=None)
    parser.add_argument("--human-template-out", type=Path, default=None)
    parser.add_argument("--match-threshold", type=float, default=0.35)
    args = parser.parse_args()

    manifest = {row["episode_id"]: row for row in read_jsonl(args.manifest.expanduser())}
    predictions = read_jsonl(args.predictions.expanduser())
    human_scores = read_human_scores(args.human_scores)
    global_objects = sorted({obj.lower() for row in manifest.values() for obj in row.get("objects", [])}, key=len, reverse=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pred in predictions:
        ref = manifest[pred["episode_id"]]
        reference = ref.get("reference_steps") or ref.get("reference_subgoals", [])
        generated = pred.get("generated_plan", [])
        generated_text = "\n".join(generated)
        objects = ref.get("objects", [])
        object_reference = {obj.lower() for obj in objects}
        object_generated = generated_objects(objects, generated_text)
        all_generated_objects = generated_objects(global_objects, generated_text)
        hallucinated_objects = all_generated_objects - object_reference
        auto = {
            "episode_id": pred["episode_id"],
            "method": pred["method"],
            "plan_length_bucket": ref.get("plan_length_bucket", ""),
            "reference_len": len(reference),
            "generated_len": len(generated),
            "semantic_step_coverage": round(best_step_coverage(reference, generated), 4),
            "order_lcs": round(order_score(reference, generated, args.match_threshold), 4),
            "object_f1": round(f1(object_reference, object_generated), 4),
            "verb_f1": round(f1(verbs_in_steps(reference), verbs_in_steps(generated)), 4),
            "plan_length_error": len(generated) - len(reference),
            "hallucinated_object_count": len(hallucinated_objects),
            "hallucinated_objects": "; ".join(sorted(hallucinated_objects)),
        }
        score = human_scores.get((pred["episode_id"], pred["method"]))
        if score:
            auto.update({f"human_{key}": score.get(key, "") for key in score})
        rows.append(auto)

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    if args.human_template_out:
        args.human_template_out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "episode_id",
            "method",
            "goal_completion",
            "subgoal_coverage",
            "step_ordering",
            "object_grounding",
            "no_hallucination",
            "total",
            "notes",
        ]
        with args.human_template_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for pred in predictions:
                writer.writerow({"episode_id": pred["episode_id"], "method": pred["method"]})

    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    summary = {}
    for method, method_rows in by_method.items():
        summary[method] = {
            "n": len(method_rows),
            "semantic_step_coverage": sum(float(r["semantic_step_coverage"]) for r in method_rows) / len(method_rows),
            "order_lcs": sum(float(r["order_lcs"]) for r in method_rows) / len(method_rows),
            "object_f1": sum(float(r["object_f1"]) for r in method_rows) / len(method_rows),
            "verb_f1": sum(float(r["verb_f1"]) for r in method_rows) / len(method_rows),
        }

    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote evaluation CSV to {args.out_csv}")
    print(f"Wrote summary JSON to {args.summary_json}")


if __name__ == "__main__":
    main()

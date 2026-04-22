# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedding-based evaluation of generated plans.

Replaces the bag-of-words token_jaccard used in evaluate_plans.py with
sentence-embedding cosine similarity. This removes the vocabulary-alignment
artifact where plans that happen to reuse the reference's exact wording
score artificially higher than plans that express the same idea in
different words.

Outputs CSV + summary JSON mirroring evaluate_plans.py so downstream
scripts can consume either.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agibot_planning_common import read_jsonl


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def lcs_length(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, left in enumerate(a, start=1):
        for j, right in enumerate(b, start=1):
            if left == right:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def encode(model: Any, texts: list[str]) -> Any:
    return model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def coverage_from_sim(sim: Any) -> float:
    """Mean over rows of per-row max similarity. Rows are reference steps."""
    if sim.numel() == 0:
        return 0.0
    return float(sim.max(dim=1).values.mean().item())


def order_from_sim(sim: Any, threshold: float) -> float:
    """LCS of matched reference indices / len(reference)."""
    import torch

    n_ref, n_gen = sim.shape
    if n_ref == 0 or n_gen == 0:
        return 0.0
    matched: list[str] = []
    for j in range(n_gen):
        col = sim[:, j]
        best_i = int(torch.argmax(col).item())
        if float(col[best_i].item()) >= threshold:
            matched.append(str(best_i))
    return lcs_length([str(i) for i in range(n_ref)], matched) / n_ref


def object_f1(reference: set[str], generated: set[str]) -> float:
    if not reference and not generated:
        return 1.0
    if not reference or not generated:
        return 0.0
    tp = len(reference & generated)
    if tp == 0:
        return 0.0
    precision = tp / len(generated)
    recall = tp / len(reference)
    return 2 * precision * recall / (precision + recall)


def objects_in_text(objects: list[str], text: str) -> set[str]:
    lower = text.lower()
    return {obj for obj in objects if obj.lower() in lower}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--order-threshold",
        type=float,
        default=0.55,
        help="Cosine similarity required to count a generated step as matching a reference step for ordering.",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    manifest = {row["episode_id"]: row for row in read_jsonl(args.manifest.expanduser())}
    predictions = read_jsonl(args.predictions.expanduser())
    global_objects = sorted(
        {obj.lower() for row in manifest.values() for obj in row.get("objects", [])},
        key=len,
        reverse=True,
    )

    print(f"Loading embedding model {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    rows: list[dict[str, Any]] = []
    for pred in predictions:
        ref = manifest[pred["episode_id"]]
        reference = ref.get("reference_steps") or ref.get("reference_subgoals", [])
        generated = pred.get("generated_plan", [])
        generated_text = "\n".join(generated)
        objects = ref.get("objects", [])
        object_reference = {obj.lower() for obj in objects}
        object_generated = objects_in_text(objects, generated_text)
        all_generated_objects = objects_in_text(global_objects, generated_text)
        hallucinated_objects = all_generated_objects - object_reference

        if reference and generated:
            ref_emb = encode(model, reference)
            gen_emb = encode(model, generated)
            sim = ref_emb @ gen_emb.T  # cosine since embeddings are normalized
            emb_coverage = coverage_from_sim(sim)
            emb_order = order_from_sim(sim, args.order_threshold)
        else:
            emb_coverage = 0.0
            emb_order = 0.0

        rows.append(
            {
                "episode_id": pred["episode_id"],
                "method": pred["method"],
                "plan_length_bucket": ref.get("plan_length_bucket", ""),
                "reference_len": len(reference),
                "generated_len": len(generated),
                "embedding_step_coverage": round(emb_coverage, 4),
                "embedding_order_lcs": round(emb_order, 4),
                "object_f1": round(object_f1(object_reference, object_generated), 4),
                "plan_length_error": len(generated) - len(reference),
                "hallucinated_object_count": len(hallucinated_objects),
                "hallucinated_objects": "; ".join(sorted(hallucinated_objects)),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    metrics = ("embedding_step_coverage", "embedding_order_lcs", "object_f1")
    summary: dict[str, dict[str, float | int]] = {}
    for method, method_rows in sorted(by_method.items()):
        entry: dict[str, float | int] = {"n": len(method_rows)}
        for m in metrics:
            entry[m] = sum(float(r[m]) for r in method_rows) / len(method_rows)
        summary[method] = entry

    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote embedding eval CSV to {args.out_csv}")
    print(f"Wrote embedding summary JSON to {args.summary_json}")


if __name__ == "__main__":
    main()

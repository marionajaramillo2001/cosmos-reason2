# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build per-task RAG indexes whose examples come only from other tasks."""

from __future__ import annotations

import argparse
import glob
import math
from collections import Counter
from pathlib import Path
from typing import Any

from agibot_planning_common import read_jsonl, token_set, write_jsonl


def text_counts(text: str) -> Counter[str]:
    return Counter(token_set(text))


def cosine(a: Counter[str], b: Counter[str]) -> float:
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    a_norm = math.sqrt(sum(v * v for v in a.values()))
    b_norm = math.sqrt(sum(v * v for v in b.values()))
    if not a_norm or not b_norm:
        return 0.0
    return dot / (a_norm * b_norm)


def vector_cosine(a: Any, b: Any) -> float:
    dot = float(a @ b)
    a_norm = float(math.sqrt(a @ a))
    b_norm = float(math.sqrt(b @ b))
    if not a_norm or not b_norm:
        return 0.0
    return dot / (a_norm * b_norm)


def load_sentence_transformer(model_name: str, backend: str) -> Any | None:
    if backend == "lexical":
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        if backend == "sentence-transformers":
            raise SystemExit("sentence-transformers is not installed. Install it or use --backend lexical.")
        return None
    return SentenceTransformer(model_name)


def object_overlap(a: list[str], b: list[str]) -> float:
    a_set = {x.lower() for x in a}
    b_set = {x.lower() for x in b}
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def length_similarity(a: list[str], b: list[str]) -> float:
    a_len = max(1, len(a))
    b_len = max(1, len(b))
    return 1.0 - min(abs(a_len - b_len) / max(a_len, b_len), 1.0)


def expand_manifest_args(values: list[str]) -> list[Path]:
    paths = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    return sorted({path.expanduser() for path in paths})


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    task_key = path.name.removesuffix("_with_clips.jsonl").removesuffix(".jsonl")
    rows = []
    manifest_rows = [row for row in read_jsonl(path) if row.get("include", True)]
    manifest_rows.sort(key=lambda row: row.get("episode_id", ""))
    for index, row in enumerate(manifest_rows):
        if not row.get("include", True):
            continue
        item = dict(row)
        item["_task_key"] = task_key
        item["_manifest_path"] = str(path)
        item["_episode_rank"] = index
        item["_row_key"] = f"{task_key}:{item.get('episode_id', index)}"
        rows.append(item)
    return rows


def build_vectors(rows: list[dict[str, Any]], backend: str, embedding_model: str) -> tuple[Any | None, dict[str, Any], dict[str, Counter[str]]]:
    model = load_sentence_transformer(embedding_model, backend)
    if model is None:
        print("Using lexical token cosine for task-text similarity.")
        counts = {row["_row_key"]: text_counts(row.get("high_level_task", "")) for row in rows}
        return None, {}, counts
    print(f"Using sentence-transformer embeddings: {embedding_model}")
    embeddings = model.encode([row.get("high_level_task", "") for row in rows], normalize_embeddings=True)
    vectors = {row["_row_key"]: embeddings[i] for i, row in enumerate(rows)}
    return model, vectors, {}


def score_candidate(
    query: dict[str, Any],
    candidate: dict[str, Any],
    model: Any | None,
    vectors: dict[str, Any],
    counts: dict[str, Counter[str]],
) -> tuple[float, float, float, float]:
    if model is None:
        text_sim = cosine(counts[query["_row_key"]], counts[candidate["_row_key"]])
    else:
        text_sim = vector_cosine(vectors[query["_row_key"]], vectors[candidate["_row_key"]])
    obj_sim = object_overlap(query.get("objects", []), candidate.get("objects", []))
    len_sim = length_similarity(query.get("reference_steps", []), candidate.get("reference_steps", []))
    score = 0.60 * text_sim + 0.25 * obj_sim + 0.15 * len_sim
    return score, text_sim, obj_sim, len_sim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True, help="Manifest paths or glob patterns.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--episode-rank", type=int, default=None, help="Only build query rows for this within-task episode rank.")
    parser.add_argument("--pool-same-rank-only", action="store_true", help="Retrieve only examples with the same within-task episode rank.")
    parser.add_argument("--backend", choices=("auto", "sentence-transformers", "lexical"), default="auto")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    manifest_paths = expand_manifest_args(args.manifests)
    if len(manifest_paths) < 2:
        raise SystemExit("Inter-task RAG requires at least two manifests.")

    rows_by_task = {}
    all_rows = []
    for path in manifest_paths:
        rows = load_manifest_rows(path)
        if not rows:
            print(f"Skipping empty manifest: {path}")
            continue
        task_key = rows[0]["_task_key"]
        rows_by_task[task_key] = rows
        all_rows.extend(rows)

    if len(rows_by_task) < 2:
        raise SystemExit("Inter-task RAG requires at least two non-empty task manifests.")

    model, vectors, counts = build_vectors(all_rows, args.backend, args.embedding_model)
    args.out_dir.expanduser().mkdir(parents=True, exist_ok=True)

    suffix = "_inter_task"
    if args.episode_rank is not None:
        suffix = f"_rank{args.episode_rank}_inter_task"

    for task_key, queries in sorted(rows_by_task.items()):
        if args.episode_rank is not None:
            queries = [row for row in queries if row["_episode_rank"] == args.episode_rank]
        pool = [row for row in all_rows if row["_task_key"] != task_key]
        if args.pool_same_rank_only and args.episode_rank is not None:
            pool = [row for row in pool if row["_episode_rank"] == args.episode_rank]
        out_rows = []
        for query in queries:
            scored = []
            for candidate in pool:
                score, text_sim, obj_sim, len_sim = score_candidate(query, candidate, model, vectors, counts)
                scored.append((score, text_sim, obj_sim, len_sim, candidate))
            scored.sort(key=lambda item: item[0], reverse=True)

            retrieved = []
            for score, text_sim, obj_sim, len_sim, candidate in scored[: args.top_k]:
                retrieved.append(
                    {
                        "episode_id": candidate["episode_id"],
                        "source_task": candidate["_task_key"],
                        "goal": candidate.get("high_level_task", ""),
                        "reference_plan": candidate.get("reference_steps") or candidate.get("reference_subgoals", []),
                        "objects": candidate.get("objects", []),
                        "similarity": round(score, 4),
                        "text_similarity": round(text_sim, 4),
                        "object_overlap": round(obj_sim, 4),
                        "length_similarity": round(len_sim, 4),
                    }
                )
            out_rows.append({"query_episode_id": query["episode_id"], "retrieved_examples": retrieved})

        out_path = args.out_dir.expanduser() / f"{task_key}{suffix}.jsonl"
        write_jsonl(out_path, out_rows)
        print(f"Wrote inter-task RAG index for {task_key}: {out_path}")


if __name__ == "__main__":
    main()

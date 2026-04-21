# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build retrieval examples for RAG planning prompts."""

from __future__ import annotations

import argparse
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
    if not a_set and not b_set:
        return 0.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def length_similarity(a: list[str], b: list[str]) -> float:
    a_len = max(1, len(a))
    b_len = max(1, len(b))
    return 1.0 - min(abs(a_len - b_len) / max(a_len, b_len), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--pool", type=Path, default=None, help="Optional larger retrieval-pool manifest.")
    parser.add_argument("--backend", choices=("auto", "sentence-transformers", "lexical"), default="auto")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    queries = [row for row in read_jsonl(args.manifest.expanduser()) if row.get("include", True)]
    pool = [row for row in read_jsonl((args.pool or args.manifest).expanduser()) if row.get("include", True)]
    model = load_sentence_transformer(args.embedding_model, args.backend)
    if model is None:
        print("Using lexical token cosine for task-text similarity.")
        pool_counts = {row["episode_id"]: text_counts(row.get("high_level_task", "")) for row in pool}
        query_vectors = {}
        pool_vectors = {}
    else:
        print(f"Using sentence-transformer embeddings: {args.embedding_model}")
        all_rows = queries + [row for row in pool if row["episode_id"] not in {q["episode_id"] for q in queries}]
        embeddings = model.encode([row.get("high_level_task", "") for row in all_rows], normalize_embeddings=True)
        vectors = {row["episode_id"]: embeddings[i] for i, row in enumerate(all_rows)}
        pool_vectors = vectors
        query_vectors = vectors
        pool_counts = {}
    out_rows = []

    for query in queries:
        query_counts = text_counts(query.get("high_level_task", "")) if model is None else Counter()
        scored = []
        for candidate in pool:
            if candidate["episode_id"] == query["episode_id"]:
                continue
            if model is None:
                text_sim = cosine(query_counts, pool_counts[candidate["episode_id"]])
            else:
                text_sim = vector_cosine(query_vectors[query["episode_id"]], pool_vectors[candidate["episode_id"]])
            obj_sim = object_overlap(query.get("objects", []), candidate.get("objects", []))
            len_sim = length_similarity(query.get("reference_steps", []), candidate.get("reference_steps", []))
            score = 0.60 * text_sim + 0.25 * obj_sim + 0.15 * len_sim
            scored.append((score, text_sim, obj_sim, len_sim, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        retrieved = []
        for score, text_sim, obj_sim, len_sim, candidate in scored[: args.top_k]:
            retrieved.append(
                {
                    "episode_id": candidate["episode_id"],
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

    write_jsonl(args.out.expanduser(), out_rows)
    print(f"Wrote RAG index for {len(out_rows)} queries to {args.out}")


if __name__ == "__main__":
    main()

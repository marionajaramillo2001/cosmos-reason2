# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Direct, Hierarchical, and RAG planning prompts against a vLLM server."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from agibot_planning_common import read_jsonl, write_jsonl


METHOD_TO_PROMPT = {
    "direct": "prompts/planning_direct.yaml",
    "hierarchical": "prompts/planning_hierarchical.yaml",
    "intra_task_rag": "prompts/planning_rag.yaml",
    # Backward-compatible alias for older prediction runs.
    "rag": "prompts/planning_rag.yaml",
    "inter_task_rag": "prompts/planning_rag.yaml",
}
RAG_METHODS = {"rag", "intra_task_rag", "inter_task_rag"}
DEFAULT_MODEL = "nvidia/Cosmos-Reason2-2B"


def load_prompt_template(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if re.match(r"\s*user_prompt:\s*\|", line))
    except StopIteration:
        raise ValueError(f"Could not parse user_prompt block from {path}")
    body = []
    for line in lines[start + 1 :]:
        if line.startswith("  "):
            body.append(line[2:])
        elif not line.strip():
            body.append("")
        else:
            break
    return "\n".join(body).strip()


def load_rag_index(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    index = {}
    for row in read_jsonl(path.expanduser()):
        index[row["query_episode_id"]] = row.get("retrieved_examples", [])
    return index


def format_reference_examples(examples: list[dict[str, Any]]) -> str:
    blocks = []
    for idx, example in enumerate(examples, start=1):
        plan_lines = "\n".join(f"{i}. {step}" for i, step in enumerate(example.get("reference_plan", []), start=1))
        blocks.append(f"Example {idx}\nGoal: {example.get('goal', '')}\nReference plan:\n{plan_lines}")
    return "\n\n".join(blocks) if blocks else "No retrieved examples available."


def parse_plan(text: str) -> list[str]:
    json_text = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", json_text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        json_text = fence_match.group("body").strip()
    try:
        value = json.loads(json_text)
        if isinstance(value, dict):
            if isinstance(value.get("steps"), list):
                return [str(x) for x in value["steps"]]
            if isinstance(value.get("plan"), list):
                return [str(x) for x in value["plan"]]
            if isinstance(value.get("subgoals"), list):
                steps = []
                for subgoal in value["subgoals"]:
                    if isinstance(subgoal, dict):
                        steps.extend(str(x) for x in subgoal.get("steps", []))
                    else:
                        steps.append(str(subgoal))
                if steps:
                    return steps
        if isinstance(value, list):
            return [str(x) for x in value]
    except json.JSONDecodeError:
        pass
    steps = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[\.)]\s+", "", line)
        if line and len(line) > 3 and not line.lower().startswith(("<think>", "</think>")):
            steps.append(line)
    return steps


def get_model(base_url: str, requested: str | None) -> str:
    if requested:
        return requested
    with urllib.request.urlopen(f"{base_url}/models", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["data"][0]["id"]


def chat_completion(base_url: str, model: str, video_path: str, prompt: str, max_tokens: int, fps: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful robot planning assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"file://{Path(video_path).resolve()}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "mm_processor_kwargs": {"fps": fps, "do_sample_frames": True},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rag-index", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_TO_PROMPT), default=["direct", "hierarchical", "intra_task_rag"])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episode-rank", type=int, default=None, help="Run only this within-manifest episode rank after sorting by episode_id.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    model = "DRY_RUN" if args.dry_run else get_model(base_url, args.model)
    rows = [row for row in read_jsonl(args.manifest.expanduser()) if row.get("include", True)]
    rows.sort(key=lambda row: row.get("episode_id", ""))
    if args.episode_rank is not None:
        if args.episode_rank < 0 or args.episode_rank >= len(rows):
            raise SystemExit(f"episode-rank {args.episode_rank} is out of range for {args.manifest}")
        rows = [rows[args.episode_rank]]
    rag_index = load_rag_index(args.rag_index)
    templates = {method: load_prompt_template(Path(METHOD_TO_PROMPT[method])) for method in args.methods}
    outputs = []

    for row in rows:
        video_path = row.get("initial_video") or row.get("clip_path") or row.get("video_path")
        if not video_path:
            continue
        for method in args.methods:
            retrieved = rag_index.get(row["episode_id"], []) if method in RAG_METHODS else []
            prompt = templates[method].format(
                goal=row.get("high_level_task", ""),
                objects=", ".join(row.get("objects", [])) or "unknown",
                reference_examples=format_reference_examples(retrieved),
            )
            if args.dry_run:
                raw = f"DRY RUN for {row['episode_id']} {method}"
            else:
                raw = chat_completion(base_url, model, video_path, prompt, args.max_tokens, args.fps)
            outputs.append(
                {
                    "episode_id": row["episode_id"],
                    "method": method,
                    "model": model,
                    "video_path": video_path,
                    "prompt": prompt,
                    "raw_response": raw,
                    "generated_plan": parse_plan(raw),
                    "retrieved_examples": retrieved,
                }
            )
            print(f"Completed {row['episode_id']} / {method}")

    write_jsonl(args.out.expanduser(), outputs)
    print(f"Wrote {len(outputs)} predictions to {args.out}")


if __name__ == "__main__":
    main()

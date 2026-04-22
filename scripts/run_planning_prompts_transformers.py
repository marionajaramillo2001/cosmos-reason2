# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run planning prompts with Hugging Face Transformers instead of vLLM.

This is slower than the vLLM server path, but it avoids vLLM-specific CUDA
attention kernels and is useful on clusters where those kernels are incompatible.
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path
from typing import Any

from agibot_planning_common import read_jsonl, write_jsonl
from run_planning_prompts import (
    DEFAULT_MODEL,
    HIERARCHICAL_TWO_CALL_EXPAND_PROMPT,
    METHOD_TO_PROMPT,
    RAG_METHODS,
    TWO_CALL_METHODS,
    format_reference_examples,
    load_prompt_template,
    load_rag_index,
    parse_plan,
)


PIXELS_PER_TOKEN = 32**2


def dtype_from_name(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    import transformers

    transformers.set_seed(args.seed)
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(args.model)
    processor.image_processor.size = {
        "shortest_edge": args.min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": args.max_vision_tokens * PIXELS_PER_TOKEN,
    }
    processor.video_processor.size = {
        "shortest_edge": args.min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": args.max_vision_tokens * PIXELS_PER_TOKEN,
    }
    return model, processor


def model_input_device(model: Any) -> torch.device:
    import torch

    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def generate_one(
    model: Any,
    processor: Any,
    video_path: str,
    prompt: str,
    fps: float,
    max_new_tokens: int,
) -> str:
    import torch

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful robot planning assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(Path(video_path).resolve())},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    )
    inputs = inputs.to(model_input_device(model))
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    del inputs, generated_ids, generated_ids_trimmed
    return output_text.strip()


def generate_hierarchical_two_call(
    model: Any,
    processor: Any,
    video_path: str,
    goal: str,
    objects_str: str,
    subgoals_template: str,
    expand_template: str,
    fps: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run the two-call hierarchical planner.

    Call 1: ask for a short list of high-level subgoals.
    Call 2 (per subgoal): ask the model to expand that subgoal into low-level actions,
    conditioned on the video and the full subgoal list.

    Returns a dict with the intermediate subgoals prompt/response, the per-subgoal
    expansion prompts/responses, and a flattened final plan compatible with the
    existing evaluation scripts.
    """

    subgoals_prompt = subgoals_template.format(goal=goal, objects=objects_str)
    subgoals_raw = generate_one(
        model=model,
        processor=processor,
        video_path=video_path,
        prompt=subgoals_prompt,
        fps=fps,
        max_new_tokens=max_new_tokens,
    )
    subgoals = parse_plan(subgoals_raw)

    subgoal_list_str = "\n".join(f"{i}. {sg}" for i, sg in enumerate(subgoals, start=1))
    expansions: list[dict[str, Any]] = []
    flat_plan: list[str] = []

    for i, subgoal in enumerate(subgoals, start=1):
        expand_prompt = expand_template.format(
            goal=goal,
            objects=objects_str,
            subgoal_list=subgoal_list_str,
            subgoal_index=i,
            subgoal_total=len(subgoals),
            subgoal=subgoal,
        )
        expand_raw = generate_one(
            model=model,
            processor=processor,
            video_path=video_path,
            prompt=expand_prompt,
            fps=fps,
            max_new_tokens=max_new_tokens,
        )
        expand_steps = parse_plan(expand_raw)
        expansions.append(
            {
                "subgoal_index": i,
                "subgoal": subgoal,
                "prompt": expand_prompt,
                "raw_response": expand_raw,
                "steps": expand_steps,
            }
        )
        flat_plan.extend(expand_steps)

    combined_raw = json.dumps(
        {
            "subgoals": subgoals,
            "expansions": [
                {"subgoal": exp["subgoal"], "steps": exp["steps"]} for exp in expansions
            ],
        },
        indent=2,
    )

    return {
        "subgoals_prompt": subgoals_prompt,
        "subgoals_raw_response": subgoals_raw,
        "subgoals": subgoals,
        "expansions": expansions,
        "raw_response": combined_raw,
        "generated_plan": flat_plan,
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rag-index", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_TO_PROMPT), default=["direct", "hierarchical", "intra_task_rag"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--episode-rank", type=int, default=None, help="Run only this within-manifest episode rank after sorting by episode_id.")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-vision-tokens", type=int, default=128)
    parser.add_argument("--max-vision-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.manifest.expanduser()) if row.get("include", True)]
    rows.sort(key=lambda row: row.get("episode_id", ""))
    if args.episode_rank is not None:
        if args.episode_rank < 0 or args.episode_rank >= len(rows):
            raise SystemExit(f"episode-rank {args.episode_rank} is out of range for {args.manifest}")
        rows = [rows[args.episode_rank]]
    if args.limit:
        rows = rows[: args.limit]
    rag_index = load_rag_index(args.rag_index)
    templates = {method: load_prompt_template(Path(METHOD_TO_PROMPT[method])) for method in args.methods}
    expand_template = (
        load_prompt_template(Path(HIERARCHICAL_TWO_CALL_EXPAND_PROMPT))
        if any(method in TWO_CALL_METHODS for method in args.methods)
        else None
    )

    print(f"Loading {args.model} with Transformers ({args.attn_implementation=}, {args.dtype=})")
    model, processor = load_model_and_processor(args)
    outputs = []

    for row in rows:
        video_path = row.get("initial_video") or row.get("clip_path") or row.get("video_path")
        if not video_path:
            continue
        for method in args.methods:
            retrieved = rag_index.get(row["episode_id"], []) if method in RAG_METHODS else []
            goal = row.get("high_level_task", "")
            objects_str = ", ".join(row.get("objects", [])) or "unknown"

            if method in TWO_CALL_METHODS:
                assert expand_template is not None
                result = generate_hierarchical_two_call(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    goal=goal,
                    objects_str=objects_str,
                    subgoals_template=templates[method],
                    expand_template=expand_template,
                    fps=args.fps,
                    max_new_tokens=args.max_new_tokens,
                )
                outputs.append(
                    {
                        "episode_id": row["episode_id"],
                        "method": method,
                        "model": args.model,
                        "runtime": "transformers",
                        "video_path": video_path,
                        "prompt": result["subgoals_prompt"],
                        "raw_response": result["raw_response"],
                        "generated_plan": result["generated_plan"],
                        "subgoals": result["subgoals"],
                        "subgoals_raw_response": result["subgoals_raw_response"],
                        "expansions": result["expansions"],
                        "retrieved_examples": retrieved,
                    }
                )
            else:
                prompt = templates[method].format(
                    goal=goal,
                    objects=objects_str,
                    reference_examples=format_reference_examples(retrieved),
                )
                raw = generate_one(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    prompt=prompt,
                    fps=args.fps,
                    max_new_tokens=args.max_new_tokens,
                )
                outputs.append(
                    {
                        "episode_id": row["episode_id"],
                        "method": method,
                        "model": args.model,
                        "runtime": "transformers",
                        "video_path": video_path,
                        "prompt": prompt,
                        "raw_response": raw,
                        "generated_plan": parse_plan(raw),
                        "retrieved_examples": retrieved,
                    }
                )
            write_jsonl(args.out.expanduser(), outputs)
            print(f"Completed {row['episode_id']} / {method}")
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Wrote {len(outputs)} predictions to {args.out}")


if __name__ == "__main__":
    main()

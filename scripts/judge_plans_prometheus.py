# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-as-judge evaluation of generated plans using Prometheus-2-7B.

For each (episode, method) prediction, grades the plan against the gold
reference steps on four rubrics (goal_completion, step_ordering,
subgoal_coverage, no_hallucination) on a 1-5 scale. Each rubric is scored
multiple times at a low temperature and averaged to reduce judge noise.

Designed as a drop-in companion to evaluate_plans.py. Runs on a single
24GB+ GPU via Hugging Face Transformers. Prometheus is a Mistral-7B
variant fine-tuned specifically for reference-guided rubric scoring
(https://arxiv.org/abs/2405.01535), so it is a different model family
from Cosmos Reason (Qwen2.5-VL) and avoids self-preference bias.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from agibot_planning_common import read_jsonl


DEFAULT_JUDGE_MODEL = "prometheus-eval/prometheus-7b-v2.0"


PROMETHEUS_TEMPLATE = """###Task Description:
An instruction, a response to evaluate, a reference answer that gets a score of 5, and a score rubric are given.
1. Write detailed feedback assessing the response strictly based on the rubric.
2. After writing feedback, write a score that is an integer between 1 and 5.
3. Output format: "Feedback: (write feedback) [RESULT] (an integer between 1 and 5)"
4. Do not generate any other opening, closing, or explanations.

###The instruction to evaluate:
Generate an ordered action plan for the following robot task. Use only objects and actions consistent with the scene.

High-level task: {goal}
Annotated objects in scene: {objects}

###Response to evaluate:
{candidate_plan}

###Reference Answer (Score 5):
{reference_plan}

###Score Rubrics:
[{rubric_title}: {rubric_desc}]
Score 1: {s1}
Score 2: {s2}
Score 3: {s3}
Score 4: {s4}
Score 5: {s5}

###Feedback:"""


RUBRICS: dict[str, dict[str, str]] = {
    "goal_completion": {
        "title": "Goal completion",
        "desc": "Does the plan, if executed, achieve the high-level task goal?",
        "s1": "Plan clearly cannot achieve the goal or addresses a different task.",
        "s2": "Plan addresses the goal but misses critical steps needed to complete it.",
        "s3": "Plan partially achieves the goal; some important steps are missing or wrong.",
        "s4": "Plan achieves the goal with only minor issues or small omissions.",
        "s5": "Plan fully and correctly achieves the goal described by the reference.",
    },
    "step_ordering": {
        "title": "Step ordering",
        "desc": "Are the actions in a valid, executable order consistent with the reference?",
        "s1": "Order is nonsensical, contradictory, or physically impossible.",
        "s2": "Several steps are out of order in ways that would prevent execution.",
        "s3": "Roughly half of the ordering is correct; some misplaced steps.",
        "s4": "Ordering is mostly correct with only a few minor out-of-order steps.",
        "s5": "All steps are in a valid, executable order consistent with the reference.",
    },
    "subgoal_coverage": {
        "title": "Subgoal coverage",
        "desc": "Does the plan cover the key subgoals present in the reference plan?",
        "s1": "Plan misses nearly all of the reference subgoals.",
        "s2": "Plan covers only a small fraction of the reference subgoals.",
        "s3": "Plan covers roughly half of the reference subgoals.",
        "s4": "Plan covers most reference subgoals with a few omissions.",
        "s5": "Plan covers all reference subgoals.",
    },
    "no_hallucination": {
        "title": "No hallucination",
        "desc": "Does the plan avoid objects or actions not grounded in the scene or reference?",
        "s1": "Plan relies heavily on hallucinated objects or actions not present in the scene.",
        "s2": "Multiple hallucinated objects or actions appear in the plan.",
        "s3": "A few hallucinated references are present but do not dominate.",
        "s4": "Almost all references are grounded; at most one minor hallucination.",
        "s5": "All referenced objects and actions are grounded in the scene and reference.",
    },
}


RESULT_RE = re.compile(r"\[RESULT\]\s*([1-5])")


def format_plan(steps: list[str]) -> str:
    if not steps:
        return "(empty plan)"
    return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))


def build_prompt(
    goal: str,
    objects: list[str],
    candidate: list[str],
    reference: list[str],
    rubric_key: str,
) -> str:
    r = RUBRICS[rubric_key]
    return PROMETHEUS_TEMPLATE.format(
        goal=goal or "(unknown)",
        objects=", ".join(objects) if objects else "unknown",
        candidate_plan=format_plan(candidate),
        reference_plan=format_plan(reference),
        rubric_title=r["title"],
        rubric_desc=r["desc"],
        s1=r["s1"],
        s2=r["s2"],
        s3=r["s3"],
        s4=r["s4"],
        s5=r["s5"],
    )


def parse_score(text: str) -> int | None:
    m = RESULT_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def load_judge(model_id: str, dtype: str, device_map: str) -> tuple[Any, Any]:
    import torch
    import transformers

    transformers.set_seed(0)
    torch_dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tok


def generate_feedback(
    model: Any,
    tok: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    attention_mask = (inputs != tok.pad_token_id).long()
    with torch.inference_mode():
        out = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0][inputs.shape[1] :], skip_special_tokens=True)
    return text.strip()


def score_one(
    model: Any,
    tok: Any,
    goal: str,
    objects: list[str],
    candidate: list[str],
    reference: list[str],
    rubric_key: str,
    runs: int,
    temperature: float,
    max_new_tokens: int,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Run the judge `runs` times for one (item, rubric) and return (mean_score, samples)."""
    samples: list[dict[str, Any]] = []
    for run_idx in range(runs):
        prompt = build_prompt(goal, objects, candidate, reference, rubric_key)
        feedback = generate_feedback(model, tok, prompt, max_new_tokens, temperature)
        score = parse_score(feedback)
        samples.append({"run": run_idx, "score": score, "feedback": feedback})
    valid = [s["score"] for s in samples if s["score"] is not None]
    mean = sum(valid) / len(valid) if valid else None
    return mean, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path, help="Manifest with reference_steps/reference_subgoals and objects.")
    parser.add_argument("--predictions", required=True, type=Path, help="Reparsed predictions JSONL (with generated_plan field).")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--details-jsonl", type=Path, default=None, help="Optional per-run feedback JSONL for debugging.")
    parser.add_argument("--rubrics", nargs="+", default=sorted(RUBRICS.keys()), choices=sorted(RUBRICS.keys()))
    parser.add_argument("--runs", type=int, default=3, help="Judge runs per (item, rubric) to average.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N prediction rows.")
    args = parser.parse_args()

    manifest = {row["episode_id"]: row for row in read_jsonl(args.manifest.expanduser())}
    predictions = read_jsonl(args.predictions.expanduser())
    if args.limit:
        predictions = predictions[: args.limit]

    missing = [p["episode_id"] for p in predictions if p["episode_id"] not in manifest]
    if missing:
        raise SystemExit(f"Predictions reference episode_ids not in manifest: {sorted(set(missing))[:5]} ...")

    print(f"Loading judge {args.judge_model} (dtype={args.dtype})")
    model, tok = load_judge(args.judge_model, args.dtype, args.device_map)

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    total = len(predictions) * len(args.rubrics)
    done = 0
    for pred in predictions:
        ref = manifest[pred["episode_id"]]
        reference = ref.get("reference_steps") or ref.get("reference_subgoals", [])
        goal = ref.get("high_level_task", "")
        objects = ref.get("objects", [])
        candidate = pred.get("generated_plan", [])

        row: dict[str, Any] = {
            "episode_id": pred["episode_id"],
            "method": pred["method"],
            "reference_len": len(reference),
            "generated_len": len(candidate),
        }
        rubric_means: list[float] = []
        for rubric_key in args.rubrics:
            mean, samples = score_one(
                model,
                tok,
                goal,
                objects,
                candidate,
                reference,
                rubric_key,
                args.runs,
                args.temperature,
                args.max_new_tokens,
            )
            row[f"{rubric_key}_mean"] = round(mean, 4) if mean is not None else ""
            row[f"{rubric_key}_valid_runs"] = sum(1 for s in samples if s["score"] is not None)
            if mean is not None:
                rubric_means.append(mean)
            details.append(
                {
                    "episode_id": pred["episode_id"],
                    "method": pred["method"],
                    "rubric": rubric_key,
                    "mean": mean,
                    "samples": samples,
                }
            )
            done += 1
            print(
                f"[{done}/{total}] {pred['episode_id']} / {pred['method']} / {rubric_key}: "
                f"mean={mean if mean is not None else 'NA'}"
            )

        row["overall_mean"] = round(sum(rubric_means) / len(rubric_means), 4) if rubric_means else ""
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        ["episode_id", "method", "reference_len", "generated_len"]
        + [f"{r}_mean" for r in args.rubrics]
        + [f"{r}_valid_runs" for r in args.rubrics]
        + ["overall_mean"]
    )
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for method, method_rows in sorted(by_method.items()):
        entry: dict[str, float | int] = {"n": len(method_rows)}
        for rubric_key in args.rubrics:
            vals = [r[f"{rubric_key}_mean"] for r in method_rows if r[f"{rubric_key}_mean"] != ""]
            entry[rubric_key] = sum(vals) / len(vals) if vals else None  # type: ignore[assignment]
        overalls = [r["overall_mean"] for r in method_rows if r["overall_mean"] != ""]
        entry["overall_mean"] = sum(overalls) / len(overalls) if overalls else None  # type: ignore[assignment]
        summary[method] = entry

    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.details_jsonl:
        args.details_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.details_jsonl.open("w", encoding="utf-8") as f:
            for record in details:
                f.write(json.dumps(record) + "\n")

    print(f"\nWrote per-row judge CSV to {args.out_csv}")
    print(f"Wrote judge summary JSON to {args.summary_json}")
    if args.details_jsonl:
        print(f"Wrote per-run details to {args.details_jsonl}")


if __name__ == "__main__":
    main()

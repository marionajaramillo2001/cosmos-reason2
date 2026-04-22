# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a static HTML viewer for per-episode planning comparisons.

For each episode present in the given manifests + predictions, emits a page
showing:
  - the high-level goal
  - the input video clip fed to the model
  - the full reference demonstration video
  - the reference step list (ground truth from AgiBot instruction_segments)
  - one column per method with its generated plan (and subgoals/expansions
    for hierarchical_two_call)
  - optional Prometheus judge scores per method, if a judge CSV is provided

The site is plain static HTML + one CSS file. View locally with:
    python -m http.server 8000 --directory <out_dir>
and browse to http://localhost:8000.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from agibot_planning_common import read_jsonl


METHOD_ORDER = [
    "direct",
    "hierarchical",
    "hierarchical_two_call",
    "intra_task_rag",
    "inter_task_rag",
    "rag",
]


METHOD_LABEL = {
    "direct": "Direct",
    "hierarchical": "Hierarchical (1-call)",
    "hierarchical_two_call": "Hierarchical (2-call)",
    "intra_task_rag": "Intra-task RAG",
    "inter_task_rag": "Inter-task RAG",
    "rag": "RAG",
}


STYLE_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 0; padding: 0 24px 64px; color: #1a1a1a; background: #fafafa; }
header { padding: 24px 0 8px; border-bottom: 1px solid #ddd; margin-bottom: 24px; }
header h1 { margin: 0 0 4px; font-size: 22px; }
header a { color: #0a58ca; text-decoration: none; font-size: 14px; }
header a:hover { text-decoration: underline; }
h2 { margin-top: 32px; border-bottom: 1px solid #eee; padding-bottom: 4px; font-size: 18px; }
h3 { margin: 16px 0 6px; font-size: 15px; }
ul.episode-list { padding-left: 18px; }
ul.episode-list li { margin: 4px 0; }
.grid-videos { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
.grid-videos figure { margin: 0; }
.grid-videos figcaption { font-size: 13px; color: #555; margin-bottom: 4px; }
video { width: 100%; max-height: 360px; background: #000; border-radius: 6px; }
.methods-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.method-card { background: #fff; border: 1px solid #e3e3e3; border-radius: 8px; padding: 12px 14px; }
.method-card h3 { margin-top: 0; }
.method-card ol { padding-left: 22px; margin: 4px 0; }
.method-card li { margin: 2px 0; font-size: 14px; }
.scores { font-size: 13px; color: #333; margin: 6px 0 10px; }
.scores span { display: inline-block; margin-right: 12px; background: #eef4ff; padding: 2px 6px; border-radius: 4px; }
.scores .overall { background: #ffe7c2; font-weight: 600; }
.goal { background: #fff3cd; border-left: 4px solid #f5b400; padding: 10px 14px; border-radius: 4px; }
.objects { color: #444; font-size: 13px; margin: 6px 0 16px; }
.ref-steps { background: #eaf7e8; border-left: 4px solid #2c974b; padding: 10px 14px; border-radius: 4px; }
.ref-steps ol { padding-left: 22px; margin: 4px 0; }
.subgoal-block { margin: 8px 0; }
.subgoal-block .sg-title { font-weight: 600; font-size: 14px; color: #333; }
details.raw { margin-top: 10px; }
details.raw summary { cursor: pointer; color: #666; font-size: 12px; }
details.raw pre { background: #f4f4f4; padding: 8px; border-radius: 4px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; }
table.index { border-collapse: collapse; width: 100%; background: #fff; }
table.index th, table.index td { border: 1px solid #e3e3e3; padding: 6px 10px; font-size: 14px; text-align: left; }
table.index th { background: #f0f0f0; }
"""


def expand_globs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(value))
    return sorted({p.expanduser() for p in paths})


def load_manifests(paths: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for p in paths:
        for row in read_jsonl(p):
            ep = row.get("episode_id")
            if ep:
                index[ep] = row
    return index


def load_predictions(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """Latest row wins if the same (episode_id, method) appears multiple times."""
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    for p in paths:
        for row in read_jsonl(p):
            key = (row.get("episode_id", ""), row.get("method", ""))
            if key[0] and key[1]:
                predictions[key] = row
    return predictions


def load_judge_csv(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {(row["episode_id"], row["method"]): row for row in reader}


def link_video(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return True
    if not src.exists():
        return False
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        try:
            import shutil

            shutil.copy2(src, dst)
        except OSError:
            return False
    return True


def safe_html(text: str) -> str:
    return html.escape(str(text))


def render_steps(steps: list[str]) -> str:
    if not steps:
        return "<p><em>(empty)</em></p>"
    items = "\n".join(f"<li>{safe_html(s)}</li>" for s in steps)
    return f"<ol>{items}</ol>"


def render_scores(score_row: dict[str, str] | None) -> str:
    if not score_row:
        return ""
    rubric_keys = [
        "goal_completion_mean",
        "step_ordering_mean",
        "subgoal_coverage_mean",
        "no_hallucination_mean",
        "overall_mean",
    ]
    chips = []
    for key in rubric_keys:
        val = score_row.get(key, "")
        if val == "":
            continue
        label = key.replace("_mean", "").replace("_", " ")
        cls = "overall" if key == "overall_mean" else ""
        chips.append(f'<span class="{cls}">{safe_html(label)}: {safe_html(val)}</span>')
    if not chips:
        return ""
    return f'<div class="scores">{"".join(chips)}</div>'


def render_method_card(
    method: str,
    pred: dict[str, Any] | None,
    score_row: dict[str, str] | None,
) -> str:
    label = METHOD_LABEL.get(method, method)
    if pred is None:
        return (
            f'<section class="method-card"><h3>{safe_html(label)}</h3>'
            f"<p><em>No prediction available.</em></p></section>"
        )

    body_parts: list[str] = [f"<h3>{safe_html(label)}</h3>"]
    body_parts.append(render_scores(score_row))

    if method == "hierarchical_two_call" and pred.get("expansions"):
        subgoals = pred.get("subgoals") or []
        if subgoals:
            body_parts.append("<p><strong>Subgoals:</strong></p>")
            body_parts.append(render_steps(subgoals))
        for exp in pred["expansions"]:
            title = f"Subgoal {exp.get('subgoal_index', '?')}: {safe_html(exp.get('subgoal', ''))}"
            body_parts.append(
                f'<div class="subgoal-block"><div class="sg-title">{title}</div>'
                f"{render_steps(exp.get('steps', []))}</div>"
            )
        body_parts.append("<p><strong>Flattened plan:</strong></p>")
        body_parts.append(render_steps(pred.get("generated_plan", [])))
    else:
        body_parts.append(render_steps(pred.get("generated_plan", [])))

    raw = pred.get("raw_response", "")
    if raw:
        body_parts.append(
            f'<details class="raw"><summary>raw model response</summary>'
            f"<pre>{safe_html(raw)}</pre></details>"
        )

    return f'<section class="method-card">{"".join(body_parts)}</section>'


def render_episode_page(
    episode_id: str,
    manifest_row: dict[str, Any],
    preds_by_method: dict[str, dict[str, Any]],
    scores_by_method: dict[str, dict[str, str]],
    input_video_rel: str | None,
    reference_video_rel: str | None,
) -> str:
    goal = manifest_row.get("high_level_task", "")
    objects = manifest_row.get("objects") or []
    reference = (
        manifest_row.get("reference_steps")
        or manifest_row.get("reference_subgoals")
        or []
    )

    video_section = []
    if input_video_rel:
        video_section.append(
            f'<figure><figcaption>Input clip (given to the model)</figcaption>'
            f'<video controls preload="metadata" src="{safe_html(input_video_rel)}"></video></figure>'
        )
    if reference_video_rel:
        video_section.append(
            f'<figure><figcaption>Full reference demonstration (ground truth video)</figcaption>'
            f'<video controls preload="metadata" src="{safe_html(reference_video_rel)}"></video></figure>'
        )

    ordered_methods = [m for m in METHOD_ORDER if m in preds_by_method]
    ordered_methods.extend(sorted(m for m in preds_by_method if m not in METHOD_ORDER))

    method_cards = "\n".join(
        render_method_card(m, preds_by_method.get(m), scores_by_method.get(m))
        for m in ordered_methods
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Episode {safe_html(episode_id)}</title>
<link rel="stylesheet" href="../style.css">
</head>
<body>
<header>
  <a href="../index.html">&larr; back to index</a>
  <h1>Episode {safe_html(episode_id)}</h1>
</header>

<h2>Task goal</h2>
<div class="goal">{safe_html(goal) or "<em>(no goal)</em>"}</div>
<p class="objects"><strong>Annotated objects:</strong> {safe_html(", ".join(objects)) if objects else "<em>none</em>"}</p>

<h2>Videos</h2>
<div class="grid-videos">
{''.join(video_section) if video_section else '<p><em>No videos available.</em></p>'}
</div>

<h2>Reference plan (ground truth)</h2>
<div class="ref-steps">{render_steps(reference)}</div>

<h2>Generated plans</h2>
<div class="methods-grid">
{method_cards}
</div>

</body></html>
"""


def render_index_page(
    episodes: list[dict[str, Any]],
    scores_by_ep_method: dict[tuple[str, str], dict[str, str]],
    methods_present: list[str],
) -> str:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.get("task_key", "unknown")].append(ep)

    sections: list[str] = []
    for task_key in sorted(by_task.keys()):
        rows_html: list[str] = []
        header_cells = "".join(
            f"<th>{safe_html(METHOD_LABEL.get(m, m))} overall</th>"
            for m in methods_present
        )
        for ep in sorted(by_task[task_key], key=lambda e: e["episode_id"]):
            score_cells = []
            for m in methods_present:
                row = scores_by_ep_method.get((ep["episode_id"], m))
                score_cells.append(
                    f"<td>{safe_html(row.get('overall_mean', '')) if row else ''}</td>"
                )
            rows_html.append(
                f'<tr><td><a href="episodes/{safe_html(ep["episode_id"])}.html">'
                f'{safe_html(ep["episode_id"])}</a></td>'
                f'<td>{safe_html(ep.get("goal", ""))}</td>'
                f'{"".join(score_cells)}</tr>'
            )
        sections.append(
            f"<h2>{safe_html(task_key)}</h2>"
            f'<table class="index"><thead><tr><th>Episode</th><th>Goal</th>{header_cells}</tr></thead>'
            f'<tbody>{"".join(rows_html)}</tbody></table>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Planning comparison viewer</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header>
  <h1>Planning comparison viewer</h1>
  <p>Per-episode comparison of direct, hierarchical, and RAG planning methods on Cosmos-Reason-derived plans.</p>
</header>
{''.join(sections)}
</body></html>
"""


def infer_task_key(episode_id: str, manifest_row: dict[str, Any]) -> str:
    task_id = manifest_row.get("task_id")
    if task_id:
        return str(task_id)
    # Fall back to prefix parsing of the manifest filename convention used elsewhere.
    return episode_id.split("_", 1)[0] if "_" in episode_id else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True, help="Manifest JSONL paths or glob patterns.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction JSONL paths or glob patterns (merged reparsed files work well).")
    parser.add_argument("--judge-csv", type=Path, default=None, help="Optional Prometheus judge per-row CSV from judge_plans_prometheus.py.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--videos-subdir", default="videos")
    parser.add_argument("--copy-videos", action="store_true", help="Copy video files instead of symlinking.")
    args = parser.parse_args()

    manifest_paths = expand_globs(args.manifests)
    prediction_paths = expand_globs(args.predictions)
    if not manifest_paths:
        raise SystemExit("No manifests found.")
    if not prediction_paths:
        raise SystemExit("No predictions found.")

    manifests = load_manifests(manifest_paths)
    predictions = load_predictions(prediction_paths)
    scores = load_judge_csv(args.judge_csv)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(exist_ok=True)
    (out_dir / args.videos_subdir).mkdir(exist_ok=True)
    (out_dir / "style.css").write_text(STYLE_CSS, encoding="utf-8")

    preds_by_episode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (episode_id, method), pred in predictions.items():
        preds_by_episode[episode_id][method] = pred

    methods_present = sorted(
        {m for per_method in preds_by_episode.values() for m in per_method.keys()},
        key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER), m),
    )

    index_rows: list[dict[str, Any]] = []
    for episode_id, preds_by_method in sorted(preds_by_episode.items()):
        manifest_row = manifests.get(episode_id)
        if manifest_row is None:
            print(f"[skip] no manifest row for {episode_id}")
            continue

        input_src = Path(manifest_row.get("initial_video") or manifest_row.get("clip_path") or "")
        reference_src = Path(manifest_row.get("video_path") or "")

        input_rel: str | None = None
        reference_rel: str | None = None
        if input_src and str(input_src):
            input_dst = out_dir / args.videos_subdir / f"{episode_id}_input{input_src.suffix or '.mp4'}"
            if link_video(input_src, input_dst):
                input_rel = f"../{args.videos_subdir}/{input_dst.name}"
        if reference_src and str(reference_src):
            ref_dst = out_dir / args.videos_subdir / f"{episode_id}_reference{reference_src.suffix or '.mp4'}"
            if link_video(reference_src, ref_dst):
                reference_rel = f"../{args.videos_subdir}/{ref_dst.name}"

        scores_for_ep = {m: scores.get((episode_id, m), {}) for m in preds_by_method.keys()}

        page = render_episode_page(
            episode_id=episode_id,
            manifest_row=manifest_row,
            preds_by_method=preds_by_method,
            scores_by_method=scores_for_ep,
            input_video_rel=input_rel,
            reference_video_rel=reference_rel,
        )
        (out_dir / "episodes" / f"{episode_id}.html").write_text(page, encoding="utf-8")

        index_rows.append(
            {
                "episode_id": episode_id,
                "goal": manifest_row.get("high_level_task", ""),
                "task_key": infer_task_key(episode_id, manifest_row),
            }
        )

    index_html = render_index_page(index_rows, scores, methods_present)
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")

    print(f"Wrote {len(index_rows)} episode pages to {out_dir}/episodes/")
    print(f"Wrote index to {out_dir / 'index.html'}")
    print(
        f"\nView locally:\n  python -m http.server 8000 --directory {out_dir}\n"
        f"  then open http://localhost:8000"
    )


if __name__ == "__main__":
    main()

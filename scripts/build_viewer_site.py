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
.scores span { display: inline-block; margin-right: 8px; margin-bottom: 4px; background: #eef4ff; padding: 2px 6px; border-radius: 4px; }
.scores .overall { background: #ffe7c2; font-weight: 600; }
.scores .embedding { background: #e6f2ea; }
.score-sub { color: #777; font-size: 11px; font-weight: 400; }
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
.legend { background: #fff; border: 1px solid #e3e3e3; border-radius: 8px; padding: 12px 16px; margin: 12px 0 24px; font-size: 13px; }
.legend h3 { margin: 0 0 6px; font-size: 14px; }
.legend ul { margin: 4px 0; padding-left: 20px; }
.legend li { margin: 2px 0; }
.legend code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
[title] { cursor: help; border-bottom: 1px dotted #888; }
th[title] { border-bottom: 1px dotted #888; }
"""


LEGEND_HTML = """
<section class="legend">
  <h3>How to read the scores</h3>
  <p>Each cell shows <strong>judge overall (bold, out of 5)</strong> on top and <em>embedding step coverage (out of 1)</em> below. Hover any header or metric chip for details.</p>
  <ul>
    <li><strong>Judge overall /5</strong> — mean of four Prometheus-2-7B rubrics (1–5 each), averaged across 3 runs at temperature 0.2:
      <ul>
        <li><code>goal_completion</code>: does the plan achieve the high-level goal?</li>
        <li><code>step_ordering</code>: are actions in a valid executable order?</li>
        <li><code>subgoal_coverage</code>: are the reference subgoals covered?</li>
        <li><code>no_hallucination</code>: are all referenced objects/actions grounded in the scene?</li>
      </ul>
    </li>
    <li><strong>Embedding step coverage /1</strong> — for each reference step, take the max cosine similarity (MiniLM embeddings) against any generated step; average across reference steps. Measures semantic recall, robust to paraphrasing.</li>
    <li><strong>Embedding order LCS /1</strong> (episode pages) — longest-common-subsequence over matched step indices (threshold 0.55), normalized by reference length. Measures whether matched steps appear in the right order.</li>
    <li><strong>Object F1 /1</strong> (episode pages) — token-level F1 on object nouns mentioned. Lexical sanity check; easy to game.</li>
  </ul>
</section>
"""


RUBRIC_TOOLTIPS = {
    "goal_completion_mean": "Goal completion (1-5): does the plan achieve the high-level goal?",
    "step_ordering_mean": "Step ordering (1-5): are actions in a valid executable order?",
    "subgoal_coverage_mean": "Subgoal coverage (1-5): are reference subgoals covered?",
    "no_hallucination_mean": "No hallucination (1-5): are all referenced objects/actions grounded?",
    "overall_mean": "Overall = mean of the 4 Prometheus rubrics (scale 1-5)",
}


EMBEDDING_TOOLTIPS = {
    "embedding_step_coverage": "Embedding step coverage (0-1): mean over reference steps of max cosine similarity to any generated step. Semantic recall.",
    "embedding_order_lcs": "Embedding order LCS (0-1): LCS over matched step indices (sim threshold 0.55), normalized by reference length.",
    "object_f1": "Object F1 (0-1): token-level F1 on object nouns in generated vs reference plans.",
}


def expand_globs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(value))
    return sorted({p.expanduser() for p in paths})


def task_name_from_path(path: Path) -> str:
    """Derive a task identifier from a manifest or predictions filename.

    Manifests are named like  task_3400_313498_314085_with_clips.jsonl
    Predictions are named like task_3400_313498_314085_rank0_<flavor>_predictions.jsonl
    or task_3400_313498_314085_rank0_all_methods_reparsed.jsonl
    """
    name = path.name
    for suffix in (
        "_with_clips.jsonl",
        "_all_methods_reparsed.jsonl",
        "_predictions.jsonl",
        "_reparsed.jsonl",
        "_auto_eval.csv",
        "_auto_eval_summary.json",
        "_judge.csv",
        "_judge_summary.json",
        "_judge_details.jsonl",
        "_prometheus.csv",
        "_prometheus_summary.json",
        "_prometheus_details.jsonl",
        ".jsonl",
        ".csv",
        ".json",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    # Strip trailing rank markers like _rank0_same_task or _rank0 so manifest
    # and prediction file stems agree on the task key.
    for marker in ("_rank0_same_task", "_rank0_inter_task", "_rank0"):
        if name.endswith(marker):
            name = name[: -len(marker)]
            break
    return name


def load_manifests(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    """Returns {(task_name, episode_id): row}. Episode IDs repeat across tasks."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for p in paths:
        task = task_name_from_path(p)
        for row in read_jsonl(p):
            ep = row.get("episode_id")
            if ep:
                row = dict(row)
                row.setdefault("_task_name", task)
                index[(task, ep)] = row
    return index


def load_predictions(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Latest row wins on (task_name, episode_id, method)."""
    predictions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for p in paths:
        task = task_name_from_path(p)
        for row in read_jsonl(p):
            ep = row.get("episode_id", "")
            method = row.get("method", "")
            if ep and method:
                predictions[(task, ep, method)] = row
    return predictions


def _attribute_by_episode(
    csv_paths: list[Path],
    manifest_lookup: dict[tuple[str, str], dict[str, Any]],
    source_label: str,
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Read per-row CSV(s) and join to (task_name, episode_id, method).

    Each CSV carries episode_id + method. We attribute task_name from the CSV
    filename when it follows the standard pattern, falling back to manifest
    lookup for any episode_id that is unambiguous across tasks.
    """
    if not csv_paths:
        return {}

    ep_to_tasks: dict[str, list[str]] = defaultdict(list)
    for (task, ep) in manifest_lookup.keys():
        ep_to_tasks[ep].append(task)

    scored: dict[tuple[str, str, str], dict[str, str]] = {}
    ambiguous: list[str] = []
    for path in csv_paths:
        file_task = task_name_from_path(path)
        file_task_valid = bool(manifest_lookup) and any(
            t == file_task for (t, _) in manifest_lookup.keys()
        )
        if not path.exists():
            print(f"[warn] {source_label} CSV not found: {path}")
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ep = row.get("episode_id", "")
                method = row.get("method", "")
                if not ep or not method:
                    continue
                if file_task_valid:
                    scored[(file_task, ep, method)] = row
                    continue
                tasks = ep_to_tasks.get(ep, [])
                if len(tasks) == 1:
                    scored[(tasks[0], ep, method)] = row
                elif len(tasks) > 1:
                    ambiguous.append(f"{ep}/{method}")
    if ambiguous:
        print(
            f"[warn] {len(ambiguous)} {source_label} rows skipped "
            f"(episode_id appears under multiple tasks and CSV filename didn't disambiguate)"
        )
    return scored


def load_judge_csv(
    paths: list[Path],
    manifest_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    return _attribute_by_episode(paths, manifest_lookup, "judge")


def load_embedding_csv(
    paths: list[Path],
    manifest_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    return _attribute_by_episode(paths, manifest_lookup, "embedding")


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


def render_scores(
    judge_row: dict[str, str] | None,
    embedding_row: dict[str, str] | None,
) -> str:
    chips: list[str] = []
    if judge_row:
        for key in [
            "goal_completion_mean",
            "step_ordering_mean",
            "subgoal_coverage_mean",
            "no_hallucination_mean",
            "overall_mean",
        ]:
            val = judge_row.get(key, "")
            if val == "":
                continue
            label = "judge " + key.replace("_mean", "").replace("_", " ")
            cls = "overall" if key == "overall_mean" else ""
            tip = safe_html(RUBRIC_TOOLTIPS.get(key, ""))
            chips.append(
                f'<span class="{cls}" title="{tip}">{safe_html(label)}: {safe_html(val)}/5</span>'
            )
    if embedding_row:
        for key in ("embedding_step_coverage", "embedding_order_lcs", "object_f1"):
            val = embedding_row.get(key, "")
            if val == "":
                continue
            label = key.replace("_", " ")
            tip = safe_html(EMBEDDING_TOOLTIPS.get(key, ""))
            chips.append(
                f'<span class="embedding" title="{tip}">{safe_html(label)}: {safe_html(val)}/1</span>'
            )
    if not chips:
        return ""
    return f'<div class="scores">{"".join(chips)}</div>'


def render_method_card(
    method: str,
    pred: dict[str, Any] | None,
    judge_row: dict[str, str] | None,
    embedding_row: dict[str, str] | None,
) -> str:
    label = METHOD_LABEL.get(method, method)
    if pred is None:
        return (
            f'<section class="method-card"><h3>{safe_html(label)}</h3>'
            f"<p><em>No prediction available.</em></p></section>"
        )

    body_parts: list[str] = [f"<h3>{safe_html(label)}</h3>"]
    body_parts.append(render_scores(judge_row, embedding_row))

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
    task_name: str,
    manifest_row: dict[str, Any],
    preds_by_method: dict[str, dict[str, Any]],
    judge_by_method: dict[str, dict[str, str]],
    embedding_by_method: dict[str, dict[str, str]],
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
        render_method_card(
            m,
            preds_by_method.get(m),
            judge_by_method.get(m),
            embedding_by_method.get(m),
        )
        for m in ordered_methods
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{safe_html(task_name)} / {safe_html(episode_id)}</title>
<link rel="stylesheet" href="../style.css">
</head>
<body>
<header>
  <a href="../index.html">&larr; back to index</a>
  <h1>{safe_html(task_name)} &middot; Episode {safe_html(episode_id)}</h1>
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
    judge_scores: dict[tuple[str, str, str], dict[str, str]],
    embedding_scores: dict[tuple[str, str, str], dict[str, str]],
    methods_present: list[str],
) -> str:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.get("task_name", "unknown")].append(ep)

    sections: list[str] = []
    for task_name in sorted(by_task.keys()):
        rows_html: list[str] = []
        header_cells_parts = []
        for m in methods_present:
            label = safe_html(METHOD_LABEL.get(m, m))
            header_cells_parts.append(
                f'<th title="Top: Prometheus judge overall (mean of 4 rubrics, scale 1-5). Bottom: embedding step coverage (cosine similarity, 0-1).">'
                f'{label}<br><span class="score-sub">judge /5 &middot; emb cov /1</span></th>'
            )
        header_cells = "".join(header_cells_parts)
        for ep in sorted(by_task[task_name], key=lambda e: e["episode_id"]):
            score_cells = []
            for m in methods_present:
                j = judge_scores.get((task_name, ep["episode_id"], m))
                e = embedding_scores.get((task_name, ep["episode_id"], m))
                j_raw = (j.get("overall_mean", "") if j else "") or ""
                e_raw = (e.get("embedding_step_coverage", "") if e else "") or ""
                j_val = f"{j_raw}/5" if j_raw else "nan"
                e_val = f"{e_raw}/1" if e_raw else "nan"
                score_cells.append(
                    f'<td title="judge overall (1-5): {safe_html(j_raw or "n/a")} &#10;embedding step coverage (0-1): {safe_html(e_raw or "n/a")}">'
                    f"<strong>{safe_html(j_val)}</strong>"
                    f'<br><span class="score-sub">{safe_html(e_val)}</span></td>'
                )
            rows_html.append(
                f'<tr><td><a href="episodes/{safe_html(ep["page_slug"])}.html">'
                f'{safe_html(ep["episode_id"])}</a></td>'
                f'<td>{safe_html(ep.get("goal", ""))}</td>'
                f'{"".join(score_cells)}</tr>'
            )
        sections.append(
            f"<h2>{safe_html(task_name)}</h2>"
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
{LEGEND_HTML}
{''.join(sections)}
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True, help="Manifest JSONL paths or glob patterns.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction JSONL paths or glob patterns (merged reparsed files work well).")
    parser.add_argument(
        "--judge-csv",
        nargs="*",
        default=[],
        help="Optional Prometheus judge per-row CSV(s) from judge_plans_prometheus.py. Accepts glob patterns.",
    )
    parser.add_argument(
        "--embedding-csv",
        nargs="*",
        default=[],
        help="Optional embedding-based per-row CSV(s) from evaluate_plans_embedding.py. Accepts glob patterns.",
    )
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
    judge_paths = expand_globs(args.judge_csv) if args.judge_csv else []
    embedding_paths = expand_globs(args.embedding_csv) if args.embedding_csv else []
    scores = load_judge_csv(judge_paths, manifests)
    embedding_scores = load_embedding_csv(embedding_paths, manifests)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(exist_ok=True)
    (out_dir / args.videos_subdir).mkdir(exist_ok=True)
    (out_dir / "style.css").write_text(STYLE_CSS, encoding="utf-8")

    # Group predictions by (task_name, episode_id), collecting methods.
    preds_by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (task_name, episode_id, method), pred in predictions.items():
        preds_by_key[(task_name, episode_id)][method] = pred

    methods_present = sorted(
        {m for per_method in preds_by_key.values() for m in per_method.keys()},
        key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER), m),
    )

    index_rows: list[dict[str, Any]] = []
    orphan_keys: list[tuple[str, str]] = []
    for (task_name, episode_id), preds_by_method in sorted(preds_by_key.items()):
        manifest_row = manifests.get((task_name, episode_id))
        if manifest_row is None:
            orphan_keys.append((task_name, episode_id))
            continue

        page_slug = f"{task_name}__{episode_id}"
        input_src = Path(manifest_row.get("initial_video") or manifest_row.get("clip_path") or "")
        reference_src = Path(manifest_row.get("video_path") or "")

        input_rel: str | None = None
        reference_rel: str | None = None
        if input_src and str(input_src):
            input_dst = out_dir / args.videos_subdir / f"{page_slug}_input{input_src.suffix or '.mp4'}"
            if link_video(input_src, input_dst):
                input_rel = f"../{args.videos_subdir}/{input_dst.name}"
        if reference_src and str(reference_src):
            ref_dst = out_dir / args.videos_subdir / f"{page_slug}_reference{reference_src.suffix or '.mp4'}"
            if link_video(reference_src, ref_dst):
                reference_rel = f"../{args.videos_subdir}/{ref_dst.name}"

        judge_for_ep = {
            m: scores.get((task_name, episode_id, m), {}) for m in preds_by_method.keys()
        }
        emb_for_ep = {
            m: embedding_scores.get((task_name, episode_id, m), {}) for m in preds_by_method.keys()
        }

        page = render_episode_page(
            episode_id=episode_id,
            task_name=task_name,
            manifest_row=manifest_row,
            preds_by_method=preds_by_method,
            judge_by_method=judge_for_ep,
            embedding_by_method=emb_for_ep,
            input_video_rel=input_rel,
            reference_video_rel=reference_rel,
        )
        (out_dir / "episodes" / f"{page_slug}.html").write_text(page, encoding="utf-8")

        index_rows.append(
            {
                "episode_id": episode_id,
                "task_name": task_name,
                "page_slug": page_slug,
                "goal": manifest_row.get("high_level_task", ""),
            }
        )

    if orphan_keys:
        print(f"[skip] {len(orphan_keys)} (task, episode) pairs had predictions but no manifest match:")
        for t, e in orphan_keys[:10]:
            print(f"    {t} / {e}")
        if len(orphan_keys) > 10:
            print(f"    ... and {len(orphan_keys) - 10} more")

    index_html = render_index_page(index_rows, scores, embedding_scores, methods_present)
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")

    print(f"Wrote {len(index_rows)} episode pages to {out_dir}/episodes/")
    print(f"Wrote index to {out_dir / 'index.html'}")
    print(
        f"\nView locally:\n  python -m http.server 8000 --directory {out_dir}\n"
        f"  then open http://localhost:8000"
    )


if __name__ == "__main__":
    main()

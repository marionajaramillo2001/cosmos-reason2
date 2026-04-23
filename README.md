# Cosmos Reason2 — AgiBot Long-Horizon Planning

This branch evaluates [Cosmos Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) on long-horizon video-grounded robot planning using the [AgiBotWorld 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) benchmark.

**Task:** given an initial video clip of a robot in a household/store scene and a high-level goal, generate an ordered step-by-step action plan.

**Four prompting strategies are compared:**

| Method | Description |
|---|---|
| `direct` | Single-turn prompt with goal + clip |
| `hierarchical` | Two-pass: generate subgoals, then expand each into steps |
| `intra_task_rag` | RAG with retrieved examples from the same task |
| `inter_task_rag` | RAG with retrieved examples from other tasks |

**Evaluation set:** 8 tasks × 4 methods = 32 predictions, all using `episode_rank=0`.

---

## Repository Structure

```
cosmos-reason2/
├── run_full_pipeline.sbatch        # Main SLURM job — runs all 6 stages end-to-end
├── run_judge_all.sh                # SLURM job — judge stage only (re-run after predictions)
├── pyproject.toml
├── ruff.toml
│
├── scripts/
│   ├── agibot_cluster_env.sh               # Cluster env setup (sources at session start)
│   ├── agibot_planning_common.py           # Shared helpers used by other scripts
│   ├── run_planning_prompts.py             # Core planner logic (imported, not run directly)
│   │
│   │   # ── Data preparation (one-time setup) ──────────────────────────────
│   ├── download_agibot_task_archives.py    # Download task archives from HuggingFace
│   ├── build_planning_manifest.py          # Build per-task manifests with clip metadata
│   ├── extract_initial_clips.py            # Extract initial MP4 clips for each episode
│   │
│   │   # ── Pipeline stages ────────────────────────────────────────────────
│   ├── patch_manifest_goals.py             # Stage 1: overwrite goals with hand-written ones
│   ├── build_rag_index.py                  # Stage 1.5a: build intra-task RAG index
│   ├── build_inter_task_rag_indexes.py     # Stage 1.5b: build inter-task RAG indexes
│   ├── run_planning_prompts_transformers.py # Stage 2: run Cosmos Reason2 (HF Transformers)
│   ├── reparse_generated_plans.py          # Stage 3: re-parse raw model output into steps
│   ├── evaluate_plans_embedding.py         # Stage 4: embedding-based auto evaluation
│   ├── judge_plans_prometheus.py           # Stage 5: LLM-as-judge (Prometheus-2-7B)
│   └── build_viewer_site.py                # Stage 6: build static HTML viewer
│
├── prompts/
│   ├── planning_direct.yaml                # Prompt template — direct method
│   ├── planning_hierarchical.yaml          # Prompt template — hierarchical method
│   ├── planning_hierarchical_subgoals.yaml # Sub-template — subgoal generation pass
│   ├── planning_hierarchical_expand.yaml   # Sub-template — expansion pass
│   ├── planning_rag.yaml                   # Prompt template — RAG methods
│   ├── task_goals.yaml                     # Hand-written task goals (one per task)
│   └── README.md
│
├── docs/
│   ├── explorer_working_setup.md
│   └── inter_task_analysis.md
│
├── cosmos_reason2_utils/                   # Shared Python package (vision/text helpers)
│
└── cosmos_agibot_planning/                 # Large data root (git-ignored)
    ├── manifests/                          # Per-task *_with_clips.jsonl manifests
    ├── manifests_old_results/              # Backup manifests before goal patching
    ├── clips/                              # Extracted MP4 clips per task/episode
    ├── rag_indexes/
    │   ├── intra/                          # Intra-task RAG indexes
    │   └── inter/                          # Inter-task RAG indexes
    └── hf_cache/                           # HuggingFace model cache
```

**Pipeline outputs** (also git-ignored):

```
planning_outputs/rank0_all/
├── predictions/    # Raw Cosmos Reason2 outputs (JSONL per task)
├── reparsed/       # Parsed step lists (JSONL per task)
├── embedding/      # Embedding eval CSVs + summary JSONs
└── judge/          # Prometheus judge CSVs + summary JSONs

viewer_out/         # Static HTML viewer site
```

---

## Setup

### Prerequisites

- Access to an A100 GPU node (SLURM cluster at IPL Lab)
- `uv` package manager
- Access to the gated AgiBotWorld 2026 HuggingFace dataset

### Environment

```bash
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
git pull
source scripts/agibot_cluster_env.sh
```

`agibot_cluster_env.sh` sets `$EXP_ROOT`, activates the `.venv`, and points the HF/uv caches to the shared project directory.

---

## Data Preparation (one-time)

These steps only need to run once to stage the raw data. Skip if
`cosmos_agibot_planning/manifests/` and `cosmos_agibot_planning/clips/` are already populated.

**1. Download task archives from HuggingFace:**

```bash
uv run --with huggingface_hub scripts/download_agibot_task_archives.py \
  --limit 15 \
  --extract \
  --out "$EXP_ROOT/agibot_task_archives_15.jsonl"
```

**2. Build per-task manifests:**

```bash
python scripts/build_planning_manifest.py
```

Creates `$EXP_ROOT/manifests/<task>_with_clips.jsonl` for each task.

**3. Extract initial video clips:**

```bash
python scripts/extract_initial_clips.py
```

Writes per-episode MP4 clips to `$EXP_ROOT/clips/<task>/`.

---

## Running the Full Pipeline

### Single SLURM job (all stages)

Submit everything in one job:

```bash
sbatch run_full_pipeline.sbatch
```

Monitor progress:

```bash
tail -f logs/pipeline_<jobid>.out
```

The script is **idempotent** — each stage checks for existing outputs and skips completed work. Re-submitting is safe.

### Run only the judge stage

If predictions and reparsed files already exist and you only need to (re-)run Prometheus:

```bash
sbatch run_judge_all.sh
```

---

## Pipeline Stages

### Stage 1 — Patch manifest goals

Replaces the noisy concatenated subgoal strings in each manifest with the hand-written one-sentence goals from `prompts/task_goals.yaml`.

```bash
python scripts/patch_manifest_goals.py \
  --goals prompts/task_goals.yaml \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl
```

### Stage 1.5 — Build RAG indexes

**Intra-task** (examples from the same task):

```bash
python scripts/build_rag_index.py \
  --manifest "$EXP_ROOT/manifests/<task>_with_clips.jsonl" \
  --out "$EXP_ROOT/rag_indexes/intra/<task>_rank0.jsonl" \
  --top-k 3
```

**Inter-task** (examples from other tasks):

```bash
python scripts/build_inter_task_rag_indexes.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --out-dir "$EXP_ROOT/rag_indexes/inter" \
  --top-k 3 \
  --episode-rank 0
```

### Stage 2 — Generate predictions

Runs Cosmos Reason2-2B on all four methods for a given task:

```bash
# Intra methods (direct, hierarchical, intra_task_rag share the intra index)
python scripts/run_planning_prompts_transformers.py \
  --manifest "$EXP_ROOT/manifests/<task>_with_clips.jsonl" \
  --methods direct hierarchical intra_task_rag \
  --rag-index "$EXP_ROOT/rag_indexes/intra/<task>_rank0.jsonl" \
  --episode-rank 0 \
  --out planning_outputs/rank0_all/predictions/<task>_rank0_part1.jsonl

# Inter-task RAG (uses the cross-task index)
python scripts/run_planning_prompts_transformers.py \
  --manifest "$EXP_ROOT/manifests/<task>_with_clips.jsonl" \
  --methods inter_task_rag \
  --rag-index "$EXP_ROOT/rag_indexes/inter/<task>_rank0_inter_task.jsonl" \
  --episode-rank 0 \
  --out planning_outputs/rank0_all/predictions/<task>_rank0_part2.jsonl
```

### Stage 3 — Reparse

Converts raw model outputs (free-form text) into clean numbered step lists:

```bash
python scripts/reparse_generated_plans.py \
  --predictions planning_outputs/rank0_all/predictions/<task>_rank0_all_methods_predictions.jsonl \
  --out         planning_outputs/rank0_all/reparsed/<task>_rank0_all_methods_reparsed.jsonl \
  --pretty-raw-response
```

### Stage 4 — Embedding evaluation

Scores each generated plan against the reference steps using sentence-transformer cosine similarity (model: `all-MiniLM-L6-v2`):

```bash
python scripts/evaluate_plans_embedding.py \
  --manifest    "$EXP_ROOT/manifests/<task>_with_clips.jsonl" \
  --predictions planning_outputs/rank0_all/reparsed/<task>_rank0_all_methods_reparsed.jsonl \
  --out-csv     planning_outputs/rank0_all/embedding/<task>_rank0_auto_eval.csv \
  --summary-json planning_outputs/rank0_all/embedding/<task>_rank0_auto_eval_summary.json
```

### Stage 5 — Prometheus judge

LLM-as-judge scoring with Prometheus-2-7B on four rubrics:
`goal_completion`, `step_ordering`, `subgoal_coverage`, `no_hallucination` (1–5 scale, 3 runs averaged):

```bash
python scripts/judge_plans_prometheus.py \
  --manifest    "$EXP_ROOT/manifests/<task>_with_clips.jsonl" \
  --predictions planning_outputs/rank0_all/reparsed/<task>_rank0_all_methods_reparsed.jsonl \
  --out-csv     planning_outputs/rank0_all/judge/<task>_rank0_prometheus.csv \
  --summary-json planning_outputs/rank0_all/judge/<task>_rank0_prometheus_summary.json \
  --runs 3 --temperature 0.2
```

### Stage 6 — Build viewer

Generates a static HTML site with per-episode comparisons:

```bash
python scripts/build_viewer_site.py \
  --manifests     "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --predictions   planning_outputs/rank0_all/reparsed/*_reparsed.jsonl \
  --embedding-csv planning_outputs/rank0_all/embedding/*_auto_eval.csv \
  --judge-csv     planning_outputs/rank0_all/judge/*_prometheus.csv \
  --out-dir       viewer_out
```

Open `viewer_out/index.html` in a browser to browse results.

---

## Results

After the full pipeline completes, results are under `planning_outputs/rank0_all/`:

| Path | Contents |
|---|---|
| `predictions/` | Raw JSONL outputs from Cosmos Reason2 |
| `reparsed/` | Parsed step-list predictions |
| `embedding/` | Per-task embedding eval CSVs + summary JSONs |
| `judge/` | Per-task Prometheus judge CSVs + summary JSONs |
| `rank0_combined_auto_eval.csv` | Embedding scores across all 8 tasks |
| `rank0_combined_auto_eval_summary.json` | Aggregate metrics by method |

---

## Tasks

| Task ID | Goal |
|---|---|
| `task_3400_313498_314085` | Place the yellow triple-cup yogurt and blue boxed yogurt into the shopping cart |
| `task_3401_352507_353983` | Place two green and three pink bottled dish cleaners into the shopping cart |
| `task_3402_373661_375235` | Put the green and yellow bottled toilet water into the shopping cart |
| `task_3404_305627_321730` | Place the red and white bagged disposable cups into the shopping cart |
| `task_3405_389111_389369` | Place two yellow-capped soy sauce bottles into the shopping cart (×2) |
| `task_3705_312820_313211` | Place pink and yellow bagged items from the cart into the freezer |
| `task_3777_325306_327364` | Fill the popcorn bucket with popcorn using both arms (×2) |
| `task_4053_368961_369296` | Hand a flyer to the customer five times using both arms |

---

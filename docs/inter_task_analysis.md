# Inter-Task AgiBot Analysis Plan

This is the next step after the single-task intra-task pilot. The goal is to compare:

- Intra-task performance: average each prompting method across episodes within each task, then average those task-level scores.
- Inter-task performance: align episodes by within-task index, then average across tasks for the first episodes, second episodes, third episodes, and so on.

The first practical step is to download and extract multiple AgiBot task archives.

## 1. Start on Explorer

```bash
ssh jaramillocivill.m@login.explorer.northeastern.edu
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
git pull
source scripts/agibot_cluster_env.sh
mkdir -p "$AGIBOT_ROOT" "$EXP_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$UV_CACHE_DIR"
uvx hf auth whoami
```

Use a compute/GPU node for large downloads:

```bash
srun --partition=gpu-short \
  --gres=gpu:1 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=04:00:00 \
  --pty bash
```

Inside the allocated node:

```bash
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
source scripts/agibot_cluster_env.sh

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
```

## 2. Preview Available Task Archives

Run a dry run first. This discovers matching `CommercialSpaces/task_*/...tar.gz` archives and prints the first 15 selected task archives without downloading. By default, the downloader selects one archive per task so the 15-task pilot is actually inter-task.

```bash
uv run --with huggingface_hub scripts/download_agibot_task_archives.py \
  --limit 15 \
  --dry-run \
  --out "$EXP_ROOT/agibot_task_archives_15.jsonl"
```

## 3. Download the 15-Task Pilot

For the inter-task pilot, download and extract one archive from each of 15 tasks:

```bash
uv run --with huggingface_hub scripts/download_agibot_task_archives.py \
  --limit 15 \
  --extract \
  --out "$EXP_ROOT/agibot_task_archives_15.jsonl"
```

The output inventory is:

```text
$EXP_ROOT/agibot_task_archives_15.jsonl
```

Each row contains:

```text
task_id
archive_path
local_archive_path
sample_root
```

The extracted task roots include the task id and archive id:

```text
$AGIBOT_ROOT/task_3400_313498_314085_sample/data
$AGIBOT_ROOT/task_<id>_<archive_id>_sample/data
```

## 4. Download All Matching Task Archives

When storage and queue time are available, download and extract one archive from every matching CommercialSpaces task:

```bash
uv run --with huggingface_hub scripts/download_agibot_task_archives.py \
  --all \
  --extract \
  --out "$EXP_ROOT/agibot_task_archives_all.jsonl"
```

This can be resumed. Existing archives and extracted sample roots are skipped unless `--force-download` or `--force-extract` is passed.

To intentionally download every archive chunk inside each task folder, add:

```bash
--all-archives-per-task
```

Do not use that flag for the first inter-task pilot.

## 5. Process Each Downloaded Task

For each `sample_root` in the inventory, run the same per-task pipeline as the intra-task pilot:

```bash
python scripts/inspect_agibot_metadata.py \
  --agibot-root "$TASK_SAMPLE_ROOT"

python scripts/build_planning_manifest.py \
  --agibot-root "$TASK_SAMPLE_ROOT" \
  --out "$EXP_ROOT/manifests/${TASK_ID}.jsonl" \
  --limit 30 \
  --included-only

python scripts/extract_initial_clips.py \
  --manifest "$EXP_ROOT/manifests/${TASK_ID}.jsonl" \
  --out-dir "$EXP_ROOT/clips/${TASK_ID}" \
  --out-manifest "$EXP_ROOT/manifests/${TASK_ID}_with_clips.jsonl" \
  --seconds 5 \
  --fps 4 \
  --overwrite

python scripts/build_rag_index.py \
  --manifest "$EXP_ROOT/manifests/${TASK_ID}_with_clips.jsonl" \
  --out "$EXP_ROOT/rag/${TASK_ID}.jsonl" \
  --top-k 3 \
  --backend lexical
```

This produces the same-task retrieval condition. Use method name `intra_task_rag` for this index when running inference, so it is distinguishable from `inter_task_rag`.

Build different-task RAG indexes:

```bash
mkdir -p "$EXP_ROOT/rag_inter_task"

python scripts/build_inter_task_rag_indexes.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --out-dir "$EXP_ROOT/rag_inter_task" \
  --top-k 3 \
  --backend lexical
```

Then run Cosmos Reason2 with the Transformers runner on a GPU node for each task manifest.

Same-task conditions:

```bash
python scripts/run_planning_prompts_transformers.py \
  --manifest "$MANIFEST" \
  --rag-index "$EXP_ROOT/rag/$NAME.jsonl" \
  --methods direct hierarchical intra_task_rag \
  --model nvidia/Cosmos-Reason2-2B \
  --fps 4 \
  --out "$EXP_ROOT/predictions/${NAME}_predictions.jsonl"
```

Different-task RAG:

```bash
python scripts/run_planning_prompts_transformers.py \
  --manifest "$MANIFEST" \
  --rag-index "$EXP_ROOT/rag_inter_task/${NAME}_inter_task.jsonl" \
  --methods inter_task_rag \
  --model nvidia/Cosmos-Reason2-2B \
  --fps 4 \
  --out "$EXP_ROOT/predictions_inter_task_rag/${NAME}_inter_task_rag_predictions.jsonl"
```

## 6. Aligned Episode-Rank Comparison

For the current inter-task experiment, use one aligned within-task episode rank across the 10 tasks. For rank 0, that gives:

```text
10 tasks x 4 methods = 40 generated predictions
```

Build an inter-task RAG index where each rank-0 query can retrieve only rank-0 examples from the other tasks:

```bash
mkdir -p "$EXP_ROOT/rag_inter_task_rank0"

python scripts/build_inter_task_rag_indexes.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --out-dir "$EXP_ROOT/rag_inter_task_rank0" \
  --top-k 3 \
  --episode-rank 0 \
  --pool-same-rank-only \
  --backend lexical
```

Smoke test one task:

```bash
python scripts/run_episode_rank_comparison.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --intra-rag-dir "$EXP_ROOT/rag" \
  --inter-rag-dir "$EXP_ROOT/rag_inter_task_rank0" \
  --out-dir "$EXP_ROOT/predictions_rank0_smoke" \
  --episode-rank 0 \
  --limit-tasks 1
```

Run all tasks:

```bash
python scripts/run_episode_rank_comparison.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --intra-rag-dir "$EXP_ROOT/rag" \
  --inter-rag-dir "$EXP_ROOT/rag_inter_task_rank0" \
  --out-dir "$EXP_ROOT/predictions_rank0" \
  --episode-rank 0
```

The outputs are split by retrieval condition:

```text
$EXP_ROOT/predictions_rank0/same_task/*_rank0_same_task_predictions.jsonl
$EXP_ROOT/predictions_rank0/inter_task/*_rank0_inter_task_predictions.jsonl
```

Each task should produce four rows total:

```text
direct
hierarchical
intra_task_rag
inter_task_rag
```

## 7. Reparse, Prettify, and Evaluate

After the smoke run, merge the same-task and inter-task prediction files, reparse the generated plans, prettify JSON-style hierarchical outputs into numbered steps, and evaluate against the matching task manifest:

```bash
python scripts/evaluate_episode_rank_comparison.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --predictions-dir "$EXP_ROOT/predictions_rank0_smoke" \
  --out-dir planning_outputs/rank0_smoke \
  --episode-rank 0 \
  --limit-tasks 1
```

After the full 10-task run:

```bash
python scripts/evaluate_episode_rank_comparison.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --predictions-dir "$EXP_ROOT/predictions_rank0" \
  --out-dir planning_outputs/rank0 \
  --episode-rank 0
```

The batch evaluator writes:

```text
planning_outputs/rank0/reparsed/*_rank0_all_methods_reparsed.jsonl
planning_outputs/rank0/per_task/*_rank0_auto_eval.csv
planning_outputs/rank0/per_task/*_rank0_auto_eval_summary.json
planning_outputs/rank0/per_task/*_rank0_human_scores.csv
planning_outputs/rank0/rank0_combined_auto_eval.csv
planning_outputs/rank0/rank0_combined_auto_eval_summary.json
```

## 8. Analysis Shape

Use the per-task evaluation CSVs in two views:

- Intra average: average the method scores within each task, then average those task means across tasks.
- Inter by episode index: within each task, assign `episode_rank` by sorted episode id. Average all rank-0 episodes together, all rank-1 episodes together, and so on across tasks.

This separates "does the method work inside one repeated task family?" from "does the method generalize across different task families at the same within-task episode position?"

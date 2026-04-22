# Explorer Setup

This is the current working setup for running the AgiBot planning pipeline on Northeastern Explorer.

## Fresh Session

```bash
ssh jaramillocivill.m@login.explorer.northeastern.edu
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
git pull
source scripts/agibot_cluster_env.sh
```

Expected storage paths:

```text
PROJECT_ROOT=/projects/ipl_lab/jaramillocivill.m/cosmos-reason2
AGIBOT_ROOT=/projects/ipl_lab/jaramillocivill.m/cosmos-reason2/AgiBotWorld2026
EXP_ROOT=/projects/ipl_lab/jaramillocivill.m/cosmos-reason2/cosmos_agibot_planning
HF_HOME=/projects/ipl_lab/jaramillocivill.m/hf-cache
HF_HUB_CACHE=/projects/ipl_lab/jaramillocivill.m/hf-cache/hub
UV_CACHE_DIR=/projects/ipl_lab/jaramillocivill.m/uv-cache
```

Create directories and check authentication:

```bash
mkdir -p "$AGIBOT_ROOT" "$EXP_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$UV_CACHE_DIR"
uvx hf auth whoami
```

If needed:

```bash
uvx hf auth login
```

## Compute Session

Use a compute/GPU node for Hugging Face downloads and video preprocessing:

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

Inside the node:

```bash
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
git pull
source scripts/agibot_cluster_env.sh

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600
```

## Download Task Archives

Preview the distinct CommercialSpaces tasks:

```bash
uv run --with huggingface_hub scripts/download_agibot_task_archives.py \
  --limit 15 \
  --dry-run \
  --out "$EXP_ROOT/agibot_task_archives_15.jsonl"
```

Download and extract one archive per task:

```bash
uv run --with huggingface_hub scripts/download_agibot_task_archives.py \
  --limit 15 \
  --extract \
  --out "$EXP_ROOT/agibot_task_archives_15.jsonl"
```

The downloader is resumable. Existing archives and extracted roots are skipped unless `--force-download` or `--force-extract` is passed.

## Build Manifests, Clips, and RAG

Install the FFmpeg fallback once:

```bash
uv pip install imageio-ffmpeg
```

For each task root listed in the archive inventory, build:

- a planning manifest
- 5-second initial clips
- a same-task RAG index for `intra_task_rag`

```bash
mkdir -p "$EXP_ROOT/manifests" "$EXP_ROOT/clips" "$EXP_ROOT/rag"
```

Use the batch loop from [Inter-Task Analysis](inter_task_analysis.md) or run the three scripts directly:

```bash
python scripts/build_planning_manifest.py
python scripts/extract_initial_clips.py
python scripts/build_rag_index.py
```

Build different-task RAG indexes:

```bash
mkdir -p "$EXP_ROOT/rag_inter_task"

python scripts/build_inter_task_rag_indexes.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --out-dir "$EXP_ROOT/rag_inter_task" \
  --top-k 3 \
  --backend lexical
```

## Inference

Run Cosmos Reason2 through the Transformers runner. The active method names are:

```text
direct
hierarchical
intra_task_rag
inter_task_rag
```

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

## Evaluation

Reparse predictions before evaluating so hierarchical JSON is flattened into steps:

```bash
python scripts/reparse_generated_plans.py \
  --predictions "$PREDICTIONS" \
  --out "$REPARSED"
```

Evaluate:

```bash
python scripts/evaluate_plans.py \
  --manifest "$MANIFEST" \
  --predictions "$REPARSED" \
  --out-csv "$OUT_CSV" \
  --summary-json "$SUMMARY_JSON" \
  --human-template-out "$HUMAN_TEMPLATE"
```

## Storage Checks

```bash
df -h /projects/ipl_lab/jaramillocivill.m
du -sh "$AGIBOT_ROOT" "$EXP_ROOT" "$HF_HOME" "$UV_CACHE_DIR" 2>/dev/null
```

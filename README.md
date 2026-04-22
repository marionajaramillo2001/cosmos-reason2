# Cosmos Reason2 AgiBot Planning

This branch is a focused project workspace for long-horizon video-grounded robot planning with Cosmos Reason2 on AgiBotWorld2026.

The experiment asks:

```text
initial AgiBot video clip + high-level task goal -> ordered future action plan
```

It compares four prompting conditions:

```text
direct
hierarchical
intra_task_rag
inter_task_rag
```

The current preferred evaluation set is one aligned within-task episode rank across tasks. For the first comparison, use `episode_rank=0`, giving:

```text
10 tasks x 4 methods = 40 generated predictions
```

## Current Workflow

The active Explorer runbooks are:

- [Explorer Working Setup](docs/explorer_working_setup.md)
- [Inter-Task Analysis](docs/inter_task_analysis.md)

The working cluster project path is:

```bash
/projects/ipl_lab/jaramillocivill.m/cosmos-reason2
```

Start each Explorer session with:

```bash
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
git pull
source scripts/agibot_cluster_env.sh
```

## Active Pipeline

Download and extract AgiBot task archives:

```bash
uv run --with huggingface_hub scripts/download_agibot_task_archives.py \
  --limit 15 \
  --extract \
  --out "$EXP_ROOT/agibot_task_archives_15.jsonl"
```

Build per-task manifests, clips, and same-task RAG indexes:

```bash
python scripts/build_planning_manifest.py
python scripts/extract_initial_clips.py
python scripts/build_rag_index.py
```

Build inter-task RAG indexes:

```bash
python scripts/build_inter_task_rag_indexes.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --out-dir "$EXP_ROOT/rag_inter_task" \
  --top-k 3 \
  --backend lexical
```

Run Cosmos Reason2 with Transformers:

```bash
python scripts/run_planning_prompts_transformers.py \
  --manifest "$MANIFEST" \
  --rag-index "$RAG" \
  --methods direct hierarchical intra_task_rag \
  --model nvidia/Cosmos-Reason2-2B \
  --fps 4 \
  --out "$OUT"
```

Run inter-task RAG:

```bash
python scripts/run_planning_prompts_transformers.py \
  --manifest "$MANIFEST" \
  --rag-index "$INTER_TASK_RAG" \
  --methods inter_task_rag \
  --model nvidia/Cosmos-Reason2-2B \
  --fps 4 \
  --out "$OUT"
```

Run the aligned episode-rank comparison:

```bash
python scripts/build_inter_task_rag_indexes.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --out-dir "$EXP_ROOT/rag_inter_task_rank0" \
  --top-k 3 \
  --episode-rank 0 \
  --pool-same-rank-only \
  --backend lexical

python scripts/run_episode_rank_comparison.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --intra-rag-dir "$EXP_ROOT/rag" \
  --inter-rag-dir "$EXP_ROOT/rag_inter_task_rank0" \
  --out-dir "$EXP_ROOT/predictions_rank0" \
  --episode-rank 0
```

Reparse and evaluate:

```bash
python scripts/reparse_generated_plans.py \
  --predictions "$PREDICTIONS" \
  --out "$REPARSED"

python scripts/evaluate_plans.py \
  --manifest "$MANIFEST" \
  --predictions "$REPARSED" \
  --out-csv "$OUT_CSV" \
  --summary-json "$SUMMARY_JSON" \
  --human-template-out "$HUMAN_TEMPLATE"
```

## Active Files

Core scripts:

```text
scripts/agibot_cluster_env.sh
scripts/agibot_planning_common.py
scripts/download_agibot_task_archives.py
scripts/inspect_agibot_metadata.py
scripts/build_planning_manifest.py
scripts/extract_initial_clips.py
scripts/build_rag_index.py
scripts/build_inter_task_rag_indexes.py
scripts/run_episode_rank_comparison.py
scripts/run_planning_prompts.py
scripts/run_planning_prompts_transformers.py
scripts/reparse_generated_plans.py
scripts/evaluate_plans.py
```

Prompt templates:

```text
prompts/planning_direct.yaml
prompts/planning_hierarchical.yaml
prompts/planning_rag.yaml
```

## Data Policy

Do not commit AgiBot data, model weights, clips, raw predictions, or planning outputs. The ignored large-data paths are:

```text
AgiBotWorld2026/
cosmos_agibot_planning/
planning_outputs/
hf-cache/
uv-cache/
```

Large data and caches should live under:

```bash
/projects/ipl_lab/jaramillocivill.m/
```

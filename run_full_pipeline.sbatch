#!/bin/bash
#SBATCH --job-name=cosmos-planning-full
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G

set -eo pipefail

# ------------------------------------------------------------------
# 0. Environment
# ------------------------------------------------------------------
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
mkdir -p logs

source .venv/bin/activate || { echo "venv activation failed"; exit 1; }

export EXP_ROOT=/projects/ipl_lab/jaramillocivill.m/cosmos-reason2/cosmos_agibot_planning
export HF_HOME="$EXP_ROOT/hf_cache"
export TRANSFORMERS_OFFLINE=0
mkdir -p "$HF_HOME"

TASKS=(
  task_3400_313498_314085
  task_3401_352507_353983
  task_3402_373661_375235
  task_3404_305627_321730
  task_3405_389111_389369
  task_3705_312820_313211
  task_3777_325306_327364
  task_4053_368961_369296
)

OUT=planning_outputs/rank0_all
mkdir -p $OUT/predictions $OUT/reparsed $OUT/embedding $OUT/judge


# ------------------------------------------------------------------
# 1. Restore + patch manifests (runs fast, idempotent)
# ------------------------------------------------------------------
echo "=== STAGE 1: manifests ==="
if ! ls "$EXP_ROOT"/manifests/*_with_clips.jsonl >/dev/null 2>&1; then
  echo "Restoring manifests from manifests_old_results/"
  cp "$EXP_ROOT"/manifests_old_results/*_with_clips.jsonl "$EXP_ROOT"/manifests/
  rm -f "$EXP_ROOT"/manifests/task_3477_*_with_clips.jsonl
  rm -f "$EXP_ROOT"/manifests/task_3641_*_with_clips.jsonl
fi

python scripts/patch_manifest_goals.py \
  --goals prompts/task_goals.yaml \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl

# ------------------------------------------------------------------
# 1.5 Rebuild RAG indexes against the patched goals
# ------------------------------------------------------------------
echo "=== STAGE 1.5: RAG indexes ==="
mkdir -p "$EXP_ROOT"/rag_indexes/intra "$EXP_ROOT"/rag_indexes/inter

# intra-task: one index per task (same task, other episodes — since you only have
# episode_000000 per task this pool will be empty; kept for pipeline consistency).
for T in "${TASKS[@]}"; do
  IDX="$EXP_ROOT/rag_indexes/intra/${T}_rank0.jsonl"
  python scripts/build_rag_index.py \
    --manifest "$EXP_ROOT/manifests/${T}_with_clips.jsonl" \
    --out "$IDX" \
    --top-k 3
done

# inter-task: pool across all tasks, one output file per task inside --out-dir
python scripts/build_inter_task_rag_indexes.py \
  --manifests "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --out-dir "$EXP_ROOT/rag_indexes/inter" \
  --top-k 3 \
  --episode-rank 0

echo "=== STAGE 2: predictions ==="
for T in "${TASKS[@]}"; do
  MANIFEST="$EXP_ROOT/manifests/${T}_with_clips.jsonl"
  OUT_PRED="$OUT/predictions/${T}_rank0_all_methods_predictions.jsonl"
  INTRA_IDX="$EXP_ROOT/rag_indexes/intra/${T}_rank0.jsonl"
  INTER_IDX="$EXP_ROOT/rag_indexes/inter/${T}_rank0_inter_task_rag_index.jsonl"

  if [[ -s "$OUT_PRED" ]]; then
    echo "[skip] $T predictions exist"
    continue
  fi

  # Run intra methods (direct + hierarchical don't actually use the index, but the
  # planner accepts them alongside intra_task_rag with the same --rag-index arg).
  PART1="$OUT/predictions/${T}_rank0_part1_intra.jsonl"
  python scripts/run_planning_prompts_transformers.py \
    --manifest "$MANIFEST" \
    --methods direct hierarchical intra_task_rag \
    --rag-index "$INTRA_IDX" \
    --episode-rank 0 \
    --out "$PART1"

  # Inter-task needs the cross-task index.
  PART2="$OUT/predictions/${T}_rank0_part2_inter.jsonl"
  python scripts/run_planning_prompts_transformers.py \
    --manifest "$MANIFEST" \
    --methods inter_task_rag \
    --rag-index "$INTER_IDX" \
    --episode-rank 0 \
    --out "$PART2"

  cat "$PART1" "$PART2" > "$OUT_PRED"
  echo "[ok  ] predictions $T"
done

# ------------------------------------------------------------------
# 3. Reparse (fast, CPU)
# ------------------------------------------------------------------
echo "=== STAGE 3: reparse ==="
for T in "${TASKS[@]}"; do
  IN_PRED="$OUT/predictions/${T}_rank0_all_methods_predictions.jsonl"
  OUT_REP="$OUT/reparsed/${T}_rank0_all_methods_reparsed.jsonl"
  if [[ -s "$OUT_REP" ]]; then
    echo "[skip] $T reparsed exists"; continue
  fi
  python scripts/reparse_generated_plans.py \
    --predictions "$IN_PRED" \
    --out "$OUT_REP" \
    --pretty-raw-response
done

# ------------------------------------------------------------------
# 4. Embedding eval (GPU, small model; reuses HF cache)
# ------------------------------------------------------------------
echo "=== STAGE 4: embedding ==="
for T in "${TASKS[@]}"; do
  CSV="$OUT/embedding/${T}_rank0_auto_eval.csv"
  if [[ -s "$CSV" ]]; then
    echo "[skip] $T embedding exists"; continue
  fi
  python scripts/evaluate_plans_embedding.py \
    --manifest "$EXP_ROOT/manifests/${T}_with_clips.jsonl" \
    --predictions "$OUT/reparsed/${T}_rank0_all_methods_reparsed.jsonl" \
    --out-csv "$CSV" \
    --summary-json "$OUT/embedding/${T}_rank0_auto_eval_summary.json"
done

# ------------------------------------------------------------------
# 5. Prometheus judge (GPU-heavy — runs after Cosmos is gone)
# ------------------------------------------------------------------
echo "=== STAGE 5: judge ==="
for T in "${TASKS[@]}"; do
  CSV="$OUT/judge/${T}_rank0_prometheus.csv"
  if [[ -s "$CSV" ]]; then
    echo "[skip] $T judge exists"; continue
  fi
  python scripts/judge_plans_prometheus.py \
    --manifest "$EXP_ROOT/manifests/${T}_with_clips.jsonl" \
    --predictions "$OUT/reparsed/${T}_rank0_all_methods_reparsed.jsonl" \
    --out-csv "$CSV" \
    --summary-json "$OUT/judge/${T}_rank0_prometheus_summary.json" \
    --runs 3 --temperature 0.2
done

# ------------------------------------------------------------------
# 6. Build viewer
# ------------------------------------------------------------------
echo "=== STAGE 6: viewer ==="
python scripts/build_viewer_site.py \
  --manifests   "$EXP_ROOT"/manifests/*_with_clips.jsonl \
  --predictions $OUT/reparsed/*_reparsed.jsonl \
  --embedding-csv $OUT/embedding/*_auto_eval.csv \
  --judge-csv     $OUT/judge/*_prometheus.csv \
  --out-dir       viewer_out

echo "=== DONE ==="

#!/bin/bash
#SBATCH --job-name=cosmos-planning-full
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G

set -eo pipefail

# ------------------------------------------------------------------
# 0. Environment
# ------------------------------------------------------------------
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
mkdir -p logs

# TODO: source your venv exactly as you do interactively
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

# TODO: confirm these RAG index paths from your prior runs
INTRA_RAG_INDEX="$EXP_ROOT/rag_indexes/intra_task_rank0.jsonl"
INTER_RAG_INDEX="$EXP_ROOT/rag_indexes/inter_task_rank0.jsonl"

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
# 2. Predictions (Cosmos Reason 2B, 4 methods per task)
# ------------------------------------------------------------------
echo "=== STAGE 2: predictions ==="
for T in "${TASKS[@]}"; do
  MANIFEST="$EXP_ROOT/manifests/${T}_with_clips.jsonl"
  OUT_PRED="$OUT/predictions/${T}_rank0_all_methods_predictions.jsonl"
  if [[ -s "$OUT_PRED" ]]; then
    echo "[skip] $T predictions exist"
    continue
  fi
  echo "[run ] predictions $T"
  python scripts/run_planning_prompts_transformers.py \
    --manifest "$MANIFEST" \
    --methods direct hierarchical intra_task_rag inter_task_rag \
    --rag-index "$INTRA_RAG_INDEX" \
    --episode-rank 0 \
    --out "$OUT_PRED"
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

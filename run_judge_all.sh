#!/bin/bash
#SBATCH --job-name=prometheus-judge
#SBATCH --output=logs/judge_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=4G

cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
source scripts/agibot_cluster_env.sh

mkdir -p planning_outputs/rank0_all/judge

for manifest in "$EXP_ROOT"/manifests/*_with_clips.jsonl; do
  task=$(basename "$manifest" _with_clips.jsonl)
  echo "=== judging $task ==="

  python scripts/judge_plans_prometheus.py \
    --manifest "$manifest" \
    --predictions "planning_outputs/rank0_all/reparsed/${task}_rank0_all_methods_reparsed.jsonl" \
    --out-csv "planning_outputs/rank0_all/judge/${task}_rank0_prometheus.csv" \
    --summary-json "planning_outputs/rank0_all/judge/${task}_rank0_prometheus_summary.json" \
    --runs 3
done

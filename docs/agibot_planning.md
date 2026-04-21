# AgiBot Long-Horizon Planning Pipeline

This project evaluates Cosmos Reason2 as a video-grounded text planner:

```text
initial AgiBot video clip + high-level task -> full ordered future plan
```

Use the IPL lab storage for large data and caches:

```bash
export IPL_ROOT=/projects/ipl_lab/jaramillocivill.m
export PROJECT_ROOT=$IPL_ROOT/cosmos-reason2
export AGIBOT_ROOT=$PROJECT_ROOT/AgiBotWorld2026
export EXP_ROOT=$PROJECT_ROOT/cosmos_agibot_planning
export HF_HOME=$IPL_ROOT/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export UV_CACHE_DIR=$IPL_ROOT/uv-cache
```

Do not commit AgiBot videos, parquet files, model weights, extracted clips, or raw predictions. AgiBot data and experiment outputs live inside the project checkout for convenience and are ignored by git. Hugging Face and UV caches live directly under the IPL folder so they can be reused across runs.

On the H100 cluster, the project checkout should live at:

```bash
/projects/ipl_lab/jaramillocivill.m/cosmos-reason2
```

Create the large-data/cache folders:

```bash
mkdir -p "$AGIBOT_ROOT" "$EXP_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$UV_CACHE_DIR"
```

The previously working Explorer setup used `uvx hf`, not the standalone `hf` installer. Start each session with:

```bash
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
source scripts/agibot_cluster_env.sh
```

If `.venv` is missing, recreate it once:

```bash
uv sync --extra cu128
source .venv/bin/activate
```

Authenticate/check Hugging Face with `uvx`:

```bash
uvx hf auth login
uvx hf auth whoami
```

Do not use the standalone `hf` installer unless `uvx hf` fails. On Explorer, that installer may be killed during its pip upgrade step on login nodes.

## Local Prep

Inspect the downloaded metadata:

```bash
python scripts/inspect_agibot_metadata.py --agibot-root "$AGIBOT_ROOT"
```

Build a small planning manifest:

```bash
python scripts/build_planning_manifest.py \
  --agibot-root "$AGIBOT_ROOT" \
  --out "$EXP_ROOT/manifest.jsonl" \
  --limit 30 \
  --included-only
```

Extract only the initial clips:

```bash
python scripts/extract_initial_clips.py \
  --manifest "$EXP_ROOT/manifest.jsonl" \
  --out-dir "$EXP_ROOT/clips" \
  --out-manifest "$EXP_ROOT/manifest_with_clips.jsonl" \
  --seconds 5 \
  --fps 4 \
  --overwrite
```

Build RAG examples:

```bash
python scripts/build_rag_index.py \
  --manifest "$EXP_ROOT/manifest_with_clips.jsonl" \
  --out "$EXP_ROOT/rag_index.jsonl" \
  --top-k 3 \
  --backend auto
```

RAG uses `sentence-transformers/all-MiniLM-L6-v2` when installed. If that package is unavailable, it falls back to lexical token cosine. The final score is:

```text
0.60 task text similarity + 0.25 object overlap + 0.15 plan length similarity
```

## H100 Inference

Pull this branch on the cluster, install the Cosmos Reason2 CUDA/vLLM environment, and point all caches to IPL storage. Start vLLM:

```bash
uv run vllm serve nvidia/Cosmos-Reason2-2B \
  --allowed-local-media-path "$EXP_ROOT" \
  --max-model-len 8192 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --port 8000
```

Run the three prompting methods:

```bash
python scripts/run_planning_prompts.py \
  --manifest "$EXP_ROOT/manifest_with_clips.jsonl" \
  --rag-index "$EXP_ROOT/rag_index.jsonl" \
  --methods direct hierarchical rag \
  --port 8000 \
  --model nvidia/Cosmos-Reason2-2B \
  --fps 4 \
  --out "$EXP_ROOT/predictions.jsonl"
```

## Evaluation

Automatic metrics:

```bash
python scripts/evaluate_plans.py \
  --manifest "$EXP_ROOT/manifest_with_clips.jsonl" \
  --predictions "$EXP_ROOT/predictions.jsonl" \
  --out-csv planning_outputs/agibot_auto_eval.csv \
  --summary-json planning_outputs/agibot_auto_eval_summary.json \
  --human-template-out planning_outputs/agibot_human_scores_template.csv
```

Human rubric scores are the main result:

```text
Goal completion: 0-2
Subgoal coverage: 0-2
Step ordering: 0-2
Object grounding: 0-2
No hallucination: 0-2
Total: 10
```

Score all 30 episodes x 3 methods if possible; otherwise score 15 episodes x 3 methods.

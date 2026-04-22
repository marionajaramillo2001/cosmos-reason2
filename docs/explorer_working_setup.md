# Explorer Working Setup for AgiBot + Cosmos Reason2

This is the clean runbook for the setup that worked on Northeastern Explorer for the AgiBot long-horizon planning pilot. It keeps only the successful path: Explorer project storage, `uvx hf` for Hugging Face, bundled FFmpeg through `imageio-ffmpeg`, H200 GPU inference, and the Transformers runner.

The project checkout used on Explorer is:

```bash
/projects/ipl_lab/jaramillocivill.m/cosmos-reason2
```

## 1. Start a Fresh Explorer Session

```bash
ssh jaramillocivill.m@login.explorer.northeastern.edu
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
```

If the branch does not already track the remote branch, set it once:

```bash
git branch --set-upstream-to=origin/long-horizon-planning long-horizon-planning
```

For every new session, update the repo and load the project environment:

```bash
git pull
source scripts/agibot_cluster_env.sh
```

Create the cache and output folders:

```bash
mkdir -p "$AGIBOT_ROOT" "$EXP_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$UV_CACHE_DIR"
```

Verify that the expected environment is active:

```bash
which uv
uv --version
which python
python --version
echo "$PROJECT_ROOT"
echo "$AGIBOT_ROOT"
echo "$EXP_ROOT"
echo "$HF_HOME"
echo "$HF_HUB_CACHE"
echo "$UV_CACHE_DIR"
```

Expected important paths:

```text
PROJECT_ROOT=/projects/ipl_lab/jaramillocivill.m/cosmos-reason2
AGIBOT_ROOT=/projects/ipl_lab/jaramillocivill.m/cosmos-reason2/AgiBotWorld2026
EXP_ROOT=/projects/ipl_lab/jaramillocivill.m/cosmos-reason2/cosmos_agibot_planning
HF_HOME=/projects/ipl_lab/jaramillocivill.m/hf-cache
HF_HUB_CACHE=/projects/ipl_lab/jaramillocivill.m/hf-cache/hub
UV_CACHE_DIR=/projects/ipl_lab/jaramillocivill.m/uv-cache
```

Check Hugging Face authentication through `uvx`:

```bash
uvx hf auth whoami
```

If authentication is missing:

```bash
uvx hf auth login
uvx hf auth whoami
```

## 2. Download the AgiBot Subset on a Compute Node

Request an interactive GPU-short session for download and data prep:

```bash
srun --partition=gpu-short \
  --gres=gpu:1 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:30:00 \
  --pty bash
```

Inside the allocated node:

```bash
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
git pull
source scripts/agibot_cluster_env.sh
mkdir -p "$AGIBOT_ROOT" "$EXP_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$UV_CACHE_DIR"
uvx hf auth whoami
```

Download the task archive:

```bash
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=600

uvx hf download agibot-world/AgiBotWorld2026 \
  --repo-type dataset \
  --include "ImitationLearning/CommercialSpaces/task_3401/399093_399454.tar.gz" \
  --local-dir "$AGIBOT_ROOT" \
  2>&1 | tee "$EXP_ROOT/agibot_download.log"
```

Verify the archive:

```bash
ls -lh "$AGIBOT_ROOT/ImitationLearning/CommercialSpaces/task_3401/"
tar -tzf "$AGIBOT_ROOT/ImitationLearning/CommercialSpaces/task_3401/399093_399454.tar.gz" | head -50
```

Extract it:

```bash
mkdir -p "$AGIBOT_ROOT/task_3401_sample"

tar -xzf "$AGIBOT_ROOT/ImitationLearning/CommercialSpaces/task_3401/399093_399454.tar.gz" \
  -C "$AGIBOT_ROOT/task_3401_sample"

export AGIBOT_SAMPLE_ROOT="$AGIBOT_ROOT/task_3401_sample/data"
```

Inspect the metadata:

```bash
python scripts/inspect_agibot_metadata.py \
  --agibot-root "$AGIBOT_SAMPLE_ROOT"
```

Expected sample facts:

```text
episodes.jsonl rows: 14
tasks.jsonl rows: 1
total_videos: 98
camera views include observation.images.top_head
```

## 3. Build the Planning Files

Build the task manifest:

```bash
python scripts/build_planning_manifest.py \
  --agibot-root "$AGIBOT_SAMPLE_ROOT" \
  --out "$EXP_ROOT/manifest_task_3401.jsonl" \
  --limit 30 \
  --included-only
```

Verify it:

```bash
wc -l "$EXP_ROOT/manifest_task_3401.jsonl"
head -1 "$EXP_ROOT/manifest_task_3401.jsonl"
```

Expected result:

```text
14 rows
plan buckets: long=14
```

Install the Python FFmpeg fallback used by the clip extractor:

```bash
uv pip install imageio-ffmpeg
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```

Extract 5-second initial clips at 4 FPS:

```bash
python scripts/extract_initial_clips.py \
  --manifest "$EXP_ROOT/manifest_task_3401.jsonl" \
  --out-dir "$EXP_ROOT/clips_task_3401" \
  --out-manifest "$EXP_ROOT/manifest_task_3401_with_clips.jsonl" \
  --seconds 5 \
  --fps 4 \
  --overwrite
```

Verify the clips:

```bash
wc -l "$EXP_ROOT/manifest_task_3401_with_clips.jsonl"
find "$EXP_ROOT/clips_task_3401" -name "*.mp4" | wc -l
ls -lh "$EXP_ROOT/clips_task_3401" | head
```

Expected result:

```text
14 manifest rows
14 extracted initial clips
```

Build the lexical RAG index:

```bash
python scripts/build_rag_index.py \
  --manifest "$EXP_ROOT/manifest_task_3401_with_clips.jsonl" \
  --out "$EXP_ROOT/rag_index_task_3401.jsonl" \
  --top-k 3 \
  --backend lexical
```

Verify it:

```bash
wc -l "$EXP_ROOT/rag_index_task_3401.jsonl"
head -1 "$EXP_ROOT/rag_index_task_3401.jsonl"
```

Expected result:

```text
14 RAG index rows
```

## 4. Run Cosmos Reason2 on H200 with Transformers

Request an H200 interactive GPU session:

```bash
srun --partition=gpu \
  --gres=gpu:h200:1 \
  --nodes=1 \
  --ntasks=1 \
  --mem=4G \
  --time=04:00:00 \
  --pty bash
```

Inside the allocated H200 node:

```bash
cd /projects/ipl_lab/jaramillocivill.m/cosmos-reason2
git pull
source scripts/agibot_cluster_env.sh
unset VLLM_ATTENTION_BACKEND
hostname
nvidia-smi
```

Sanity-check the prompt loader before inference:

```bash
PYTHONPATH=scripts python - <<'PY'
from pathlib import Path
from run_planning_prompts import load_prompt_template
print(load_prompt_template(Path("prompts/planning_direct.yaml")))
PY
```

The printed prompt should include `Goal: {goal}`, `Visible/annotated objects: {objects}`, and the requirements list.

Run a one-example smoke test:

```bash
head -1 "$EXP_ROOT/manifest_task_3401_with_clips.jsonl" > "$EXP_ROOT/manifest_task_3401_one.jsonl"
head -1 "$EXP_ROOT/rag_index_task_3401.jsonl" > "$EXP_ROOT/rag_index_task_3401_one.jsonl"

python scripts/run_planning_prompts_transformers.py \
  --manifest "$EXP_ROOT/manifest_task_3401_one.jsonl" \
  --rag-index "$EXP_ROOT/rag_index_task_3401_one.jsonl" \
  --methods direct hierarchical intra_task_rag \
  --model nvidia/Cosmos-Reason2-2B \
  --fps 4 \
  --out "$EXP_ROOT/predictions_task_3401_smoke.jsonl"
```

Inspect the smoke-test outputs:

```bash
python - <<'PY'
import json, os

path = os.environ["EXP_ROOT"] + "/predictions_task_3401_smoke.jsonl"
for line in open(path):
    row = json.loads(line)
    print("\n====", row["episode_id"], row["method"], "====")
    print(row["raw_response"][:2000])
PY
```

Run the full 14-episode pilot:

```bash
python scripts/run_planning_prompts_transformers.py \
  --manifest "$EXP_ROOT/manifest_task_3401_with_clips.jsonl" \
  --rag-index "$EXP_ROOT/rag_index_task_3401.jsonl" \
  --methods direct hierarchical intra_task_rag \
  --model nvidia/Cosmos-Reason2-2B \
  --fps 4 \
  --out "$EXP_ROOT/predictions_task_3401.jsonl"
```

Expected result:

```text
42 predictions
14 episodes x 3 methods: direct, hierarchical, intra_task_rag
```

## 5. Reparse and Evaluate

Reparse generated plans from the saved raw responses:

```bash
python scripts/reparse_generated_plans.py \
  --predictions "$EXP_ROOT/predictions_task_3401.jsonl" \
  --out "$EXP_ROOT/predictions_task_3401_reparsed.jsonl"
```

Evaluate the reparsed predictions:

```bash
python scripts/evaluate_plans.py \
  --manifest "$EXP_ROOT/manifest_task_3401_with_clips.jsonl" \
  --predictions "$EXP_ROOT/predictions_task_3401_reparsed.jsonl" \
  --out-csv planning_outputs/agibot_auto_eval_task_3401.csv \
  --summary-json planning_outputs/agibot_auto_eval_summary_task_3401.json \
  --human-template-out planning_outputs/agibot_human_scores_task_3401.csv
```

Inspect the automatic summary:

```bash
cat planning_outputs/agibot_auto_eval_summary_task_3401.json
```

The generated human-scoring template is:

```text
planning_outputs/agibot_human_scores_task_3401.csv
```

## 6. Main Output Artifacts

```text
$AGIBOT_ROOT/ImitationLearning/CommercialSpaces/task_3401/399093_399454.tar.gz
$AGIBOT_ROOT/task_3401_sample/data
$EXP_ROOT/manifest_task_3401.jsonl
$EXP_ROOT/manifest_task_3401_with_clips.jsonl
$EXP_ROOT/clips_task_3401/
$EXP_ROOT/rag_index_task_3401.jsonl
$EXP_ROOT/predictions_task_3401.jsonl
$EXP_ROOT/predictions_task_3401_reparsed.jsonl
planning_outputs/agibot_auto_eval_task_3401.csv
planning_outputs/agibot_auto_eval_summary_task_3401.json
planning_outputs/agibot_human_scores_task_3401.csv
```

This pilot uses one AgiBot task archive with 14 same-task-family long-horizon examples. The RAG index is useful for a first comparison, but the final experiment should use broader task coverage if more data can be downloaded and processed in time.

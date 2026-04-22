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

The extracted task roots follow this pattern:

```text
$AGIBOT_ROOT/task_3401_sample/data
$AGIBOT_ROOT/task_<id>_sample/data
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

Then run Cosmos Reason2 with the Transformers runner on an H200 node for each task manifest.

## 6. Analysis Shape

Use the per-task evaluation CSVs in two views:

- Intra average: average the method scores within each task, then average those task means across tasks.
- Inter by episode index: within each task, assign `episode_rank` by sorted episode id. Average all rank-0 episodes together, all rank-1 episodes together, and so on across tasks.

This separates "does the method work inside one repeated task family?" from "does the method generalize across different task families at the same within-task episode position?"

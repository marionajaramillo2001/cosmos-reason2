# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the AgiBot long-horizon planning scripts.

The AgiBot metadata layout has changed across releases and task packs. These
helpers intentionally use conservative field-name heuristics instead of assuming
one exact schema.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


TEXT_FIELDS = (
    "goal",
    "task",
    "task_name",
    "language_instruction",
    "instruction",
    "instruction_text",
    "description",
    "caption",
    "label",
    "name",
    "action",
)

OBJECT_FIELDS = (
    "object",
    "objects",
    "object_name",
    "object_names",
    "target_object",
    "target_objects",
    "bbox_labels",
    "labels",
)

START_FIELDS = ("start_frame", "start_frame_index", "begin_frame", "begin")
END_FIELDS = ("end_frame", "end_frame_index", "success_frame", "stop_frame", "finish")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_meta_file(root: Path, name: str) -> Any:
    path = root / "meta" / name
    if not path.exists():
        return [] if name.endswith(".jsonl") else {}
    return read_jsonl(path) if name.endswith(".jsonl") else read_json(path)


def normalize_episode_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.startswith("episode_"):
        return text
    if text.isdigit():
        return f"episode_{int(text):06d}"
    match = re.search(r"(\d+)$", text)
    if match and "episode" in text:
        return f"episode_{int(match.group(1)):06d}"
    return text


def episode_id_candidates(value: Any) -> list[str]:
    norm = normalize_episode_id(value)
    candidates = [norm]
    match = re.search(r"(\d+)$", norm)
    if match:
        number = int(match.group(1))
        candidates.extend([str(number), f"{number:06d}", f"episode_{number}"])
    return unique_keep_order([c for c in candidates if c])


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def get_first(record: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        if field in record and record[field] not in (None, "", []):
            return record[field]
    return None


def first_text(record: dict[str, Any]) -> str:
    value = get_first(record, TEXT_FIELDS)
    if isinstance(value, str):
        return clean_text(value)
    if value is not None:
        return clean_text(str(value))
    return ""


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def recursive_values_by_key(value: Any, key_names: tuple[str, ...]) -> list[Any]:
    matches = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in key_names:
                matches.append(child)
            matches.extend(recursive_values_by_key(child, key_names))
    elif isinstance(value, list):
        for child in value:
            matches.extend(recursive_values_by_key(child, key_names))
    return matches


def text_items_from_entries(entries: Any) -> list[str]:
    texts = []
    for entry in as_list(entries):
        if isinstance(entry, str):
            texts.append(clean_text(entry))
        elif isinstance(entry, dict):
            text = first_text(entry)
            if text:
                texts.append(text)
    return unique_keep_order([t for t in texts if t])


def sort_segment_entries(entries: Any) -> list[Any]:
    items = as_list(entries)

    def sort_key(item: Any) -> tuple[int, int]:
        if not isinstance(item, dict):
            return (1, 0)
        start = get_first(item, START_FIELDS)
        try:
            return (0, int(start))
        except (TypeError, ValueError):
            return (1, 0)

    return sorted(items, key=sort_key)


def lookup_by_episode(mapping_or_list: Any, episode_id: str) -> Any:
    candidates = episode_id_candidates(episode_id)
    if isinstance(mapping_or_list, dict):
        for candidate in candidates:
            if candidate in mapping_or_list:
                return mapping_or_list[candidate]
        return None
    if isinstance(mapping_or_list, list):
        for item in mapping_or_list:
            if not isinstance(item, dict):
                continue
            item_id = (
                item.get("episode_id")
                or item.get("episode")
                or item.get("episode_index")
                or item.get("index")
            )
            if normalize_episode_id(item_id) == episode_id:
                return item
    return None


def task_lookup(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for task in tasks:
        for key in ("task_id", "task_index", "index", "id"):
            if key in task:
                out[str(task[key])] = task
    return out


def infer_task_text(episode: dict[str, Any], tasks_by_id: dict[str, dict[str, Any]]) -> str:
    text = first_text(episode)
    if text:
        return text
    task_id = (
        episode.get("task_id")
        or episode.get("task_index")
        or episode.get("task")
        or episode.get("task_uuid")
    )
    task = tasks_by_id.get(str(task_id), {})
    return first_text(task)


def extract_instruction_segments(info: dict[str, Any], annotations: dict[str, Any], episode_id: str) -> list[str]:
    for source in (
        info.get("instruction_segments"),
        info.get("instruction_segment"),
        annotations.get("instruction_segments"),
        annotations.get("instruction_segment"),
    ):
        entries = lookup_by_episode(source, episode_id)
        texts = text_items_from_entries(sort_segment_entries(entries))
        if texts:
            return texts
    return []


def extract_task_frames(info: dict[str, Any], annotations: dict[str, Any], episode_id: str) -> list[str]:
    for source in (
        info.get("key_frame"),
        info.get("key_frames"),
        annotations.get("key_frame"),
        annotations.get("key_frames"),
        annotations.get("task_frames"),
    ):
        entries = lookup_by_episode(source, episode_id)
        if entries is None:
            continue
        frames = []
        for entry in sort_segment_entries(entries):
            if not isinstance(entry, dict):
                continue
            frame_type = str(entry.get("frame_type_name") or entry.get("type") or "")
            if frame_type and "task" not in frame_type.lower():
                continue
            text = first_text(entry)
            if text:
                frames.append(text)
        if frames:
            return unique_keep_order(frames)
    return []


def extract_objects(*values: Any) -> list[str]:
    objects = []
    for value in values:
        for match in recursive_values_by_key(value, OBJECT_FIELDS):
            if isinstance(match, str):
                objects.append(match)
            elif isinstance(match, dict):
                objects.extend(str(k) for k in match.keys())
                objects.extend(str(v) for v in match.values() if isinstance(v, str))
            elif isinstance(match, list):
                objects.extend(str(v) for v in match if isinstance(v, (str, int, float)))
    cleaned = []
    for obj in objects:
        obj = clean_text(str(obj)).lower()
        if obj and len(obj) <= 80:
            cleaned.append(obj)
    return unique_keep_order(cleaned)


def plan_length_bucket(reference_steps: list[str], reference_subgoals: list[str]) -> str:
    length = len(reference_steps) or len(reference_subgoals)
    if length <= 4:
        return "short"
    if length <= 8:
        return "medium"
    return "long"


def find_episode_video(root: Path, episode_id: str, preferred_camera: str = "observation.images.top_head") -> str:
    candidates = []
    for ep in episode_id_candidates(episode_id):
        candidates.extend(root.glob(f"videos/**/*{ep}*.mp4"))
    if not candidates:
        return ""
    preferred = [p for p in candidates if preferred_camera in str(p)]
    chosen = sorted(preferred or candidates, key=lambda p: (len(str(p)), str(p)))[0]
    return str(chosen)


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def token_jaccard(a: str, b: str) -> float:
    a_tokens = token_set(a)
    b_tokens = token_set(b)
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def cosine_from_counts(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
    a_norm = math.sqrt(sum(v * v for v in a.values()))
    b_norm = math.sqrt(sum(v * v for v in b.values()))
    if not a_norm or not b_norm:
        return 0.0
    return dot / (a_norm * b_norm)

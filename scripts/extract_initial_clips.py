# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract initial MP4 clips from AgiBot videos for planning prompts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import os
from pathlib import Path

from agibot_planning_common import read_jsonl, write_jsonl


def resolve_ffmpeg() -> str:
    explicit = os.environ.get("FFMPEG_BINARY") or os.environ.get("IMAGEIO_FFMPEG_EXE")
    if explicit:
        return explicit
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError:
        raise SystemExit(
            "ffmpeg is required. Load an ffmpeg module, set FFMPEG_BINARY, "
            "or install imageio-ffmpeg with `uv pip install imageio-ffmpeg`."
        ) from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_clip(video_path: Path, out_path: Path, seconds: float, fps: float, overwrite: bool, ffmpeg: str) -> None:
    if out_path.exists() and not overwrite:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-ss",
        "0",
        "-t",
        str(seconds),
        "-i",
        str(video_path),
        "-r",
        str(fps),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ffmpeg = resolve_ffmpeg()
    print(f"Using ffmpeg: {ffmpeg}")

    rows = read_jsonl(args.manifest.expanduser())
    updated = []
    for row in rows:
        if not row.get("include", True):
            updated.append(row)
            continue
        video_path = Path(row.get("video_path", "")).expanduser()
        if not video_path.exists():
            row["include"] = False
            row["qc_notes"] = (row.get("qc_notes") or "") + " missing source video"
            updated.append(row)
            continue
        out_path = args.out_dir.expanduser() / f"{row['episode_id']}_initial.mp4"
        extract_clip(video_path, out_path, args.seconds, args.fps, args.overwrite, ffmpeg)
        row["initial_video"] = str(out_path)
        row["initial_clip_seconds"] = args.seconds
        row["initial_clip_fps"] = args.fps
        updated.append(row)

    write_jsonl(args.out_manifest.expanduser(), updated)
    print(f"Wrote clip manifest to {args.out_manifest}")


if __name__ == "__main__":
    main()

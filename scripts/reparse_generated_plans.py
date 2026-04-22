# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-parse generated_plan fields from existing raw_response predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from agibot_planning_common import read_jsonl, write_jsonl
from run_planning_prompts import parse_plan


def numbered_steps(steps: list[str]) -> str:
    return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--pretty-raw-response",
        action="store_true",
        help="Rewrite raw_response as a numbered list from parsed generated_plan.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.predictions.expanduser())
    for row in rows:
        row["generated_plan"] = parse_plan(row.get("raw_response", ""))
        if args.pretty_raw_response:
            row["raw_response"] = numbered_steps(row["generated_plan"])
    write_jsonl(args.out.expanduser(), rows)
    print(f"Wrote {len(rows)} reparsed predictions to {args.out}")


if __name__ == "__main__":
    main()

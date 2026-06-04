#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SUMMARY_RE = re.compile(r"^(?P<model>.+)_(?P<modality>image|text)\.summary\.json$")
METRICS = (
    "accuracy",
    "optimal_accuracy",
    "parse_rate",
    "success_rate",
    "optimal_success_rate",
    "illegal_movement_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Base/OPCD image/text modality-gap summaries.")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).resolve().parent / "results/modality_gap_comparison",
    )
    return parser.parse_args()


def metric_block(summary: dict[str, Any], scope: str, task: str | None = None) -> dict[str, float]:
    if scope == "all_tasks_macro_avg":
        block = summary.get("all_tasks_macro_avg", {})
    elif scope == "total":
        block = summary.get("total", {})
    elif scope == "task":
        assert task is not None
        block = summary.get("by_task", {}).get(task, {})
    else:
        raise RuntimeError(f"Unsupported scope: {scope}")
    return {metric: float(block.get(metric, 0.0)) for metric in METRICS}


def collect_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(results_dir.glob("*.summary.json")):
        match = SUMMARY_RE.match(summary_path.name)
        if not match:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_label = str(summary.get("experiment", {}).get("model_label") or match.group("model"))
        modality = str(summary.get("experiment", {}).get("input_modality") or match.group("modality"))

        for scope, task in [("all_tasks_macro_avg", None), ("total", None)]:
            row = {
                "summary_file": str(summary_path),
                "model_label": model_label,
                "input_modality": modality,
                "scope": scope,
                "task": "",
            }
            row.update(metric_block(summary, scope))
            rows.append(row)

        for task in sorted(summary.get("by_task", {})):
            row = {
                "summary_file": str(summary_path),
                "model_label": model_label,
                "input_modality": modality,
                "scope": "task",
                "task": task,
            }
            row.update(metric_block(summary, "task", task))
            rows.append(row)
    return rows


def add_gap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["model_label"], row["scope"], row["task"], row["input_modality"]): row
        for row in rows
    }
    gap_rows: list[dict[str, Any]] = []
    keys = sorted({(row["model_label"], row["scope"], row["task"]) for row in rows})
    for model_label, scope, task in keys:
        image_row = indexed.get((model_label, scope, task, "image"))
        text_row = indexed.get((model_label, scope, task, "text"))
        if not image_row or not text_row:
            continue
        gap = {
            "summary_file": "",
            "model_label": model_label,
            "input_modality": "text_minus_image",
            "scope": scope,
            "task": task,
        }
        for metric in METRICS:
            gap[metric] = float(text_row[metric]) - float(image_row[metric])
        gap_rows.append(gap)
    return rows + gap_rows


def main() -> None:
    args = parse_args()
    rows = add_gap_rows(collect_rows(args.results_dir))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = ["model_label", "input_modality", "scope", "task", *METRICS, "summary_file"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {csv_path}")
    for row in rows:
        if row["scope"] == "all_tasks_macro_avg":
            print(
                f"{row['model_label']:>8} {row['input_modality']:>16} "
                f"macro_acc={row['accuracy']:.4f} optimal={row['optimal_accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()

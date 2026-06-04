#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VLM_DIR = SCRIPT_DIR.parent
CKPT_EVAL_DIR = VLM_DIR / "ckpt_eval"
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(CKPT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(CKPT_EVAL_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluate_visualplanning_ckpt as ckpt_eval  # noqa: E402


TASKS = ckpt_eval.TASKS
DEFAULT_BASE_MODEL = REPO_ROOT / "models" / "Qwen3-VL-8B-Instruct"
DEFAULT_OPCD_MODEL = REPO_ROOT / "models" / "ckpts" / "Qwen3-VL-4B-VSP-Tasks-OPCD-Mixed-v2-Step1400"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "VisualPlanning" / "dataset"
DEFAULT_FROZENLAKE_BENCH_ROOT = REPO_ROOT / "data" / "DiffThinker" / "FrozenLake" / "VSP" / "maps"
DEFAULT_API_CONFIG = REPO_ROOT / "api_config_files/api_config_vllm.json"
DEFAULT_SYSTEM_PROMPT_DIR = REPO_ROOT / "EasyR1/examples/system_prompt"
DEFAULT_TEXT_PROMPT_DIR = SCRIPT_DIR / "prompts"
DEFAULT_IMAGE_CACHE = SCRIPT_DIR / "rendered_images"
DEFAULT_OUTPUT = SCRIPT_DIR / "results/modality_gap_eval.jsonl"
DEFAULT_PROMPT_IMAGES = ckpt_eval.DEFAULT_PROMPT_IMAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VSP modality gap on Base/OPCD models with either image input "
            "or text-only environment input. This script reuses ckpt_eval data and reward logic without modifying it."
        )
    )
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--base-urls", nargs="+", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="", help="Served model name. Empty string queries /models.")
    parser.add_argument("--model-label", default="unknown", help="Metadata label, e.g. base or opcd.")
    parser.add_argument("--input-modality", choices=("image", "text"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--frozenlake-bench-root", type=Path, default=DEFAULT_FROZENLAKE_BENCH_ROOT)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--levels", nargs="+", type=int, default=None)
    parser.add_argument("--samples-per-task", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--system-prompt-dir", type=Path, default=DEFAULT_SYSTEM_PROMPT_DIR)
    parser.add_argument("--text-prompt-dir", type=Path, default=DEFAULT_TEXT_PROMPT_DIR)
    parser.add_argument("--prompt-image-1", type=Path, default=DEFAULT_PROMPT_IMAGES[0])
    parser.add_argument("--prompt-image-2", type=Path, default=DEFAULT_PROMPT_IMAGES[1])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--truncate-prompt-tokens", type=int, default=8192)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--use-system-proxy", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--store-prompt", action="store_true")
    parser.add_argument("--store-text-state", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_prompt(path: Path) -> tuple[Path, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path, path.read_text(encoding="utf-8")


def load_prompts(args: argparse.Namespace) -> dict[str, tuple[Path, str]]:
    if args.input_modality == "image":
        return {
            "frozenlake": read_prompt(args.system_prompt_dir / "opcd_student_Frozenlake.txt"),
            "maze": read_prompt(args.system_prompt_dir / "opcd_student_Maze.txt"),
            "minibehaviour": read_prompt(args.system_prompt_dir / "opcd_student_MiniBehaviour.txt"),
        }
    return {
        "frozenlake": read_prompt(args.text_prompt_dir / "text_student_Frozenlake.txt"),
        "maze": read_prompt(args.text_prompt_dir / "text_student_Maze.txt"),
        "minibehaviour": read_prompt(args.text_prompt_dir / "text_student_MiniBehaviour.txt"),
    }


def coord_text(coord: tuple[int, int] | list[int]) -> str:
    return f"({int(coord[0])},{int(coord[1])})"


def coord_list_text(coords: list[tuple[int, int]] | list[list[int]] | set[tuple[int, int]]) -> str:
    normalized = sorted((int(coord[0]), int(coord[1])) for coord in coords)
    if not normalized:
        return "none"
    return "; ".join(coord_text(coord) for coord in normalized)


def frozenlake_text_state(record: dict[str, Any]) -> str:
    meta = record["meta"]
    level = ckpt_eval.record_level(record)
    desc = ckpt_eval.normalize_frozen_desc(meta["layout"])
    start = ckpt_eval.state_to_coord(int(meta["start_pos"]), level)
    target = ckpt_eval.state_to_coord(int(meta["target_pos"]), level)
    holes = [(row, col) for row, line in enumerate(desc) for col, tile in enumerate(line) if tile == "H"]
    text_map_rows = []
    for row, line in enumerate(desc):
        rendered = list(line)
        rendered[start[1]] = "S" if row == start[0] else rendered[start[1]]
        rendered[target[1]] = "G" if row == target[0] else rendered[target[1]]
        text_map_rows.append("".join(rendered))
    return "\n".join(
        [
            "Text state:",
            "Task: FrozenLake",
            f"Map size: {level}x{level}",
            "Legend: S=player start, F=frozen safe tile, H=hole, G=goal.",
            f"Start position: {coord_text(start)}",
            f"Goal position: {coord_text(target)}",
            f"Hole positions: {coord_list_text(holes)}",
            "Text map:",
            *text_map_rows,
        ]
    )


def open_directions_for_maze_cell(cell: dict[str, bool]) -> list[str]:
    specs = (
        ("up", "north"),
        ("right", "east"),
        ("down", "south"),
        ("left", "west"),
    )
    return [label for label, wall_key in specs if not bool(cell[wall_key])]


def maze_text_state(record: dict[str, Any]) -> str:
    meta = record["meta"]
    level = ckpt_eval.record_level(record)
    start = ckpt_eval.state_to_coord(int(meta["start_pos"]), level)
    target = ckpt_eval.state_to_coord(int(meta["target_pos"]), level)
    lines = [
        "Text state:",
        "Task: Maze",
        f"Map size: {level}x{level}",
        f"Start position: {coord_text(start)}",
        f"Target position: {coord_text(target)}",
        "Open directions:",
    ]
    for row in range(level):
        for col in range(level):
            open_dirs = open_directions_for_maze_cell(meta["layout"][row][col])
            lines.append(f"{coord_text((row, col))}: {', '.join(open_dirs) if open_dirs else 'none'}")
    return "\n".join(lines)


def minibehaviour_text_state(record: dict[str, Any]) -> str:
    meta = record["meta"]
    level = ckpt_eval.record_level(record)
    agent = tuple(meta["start_pos"])
    printer = tuple(meta["printer_pos"])
    table_cells = [tuple(coord) for coord in meta["table_pos"]]
    table_cell_set = set(table_cells)
    printer_neighbors = [tuple(coord) for coord in meta["printer_neighbors"]]
    table_neighbors = [tuple(coord) for coord in meta["table_neighbors"]]
    grid_rows = []
    for row in range(level):
        cells = []
        for col in range(level):
            coord = (row, col)
            if coord == agent:
                cells.append("A")
            elif coord == printer:
                cells.append("P")
            elif coord in table_cell_set:
                cells.append("T")
            else:
                cells.append(".")
        grid_rows.append("".join(cells))
    return "\n".join(
        [
            "Text state:",
            "Task: MiniBehaviour",
            f"Grid size: {level}x{level}",
            f"Agent position: {coord_text(meta['start_pos'])}",
            f"Printer position: {coord_text(meta['printer_pos'])}",
            f"Table cells: {coord_list_text(table_cells)}",
            "Grid legend: A=agent, P=printer, T=table, .=free floor",
            "Grid:",
            *grid_rows,
            f"Cells adjacent to the printer where PICK is legal: {coord_list_text(printer_neighbors)}",
            f"Cells adjacent to the table where DROP is legal after PICK: {coord_list_text(table_neighbors)}",
            "The agent may move through any in-bounds non-table cell.",
            "The printer cell is blocked before PICK; after PICK, the printer is removed and that cell becomes traversable.",
        ]
    )


def text_state_for_record(record: dict[str, Any]) -> str:
    task = record["_task"]
    if task == "frozenlake":
        return frozenlake_text_state(record)
    if task == "maze":
        return maze_text_state(record)
    if task == "minibehaviour":
        return minibehaviour_text_state(record)
    raise RuntimeError(f"Unsupported task: {task}")


def build_text_payload(
    model: str,
    prompt: str,
    text_state: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    repetition_penalty: float | None,
    seed: int | None,
    truncate_prompt_tokens: int | None,
) -> dict[str, Any]:
    full_prompt = prompt.replace("<TEXT-STATE>", text_state)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    if top_k is not None:
        payload["top_k"] = top_k
    if min_p is not None:
        payload["min_p"] = min_p
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if seed is not None:
        payload["seed"] = seed
    if truncate_prompt_tokens is not None:
        payload["truncate_prompt_tokens"] = truncate_prompt_tokens
    return payload


def base_result(
    record: dict[str, Any],
    image_path: Path | None,
    prompt_path: Path | None,
    args: argparse.Namespace,
    text_state: str | None,
) -> dict[str, Any]:
    result = ckpt_eval.base_result(record, image_path, prompt_path)
    result["model_label"] = args.model_label
    result["input_modality"] = args.input_modality
    result["condition_id"] = f"{args.model_label}_{args.input_modality}_{result['case_id']}"
    if text_state is not None and args.store_text_state:
        result["text_state"] = text_state
    return result


def evaluate_one(
    record: dict[str, Any],
    prompts: dict[str, tuple[Path, str]],
    args: argparse.Namespace,
    endpoints: list[ckpt_eval.Endpoint] | None,
    model: str | None,
) -> dict[str, Any]:
    prompt_path, prompt = prompts[record["_task"]]
    text_state: str | None = None
    image_path: Path | None = None
    if args.input_modality == "image":
        image_path = ckpt_eval.render_record_image(record, args.image_cache, args.overwrite_images)
    else:
        text_state = text_state_for_record(record)

    result = base_result(record, image_path, prompt_path, args, text_state)
    if args.store_prompt:
        result["prompt"] = prompt if text_state is None else prompt.replace("<TEXT-STATE>", text_state)

    if args.render_only or args.dry_run:
        result["render_only"] = True
        return result

    assert endpoints is not None and model is not None
    endpoint_index = int(record.get("_eval_index", 0)) % len(endpoints)
    base_url, api_key = endpoints[endpoint_index]
    result["endpoint"] = base_url

    if args.input_modality == "image":
        assert image_path is not None
        payload = ckpt_eval.build_payload(
            model=model,
            prompt=prompt,
            image_path=image_path,
            prompt_image_1=args.prompt_image_1,
            prompt_image_2=args.prompt_image_2,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            truncate_prompt_tokens=args.truncate_prompt_tokens,
        )
    else:
        assert text_state is not None
        payload = build_text_payload(
            model=model,
            prompt=prompt,
            text_state=text_state,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            truncate_prompt_tokens=args.truncate_prompt_tokens,
        )

    response = ckpt_eval.request_json(
        f"{base_url}/chat/completions",
        api_key,
        payload,
        args.timeout,
        args.use_system_proxy,
    )
    generated_text = ckpt_eval.extract_model_text(response)
    actions, answer_text, eval_result = ckpt_eval.evaluate_actions(record, generated_text)
    result.update(
        {
            "generated_text": generated_text,
            "answer_text": answer_text,
            "actions": actions,
            "eval_result": eval_result,
            "error": None,
        }
    )
    if args.print_json:
        result["response"] = response
    return result


def evaluate_one_safe(
    record: dict[str, Any],
    prompts: dict[str, tuple[Path, str]],
    args: argparse.Namespace,
    endpoints: list[ckpt_eval.Endpoint] | None,
    model: str | None,
) -> dict[str, Any]:
    image_path: Path | None = None
    prompt_path: Path | None = None
    text_state: str | None = None
    try:
        prompt_path = prompts[record["_task"]][0]
        if args.input_modality == "image":
            image_path = ckpt_eval.render_record_image(record, args.image_cache, args.overwrite_images)
        else:
            text_state = text_state_for_record(record)
        return evaluate_one(record, prompts, args, endpoints, model)
    except Exception as exc:
        result = base_result(record, image_path, prompt_path, args, text_state)
        result["error"] = str(exc)
        return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return ckpt_eval.read_jsonl(path)


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    ckpt_eval.write_jsonl_row(path, row)


def done_condition_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(output):
        condition_id = row.get("condition_id")
        if isinstance(condition_id, str):
            ids.add(condition_id)
    return ids


def main() -> None:
    args = parse_args()
    if args.samples_per_task < 0:
        raise RuntimeError("--samples-per-task must be >= 0.")
    if args.start_index < 0:
        raise RuntimeError("--start-index must be >= 0.")
    if args.workers < 1:
        raise RuntimeError("--workers must be >= 1.")

    prompts = load_prompts(args)
    levels = set(args.levels) if args.levels else None
    records = ckpt_eval.load_records(
        dataset_root=args.dataset_root,
        frozenlake_bench_root=args.frozenlake_bench_root,
        tasks=args.tasks,
        split=args.split,
        levels=levels,
        start_index=args.start_index,
        samples_per_task=args.samples_per_task,
    )

    if args.resume:
        done = done_condition_ids(args.output)
        next_records = []
        for record in records:
            condition_id = f"{args.model_label}_{args.input_modality}_{record['_case_id']}"
            if condition_id not in done:
                next_records.append(record)
        records = next_records

    print(
        f"[select] cases={len(records)} model_label={args.model_label} "
        f"input_modality={args.input_modality} tasks={','.join(args.tasks)} split={args.split}",
        flush=True,
    )
    if not records:
        return

    if args.dry_run:
        for record in records[:20]:
            prompt_path = prompts[record["_task"]][0]
            if args.input_modality == "image":
                image_path = ckpt_eval.render_record_image(record, args.image_cache, args.overwrite_images)
                print(
                    f"[dry-run] case={record['_case_id']} task={record['_task']} "
                    f"level={ckpt_eval.record_level(record)} prompt={prompt_path} image={image_path}",
                    flush=True,
                )
            else:
                text_state = text_state_for_record(record)
                preview = text_state.replace("\n", " | ")[:400]
                print(
                    f"[dry-run] case={record['_case_id']} task={record['_task']} "
                    f"level={ckpt_eval.record_level(record)} prompt={prompt_path} text={preview}",
                    flush=True,
                )
        if len(records) > 20:
            print(f"[dry-run] omitted {len(records) - 20} more case(s)", flush=True)
        return

    endpoints: list[ckpt_eval.Endpoint] | None = None
    model = None
    if not args.render_only:
        endpoints = ckpt_eval.resolve_endpoints(args)
        model = args.model or ckpt_eval.get_default_model(
            endpoints[0][0],
            endpoints[0][1],
            args.timeout,
            args.use_system_proxy,
        )
        print(f"[api] endpoints={len(endpoints)} model={model}", flush=True)
        for endpoint_index, (base_url, _) in enumerate(endpoints):
            print(f"[api] endpoint[{endpoint_index}]={base_url}", flush=True)

    if args.output.exists() and not args.resume:
        args.output.unlink()

    for eval_index, record in enumerate(records):
        record["_eval_index"] = eval_index

    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for index, record in enumerate(records, start=1):
            row = evaluate_one_safe(record, prompts, args, endpoints, model)
            rows.append(row)
            write_jsonl_row(args.output, row)
            eval_result = row.get("eval_result") or {}
            print(
                f"[{index}/{len(records)}] case={row['case_id']} task={row['task']} "
                f"modality={row['input_modality']} success={eval_result.get('success')} "
                f"optimal={eval_result.get('optimal_success')} error={row.get('error')}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(evaluate_one_safe, record, prompts, args, endpoints, model)
                for record in records
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                write_jsonl_row(args.output, row)
                eval_result = row.get("eval_result") or {}
                print(
                    f"[{index}/{len(records)}] case={row['case_id']} task={row['task']} "
                    f"modality={row['input_modality']} success={eval_result.get('success')} "
                    f"optimal={eval_result.get('optimal_success')} error={row.get('error')}",
                    flush=True,
                )

    summary_rows = read_jsonl(args.output)
    summary = ckpt_eval.summarize_rows(summary_rows)
    summary["experiment"] = {
        "model_label": args.model_label,
        "input_modality": args.input_modality,
        "model": model,
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote rows to {args.output}", flush=True)
    print(f"[done] wrote summary to {summary_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

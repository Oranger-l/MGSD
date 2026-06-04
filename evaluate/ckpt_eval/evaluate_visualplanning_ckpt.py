#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_envs.Frozenlake.generator import choose_cell_size  # noqa: E402
from task_envs.Frozenlake.renderer import render_desc_to_image  # noqa: E402
from task_envs.Maze.renderer import render_maze  # noqa: E402
from task_envs.Minibehaviour.renderer import render_minibehaviour  # noqa: E402


TASKS = ("frozenlake", "maze", "minibehaviour")
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "VisualPlanning" / "dataset"
DEFAULT_FROZENLAKE_BENCH_ROOT = REPO_ROOT / "data" / "DiffThinker" / "FrozenLake" / "VSP" / "maps"
DEFAULT_API_CONFIG = REPO_ROOT / "api_config_files/api_config_vllm.json"
DEFAULT_CKPT_MODEL = "frozenlake-opcd-level5-step120"
DEFAULT_FROZENLAKE_PROMPT = REPO_ROOT / "EasyR1/examples/system_prompt/opcd_student_Frozenlake.txt"
DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_IMAGE_CACHE = Path(__file__).resolve().parent / "rendered_images"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/visualplanning_ckpt_eval.jsonl"
DEFAULT_PROMPT_IMAGES = [
    REPO_ROOT / "EasyR1/examples/prompt_visual_images/frozenlake/system-figure-1.png",
    REPO_ROOT / "EasyR1/examples/prompt_visual_images/frozenlake/system-figure-2.png",
]
Endpoint = tuple[str, str]

MOVE_ALIASES = {
    "L": "left",
    "LEFT": "left",
    "LEFTWARD": "left",
    "LEFTWARDS": "left",
    "R": "right",
    "RIGHT": "right",
    "RIGHTWARD": "right",
    "RIGHTWARDS": "right",
    "U": "up",
    "UP": "up",
    "UPWARD": "up",
    "UPWARDS": "up",
    "D": "down",
    "DOWN": "down",
    "DOWNWARD": "down",
    "DOWNWARDS": "down",
    "0": "left",
    "1": "down",
    "2": "right",
    "3": "up",
}
MINI_ALIASES = {
    **MOVE_ALIASES,
    "PICK": "pick",
    "PICKUP": "pick",
    "PICK-UP": "pick",
    "GRAB": "pick",
    "4": "pick",
    "DROP": "drop",
    "PLACE": "drop",
    "PUT": "drop",
    "5": "drop",
}
MOVE_DELTAS = {
    "left": (0, -1),
    "down": (1, 0),
    "right": (0, 1),
    "up": (-1, 0),
}
LONG_ACTION_PATTERN = re.compile(
    r"\b(?:PICK-UP|PICKUP|PICK|DROP|PLACE|GRAB|LEFTWARD|LEFTWARDS|RIGHTWARD|RIGHTWARDS|UPWARD|UPWARDS|DOWNWARD|DOWNWARDS|LEFT|RIGHT|UP|DOWN|L|R|U|D|0|1|2|3|4|5)\b",
    flags=re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM ckpt on DiffThinker FrozenLake plus VisualPlanning Maze/MiniBehaviour via vLLM."
    )
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument(
        "--base-urls",
        nargs="+",
        default=None,
        help="Optional OpenAI-compatible base URLs. Requests are sharded round-robin across them.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key used with --base-urls. Defaults to the key in --api-config.",
    )
    parser.add_argument("--model", default=DEFAULT_CKPT_MODEL, help="Served model name. Use empty string to query /models.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="VisualPlanning dataset root for Maze and MiniBehaviour.",
    )
    parser.add_argument(
        "--frozenlake-bench-root",
        type=Path,
        default=DEFAULT_FROZENLAKE_BENCH_ROOT,
        help="DiffThinker FrozenLake VSP maps root. Expected levels are level3..level8.",
    )
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--levels", nargs="+", type=int, default=None, help="Optional shared level filter.")
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=0,
        help="Number of filtered samples per task. 0 means all selected samples.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Offset after task/level filtering.")
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--frozenlake-prompt", type=Path, default=DEFAULT_FROZENLAKE_PROMPT)
    parser.add_argument("--prompt-image-1", type=Path, default=DEFAULT_PROMPT_IMAGES[0])
    parser.add_argument("--prompt-image-2", type=Path, default=DEFAULT_PROMPT_IMAGES[1])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096, help="OpenAI max_tokens, equivalent to max_new_tokens.")
    parser.add_argument(
        "--truncate-prompt-tokens",
        type=int,
        default=8192,
        help="vLLM request-side prompt token budget. Use with max_model_len >= prompt + max_tokens.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None, help="vLLM sampling top_k. Omit for server default.")
    parser.add_argument("--min-p", type=float, default=None, help="vLLM sampling min_p. Omit for server default.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="vLLM repetition penalty. Omit for server default.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional request seed if supported by the server.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--use-system-proxy",
        action="store_true",
        help="Use HTTP_PROXY/HTTPS_PROXY for API calls. By default local vLLM calls bypass proxies.",
    )
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip case ids already present in --output.")
    parser.add_argument("--store-prompt", action="store_true", help="Store full prompt text in every JSONL row.")
    parser.add_argument("--print-json", action="store_true", help="Store raw model response JSON in every row.")
    parser.add_argument("--render-only", action="store_true", help="Only render image cache and write metadata rows.")
    parser.add_argument("--dry-run", action="store_true", help="Render selected images and print selected cases.")
    return parser.parse_args()


def load_api_config(path: Path) -> tuple[str, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    base_url = config["base_url"].rstrip("/")
    api_key = config.get("api_key", "")
    if isinstance(api_key, list):
        api_key = api_key[0] if api_key else ""
    return base_url, api_key


def resolve_endpoints(args: argparse.Namespace) -> list[Endpoint]:
    config_base_url, config_api_key = load_api_config(args.api_config)
    if args.base_urls:
        api_key = config_api_key if args.api_key is None else args.api_key
        return [(base_url.rstrip("/"), api_key) for base_url in args.base_urls]
    return [(config_base_url, config_api_key if args.api_key is None else args.api_key)]


def request_json(
    url: str,
    api_key: str,
    payload: dict[str, Any] | None,
    timeout: float,
    use_system_proxy: bool = False,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if payload is None else "POST")
    try:
        opener = None if use_system_proxy else urllib.request.build_opener(urllib.request.ProxyHandler({}))
        open_fn = urllib.request.urlopen if opener is None else opener.open
        with open_fn(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc}") from exc


def get_default_model(base_url: str, api_key: str, timeout: float, use_system_proxy: bool = False) -> str:
    data = request_json(f"{base_url}/models", api_key, None, timeout, use_system_proxy)
    models = data.get("data") or []
    if not models:
        raise RuntimeError("No models returned by /models. Pass --model explicitly.")
    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError(f"Invalid /models response: {data}")
    return str(model_id)


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_model_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response format: {response}") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(texts).strip()
    raise RuntimeError(f"Unexpected message content format: {content}")


def extract_answer_text(text: str) -> str:
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def append_text(content: list[dict[str, Any]], text: str) -> None:
    if text:
        content.append({"type": "text", "text": text})


def append_image(content: list[dict[str, Any]], image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Prompt image not found: {image_path}")
    content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})


def build_content_from_prompt(prompt: str, test_image: Path, prompt_image_1: Path, prompt_image_2: Path) -> list[dict[str, Any]]:
    slots = {
        "<IMAGE-1>": prompt_image_1,
        "<IMAGE-2>": prompt_image_2,
        "<TEST-IMAGE>": test_image,
    }
    if not any(slot in prompt for slot in slots):
        return [
            {"type": "image_url", "image_url": {"url": image_to_data_url(test_image)}},
            {"type": "text", "text": prompt},
        ]

    content: list[dict[str, Any]] = []
    remaining = prompt
    while remaining:
        positions = [(remaining.find(token), token) for token in slots]
        positions = [(pos, token) for pos, token in positions if pos != -1]
        if not positions:
            append_text(content, remaining)
            break
        pos, token = min(positions, key=lambda item: item[0])
        append_text(content, remaining[:pos])
        append_image(content, slots[token])
        remaining = remaining[pos + len(token) :]
    return content


def build_payload(
    model: str,
    prompt: str,
    image_path: Path,
    prompt_image_1: Path,
    prompt_image_2: Path,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    repetition_penalty: float | None,
    seed: int | None,
    truncate_prompt_tokens: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_content_from_prompt(prompt, image_path, prompt_image_1, prompt_image_2),
            }
        ],
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


def dataset_path(dataset_root: Path, task: str, split: str) -> Path:
    return dataset_root / task / "tokenized_dataset/SFT" / f"{split}_dataset.jsonl"


def numeric_stem_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return sys.maxsize, path.name


def parse_diffthinker_level(level_dir: Path) -> int:
    match = re.fullmatch(r"level(\d+)", level_dir.name)
    if not match:
        raise RuntimeError(f"Unexpected DiffThinker FrozenLake level directory: {level_dir}")
    return int(match.group(1))


def parse_diffthinker_frozenlake_table(table_path: Path) -> tuple[list[str], int, int]:
    if not table_path.exists():
        raise FileNotFoundError(f"Missing DiffThinker FrozenLake table: {table_path}")

    char_map = {"@": "S", "*": "G", "#": "H", "_": "F", "S": "S", "G": "G", "H": "H", "F": "F"}
    rows: list[str] = []
    start_state: int | None = None
    target_state: int | None = None

    for line in table_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "Col" in line or "---" in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        clean_parts = [part for part in parts if part]
        if len(clean_parts) < 2 or not clean_parts[0].lower().startswith("row"):
            continue

        row_chars: list[str] = []
        row_index = len(rows)
        for col_index, raw_char in enumerate(clean_parts[1:]):
            tile = char_map.get(raw_char, "F")
            if tile == "S":
                start_state = row_index * len(clean_parts[1:]) + col_index
            elif tile == "G":
                target_state = row_index * len(clean_parts[1:]) + col_index
            row_chars.append(tile)
        rows.append("".join(row_chars))

    if not rows:
        raise RuntimeError(f"Failed to parse FrozenLake table rows: {table_path}")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise RuntimeError(f"Non-rectangular FrozenLake table: {table_path}")
    if start_state is None:
        raise RuntimeError(f"Missing start '@' in FrozenLake table: {table_path}")
    if target_state is None:
        raise RuntimeError(f"Missing goal '*' in FrozenLake table: {table_path}")
    return rows, start_state, target_state


def record_level(record: dict[str, Any]) -> int:
    return int(record["meta"]["level"])


def case_id_for(task: str, split: str, line_index: int, record: dict[str, Any]) -> str:
    level = record_level(record)
    if task == "minibehaviour" and "idx" in record.get("meta", {}):
        return f"{task}_{split}_level{level}_{int(record['meta']['idx']):04d}"
    return f"{task}_{split}_level{level}_{line_index:04d}"


def shortest_frozenlake_path_states(desc: list[str], start_state: int, target_state: int) -> list[int]:
    rows = len(desc)
    cols = len(desc[0])
    start = (start_state // cols, start_state % cols)
    target = (target_state // cols, target_state % cols)
    queue: deque[tuple[int, int]] = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current == target:
            break
        for action in ("left", "down", "right", "up"):
            next_coord = move_coord(current, action)
            row, col = next_coord
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            if desc[row][col] == "H" or next_coord in parents:
                continue
            parents[next_coord] = current
            queue.append(next_coord)

    if target not in parents:
        raise RuntimeError("No valid path from start to goal in DiffThinker FrozenLake map.")

    path: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = target
    while cursor is not None:
        path.append(cursor)
        cursor = parents[cursor]
    path.reverse()
    return [row * cols + col for row, col in path]


def load_diffthinker_frozenlake_records(
    frozenlake_bench_root: Path,
    split: str,
    levels: set[int] | None,
    start_index: int,
    samples_per_task: int,
) -> list[dict[str, Any]]:
    if not frozenlake_bench_root.exists():
        raise FileNotFoundError(f"Missing DiffThinker FrozenLake bench root: {frozenlake_bench_root}")

    records: list[dict[str, Any]] = []
    seen = 0
    level_dirs = sorted(
        [path for path in frozenlake_bench_root.iterdir() if path.is_dir() and path.name.startswith("level")],
        key=parse_diffthinker_level,
    )
    for level_dir in level_dirs:
        level = parse_diffthinker_level(level_dir)
        if levels is not None and level not in levels:
            continue

        metadata_path = level_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        for image_path in sorted((level_dir / "img").glob("*.png"), key=numeric_stem_key):
            if seen < start_index:
                seen += 1
                continue
            table_path = level_dir / "table" / f"{image_path.stem}.txt"
            pure_text_path = level_dir / "pure_text" / f"{image_path.stem}.txt"
            desc, start_state, target_state = parse_diffthinker_frozenlake_table(table_path)
            if len(desc) != level or any(len(row) != level for row in desc):
                raise RuntimeError(f"FrozenLake map level/shape mismatch: {table_path}")
            path_states = shortest_frozenlake_path_states(desc, start_state, target_state)
            reference_actions = metadata.get(image_path.name)
            case_index = int(image_path.stem) if image_path.stem.isdigit() else len(records)
            record = {
                "_task": "frozenlake",
                "_split": split,
                "_cache_split": "diffthinker_vsp",
                "_line_index": case_index,
                "_case_id": f"frozenlake_diffthinker_vsp_level{level}_{case_index:04d}",
                "_source_bench": "diffthinker_frozenlake_vsp",
                "_source_image": str(image_path),
                "_source_table": str(table_path),
                "_source_pure_text": str(pure_text_path) if pure_text_path.exists() else None,
                "input_state": path_states,
                "meta": {
                    "level": level,
                    "layout": desc,
                    "start_pos": start_state,
                    "target_pos": target_state,
                    "distance_map": {str(start_state): len(path_states) - 1},
                    "source_bench": "diffthinker_frozenlake_vsp",
                    "source_image": str(image_path),
                    "source_table": str(table_path),
                    "source_pure_text": str(pure_text_path) if pure_text_path.exists() else None,
                    "reference_actions": reference_actions,
                },
            }
            records.append(record)
            seen += 1
            if samples_per_task > 0 and len(records) >= samples_per_task:
                return records
    return records


def load_visualplanning_task_records(
    dataset_root: Path,
    task: str,
    split: str,
    levels: set[int] | None,
    start_index: int,
    samples_per_task: int,
) -> list[dict[str, Any]]:
    path = dataset_path(dataset_root, task, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing VisualPlanning dataset file: {path}")

    records: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            record = json.loads(line)
            level = record_level(record)
            if levels is not None and level not in levels:
                continue
            if seen < start_index:
                seen += 1
                continue
            record["_task"] = task
            record["_split"] = split
            record["_line_index"] = line_index
            record["_case_id"] = case_id_for(task, split, line_index, record)
            records.append(record)
            seen += 1
            if samples_per_task > 0 and len(records) >= samples_per_task:
                break
    return records


def load_records(
    dataset_root: Path,
    frozenlake_bench_root: Path,
    tasks: list[str],
    split: str,
    levels: set[int] | None,
    start_index: int,
    samples_per_task: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task in tasks:
        if task == "frozenlake":
            selected.extend(
                load_diffthinker_frozenlake_records(
                    frozenlake_bench_root=frozenlake_bench_root,
                    split=split,
                    levels=levels,
                    start_index=start_index,
                    samples_per_task=samples_per_task,
                )
            )
        else:
            selected.extend(
                load_visualplanning_task_records(
                    dataset_root=dataset_root,
                    task=task,
                    split=split,
                    levels=levels,
                    start_index=start_index,
                    samples_per_task=samples_per_task,
                )
            )
    return selected


def output_image_path(image_cache: Path, task: str, split: str, level: int, case_id: str) -> Path:
    return image_cache / task / split / f"level{level}" / f"{case_id}.png"


def resize_if_needed(image: Image.Image, image_size: int = 256) -> Image.Image:
    if image.size == (image_size, image_size):
        return image
    return image.resize((image_size, image_size), resample=Image.Resampling.LANCZOS)


def normalize_frozen_desc(layout: list[Any]) -> list[str]:
    rows: list[str] = []
    for row in layout:
        if isinstance(row, str):
            rows.append(row)
        else:
            rows.append("".join(str(cell) for cell in row))
    return rows


def state_to_coord(state: int, level: int) -> tuple[int, int]:
    return int(state) // level, int(state) % level


def maze_grid_from_meta(layout: list[list[dict[str, bool]]]) -> list[list[dict[str, bool]]]:
    return [
        [
            {
                "N": bool(cell["north"]),
                "E": bool(cell["east"]),
                "S": bool(cell["south"]),
                "W": bool(cell["west"]),
            }
            for cell in row
        ]
        for row in layout
    ]


def render_record_image(record: dict[str, Any], image_cache: Path, overwrite: bool) -> Path:
    task = record["_task"]
    split = record.get("_cache_split", record["_split"])
    level = record_level(record)
    case_id = record["_case_id"]
    meta = record["meta"]
    image_path = output_image_path(image_cache, task, split, level, case_id)
    if image_path.exists() and not overwrite:
        return image_path

    image_path.parent.mkdir(parents=True, exist_ok=True)
    if task == "frozenlake":
        source_image = record.get("_source_image") or meta.get("source_image")
        if source_image:
            image = Image.open(Path(source_image)).convert("RGB")
        else:
            desc = normalize_frozen_desc(meta["layout"])
            image = render_desc_to_image(desc, cell_size=choose_cell_size(level), agent_state=int(meta["start_pos"]))
        image = resize_if_needed(image)
    elif task == "maze":
        grid = maze_grid_from_meta(meta["layout"])
        start = state_to_coord(int(meta["start_pos"]), level)
        target = state_to_coord(int(meta["target_pos"]), level)
        image = render_maze(grid, start, target, image_size=256)
    elif task == "minibehaviour":
        image = render_minibehaviour(
            level=level,
            agent_pos=tuple(meta["start_pos"]),
            printer_pos=tuple(meta["printer_pos"]),
            table_pos=[tuple(coord) for coord in meta["table_pos"]],
            image_size=256,
        )
    else:
        raise RuntimeError(f"Unsupported task: {task}")

    image.save(image_path)
    return image_path


def load_prompts(args: argparse.Namespace) -> dict[str, tuple[Path, str]]:
    prompts: dict[str, tuple[Path, str]] = {
        "frozenlake": (args.frozenlake_prompt, args.frozenlake_prompt.read_text(encoding="utf-8")),
        "maze": (
            args.prompt_dir / "opcd_student_Maze.txt",
            (args.prompt_dir / "opcd_student_Maze.txt").read_text(encoding="utf-8"),
        ),
        "minibehaviour": (
            args.prompt_dir / "opcd_student_MiniBehaviour.txt",
            (args.prompt_dir / "opcd_student_MiniBehaviour.txt").read_text(encoding="utf-8"),
        ),
    }
    return prompts


def parse_actions(text: str, task: str) -> tuple[list[str], list[str], str]:
    answer = extract_answer_text(text)
    compact = re.sub(r"[\s,;|/>\-]+", "", answer).upper()
    aliases = MINI_ALIASES if task == "minibehaviour" else MOVE_ALIASES

    if task in {"frozenlake", "maze"} and compact and re.fullmatch(r"[LRUD0123]+", compact):
        return [MOVE_ALIASES[token] for token in compact], [], answer

    actions: list[str] = []
    invalid: list[str] = []
    for match in LONG_ACTION_PATTERN.finditer(answer):
        token = match.group(0).upper()
        normalized = aliases.get(token)
        if normalized is None:
            invalid.append(match.group(0))
        else:
            if task != "minibehaviour" and normalized in {"pick", "drop"}:
                invalid.append(match.group(0))
            else:
                actions.append(normalized)
    return actions, invalid, answer


def expected_action_count(record: dict[str, Any]) -> int | None:
    input_state = record.get("input_state")
    if isinstance(input_state, list) and input_state:
        return max(0, len(input_state) - 1)
    meta = record.get("meta", {})
    start_pos = meta.get("start_pos")
    distance_map = meta.get("distance_map")
    if start_pos is not None and isinstance(distance_map, dict):
        value = distance_map.get(str(start_pos))
        return int(value) if value is not None else None
    return None


def move_coord(coord: tuple[int, int], action: str) -> tuple[int, int]:
    drow, dcol = MOVE_DELTAS[action]
    return coord[0] + drow, coord[1] + dcol


def eval_frozenlake(record: dict[str, Any], actions: list[str], invalid_tokens: list[str]) -> dict[str, Any]:
    meta = record["meta"]
    level = record_level(record)
    desc = normalize_frozen_desc(meta["layout"])
    start = state_to_coord(int(meta["start_pos"]), level)
    target = state_to_coord(int(meta["target_pos"]), level)
    current = start
    path = [list(current)]
    invalid_reason = None
    fell_into_hole = False
    steps_executed = 0

    for action in actions:
        next_coord = move_coord(current, action)
        steps_executed += 1
        row, col = next_coord
        if not (0 <= row < level and 0 <= col < level):
            invalid_reason = f"out_of_bounds:{action}"
            break
        tile = desc[row][col]
        path.append([row, col])
        current = next_coord
        if tile == "H":
            invalid_reason = "fell_into_hole"
            fell_into_hole = True
            break
        if current == target:
            break

    success = current == target and invalid_reason is None
    expected = expected_action_count(record)
    return {
        "parse_success": bool(actions) and not invalid_tokens,
        "success": success,
        "optimal_success": success and expected is not None and steps_executed == expected,
        "expected_action_count": expected,
        "action_count": len(actions),
        "steps_executed": steps_executed,
        "final_coord": list(current),
        "target_coord": list(target),
        "path": path,
        "fell_into_hole": fell_into_hole,
        "invalid_reason": invalid_reason,
        "invalid_action_tokens": invalid_tokens,
    }


def maze_wall_blocks(layout: dict[str, bool], action: str) -> bool:
    key = {"up": "north", "down": "south", "left": "west", "right": "east"}[action]
    return bool(layout[key])


def eval_maze(record: dict[str, Any], actions: list[str], invalid_tokens: list[str]) -> dict[str, Any]:
    meta = record["meta"]
    level = record_level(record)
    layout = meta["layout"]
    start = state_to_coord(int(meta["start_pos"]), level)
    target = state_to_coord(int(meta["target_pos"]), level)
    current = start
    path = [list(current)]
    invalid_reason = None
    steps_executed = 0

    for action in actions:
        steps_executed += 1
        row, col = current
        if maze_wall_blocks(layout[row][col], action):
            invalid_reason = f"wall_blocked:{action}"
            break
        next_coord = move_coord(current, action)
        nrow, ncol = next_coord
        if not (0 <= nrow < level and 0 <= ncol < level):
            invalid_reason = f"out_of_bounds:{action}"
            break
        current = next_coord
        path.append([nrow, ncol])
        if current == target:
            break

    success = current == target and invalid_reason is None
    expected = expected_action_count(record)
    return {
        "parse_success": bool(actions) and not invalid_tokens,
        "success": success,
        "optimal_success": success and expected is not None and steps_executed == expected,
        "expected_action_count": expected,
        "action_count": len(actions),
        "steps_executed": steps_executed,
        "final_coord": list(current),
        "target_coord": list(target),
        "path": path,
        "invalid_reason": invalid_reason,
        "invalid_action_tokens": invalid_tokens,
    }


def eval_minibehaviour(record: dict[str, Any], actions: list[str], invalid_tokens: list[str]) -> dict[str, Any]:
    meta = record["meta"]
    level = record_level(record)
    current = tuple(meta["start_pos"])
    printer = tuple(meta["printer_pos"])
    table_cells = {tuple(coord) for coord in meta["table_pos"]}
    printer_neighbors = {tuple(coord) for coord in meta["printer_neighbors"]}
    table_neighbors = {tuple(coord) for coord in meta["table_neighbors"]}
    path = [[list(current), False]]
    carrying = False
    picked = False
    dropped = False
    invalid_reason = None
    steps_executed = 0

    for action in actions:
        steps_executed += 1
        if action in MOVE_DELTAS:
            next_coord = move_coord(current, action)
            row, col = next_coord
            if not (0 <= row < level and 0 <= col < level):
                invalid_reason = f"out_of_bounds:{action}"
                break
            if next_coord in table_cells:
                invalid_reason = "entered_table"
                break
            if next_coord == printer and not carrying:
                invalid_reason = "entered_printer"
                break
            current = next_coord
        elif action == "pick":
            if carrying:
                invalid_reason = "pick_while_carrying"
                break
            if current not in printer_neighbors:
                invalid_reason = "pick_not_adjacent_to_printer"
                break
            carrying = True
            picked = True
        elif action == "drop":
            if not carrying:
                invalid_reason = "drop_without_printer"
                break
            if current not in table_neighbors:
                invalid_reason = "drop_not_adjacent_to_table"
                break
            carrying = False
            dropped = True
            path.append([list(current), carrying])
            break
        else:
            invalid_reason = f"unknown_action:{action}"
            break
        path.append([list(current), carrying])

    success = dropped and invalid_reason is None
    expected = expected_action_count(record)
    return {
        "parse_success": bool(actions) and not invalid_tokens,
        "success": success,
        "optimal_success": success and expected is not None and steps_executed == expected,
        "expected_action_count": expected,
        "action_count": len(actions),
        "steps_executed": steps_executed,
        "final_coord": list(current),
        "carrying": carrying,
        "picked": picked,
        "dropped": dropped,
        "path": path,
        "invalid_reason": invalid_reason,
        "invalid_action_tokens": invalid_tokens,
    }


def evaluate_actions(record: dict[str, Any], generated_text: str) -> tuple[list[str], str, dict[str, Any]]:
    task = record["_task"]
    actions, invalid_tokens, answer_text = parse_actions(generated_text, task)
    if task == "frozenlake":
        eval_result = eval_frozenlake(record, actions, invalid_tokens)
    elif task == "maze":
        eval_result = eval_maze(record, actions, invalid_tokens)
    elif task == "minibehaviour":
        eval_result = eval_minibehaviour(record, actions, invalid_tokens)
    else:
        raise RuntimeError(f"Unsupported task: {task}")
    return actions, answer_text, eval_result


def done_case_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    ids: set[str] = set()
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = item.get("case_id")
            if isinstance(case_id, str):
                ids.add(case_id)
    return ids


def base_result(record: dict[str, Any], image_path: Path | None, prompt_path: Path | None) -> dict[str, Any]:
    meta = record["meta"]
    result = {
        "case_id": record["_case_id"],
        "task": record["_task"],
        "split": record["_split"],
        "level": record_level(record),
        "dataset_line_index": record["_line_index"],
        "image": None if image_path is None else str(image_path),
        "prompt_path": None if prompt_path is None else str(prompt_path),
        "start_pos": meta.get("start_pos"),
        "target_pos": meta.get("target_pos"),
        "expected_action_count": expected_action_count(record),
    }
    for field in ("_source_bench", "_source_image", "_source_table", "_source_pure_text"):
        value = record.get(field)
        if value is not None:
            result[field.removeprefix("_")] = value
    return result


def evaluate_one(
    record: dict[str, Any],
    prompts: dict[str, tuple[Path, str]],
    args: argparse.Namespace,
    endpoints: list[Endpoint] | None,
    model: str | None,
) -> dict[str, Any]:
    image_path = render_record_image(record, args.image_cache, args.overwrite_images)
    prompt_path, prompt = prompts[record["_task"]]
    result = base_result(record, image_path, prompt_path)
    if args.store_prompt:
        result["prompt"] = prompt

    if args.render_only or args.dry_run:
        result["render_only"] = True
        return result

    assert endpoints is not None and model is not None
    endpoint_index = int(record.get("_eval_index", 0)) % len(endpoints)
    base_url, api_key = endpoints[endpoint_index]
    result["endpoint"] = base_url
    payload = build_payload(
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
    response = request_json(
        f"{base_url}/chat/completions",
        api_key,
        payload,
        args.timeout,
        args.use_system_proxy,
    )
    generated_text = extract_model_text(response)
    actions, answer_text, eval_result = evaluate_actions(record, generated_text)
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
    endpoints: list[Endpoint] | None,
    model: str | None,
) -> dict[str, Any]:
    image_path: Path | None = None
    prompt_path: Path | None = None
    try:
        prompt_path = prompts[record["_task"]][0]
        image_path = render_record_image(record, args.image_cache, args.overwrite_images)
        return evaluate_one(record, prompts, args, endpoints, model)
    except Exception as exc:
        result = base_result(record, image_path, prompt_path)
        result["error"] = str(exc)
        return result


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def empty_summary_stats() -> dict[str, Any]:
    return {
        "num_cases": 0,
        "num_errors": 0,
        "num_parse_success": 0,
        "num_success": 0,
        "num_optimal_success": 0,
        "num_illegal_movement": 0,
    }


def illegal_movement_field(task: str, eval_result: dict[str, Any]) -> str | None:
    invalid_reason = str(eval_result.get("invalid_reason") or "")
    if task == "frozenlake" and (eval_result.get("fell_into_hole") or invalid_reason == "fell_into_hole"):
        return "num_fell_into_hole"
    if task == "maze" and invalid_reason.startswith("wall_blocked"):
        return "num_crossed_wall"
    if task == "minibehaviour" and invalid_reason == "entered_table":
        return "num_entered_table"
    return None


def update_summary_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["num_cases"] += 1
    if row.get("error"):
        stats["num_errors"] += 1
        return
    task = str(row.get("task", "unknown"))
    eval_result = row.get("eval_result") or {}
    stats["num_parse_success"] += int(bool(eval_result.get("parse_success")))
    stats["num_success"] += int(bool(eval_result.get("success")))
    stats["num_optimal_success"] += int(bool(eval_result.get("optimal_success")))
    illegal_field = illegal_movement_field(task, eval_result)
    if illegal_field is not None:
        stats["num_illegal_movement"] += 1
        stats[illegal_field] = stats.get(illegal_field, 0) + 1


def finalize_summary_stats(stats: dict[str, Any]) -> dict[str, Any]:
    denom = max(1, stats["num_cases"] - stats["num_errors"])
    stats["parse_rate"] = stats["num_parse_success"] / denom
    stats["success_rate"] = stats["num_success"] / denom
    stats["optimal_success_rate"] = stats["num_optimal_success"] / denom
    stats["illegal_movement_rate"] = stats["num_illegal_movement"] / denom
    for count_key, rate_key in (
        ("num_fell_into_hole", "fell_into_hole_rate"),
        ("num_crossed_wall", "crossed_wall_rate"),
        ("num_entered_table", "entered_table_rate"),
    ):
        if count_key in stats:
            stats[rate_key] = stats[count_key] / denom
    stats["accuracy"] = stats["success_rate"]
    stats["optimal_accuracy"] = stats["optimal_success_rate"]
    return stats


def macro_average_stats(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {
            "num_groups": 0,
            "parse_rate": 0.0,
            "success_rate": 0.0,
            "optimal_success_rate": 0.0,
            "illegal_movement_rate": 0.0,
            "accuracy": 0.0,
            "optimal_accuracy": 0.0,
        }
    metrics = (
        "parse_rate",
        "success_rate",
        "optimal_success_rate",
        "illegal_movement_rate",
        "accuracy",
        "optimal_accuracy",
    )
    result: dict[str, Any] = {"num_groups": len(groups)}
    for metric in metrics:
        result[metric] = sum(float(item[metric]) for item in groups.values()) / len(groups)
    return result


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    by_task_level: dict[str, dict[str, dict[str, Any]]] = {}

    for row in rows:
        task = str(row.get("task", "unknown"))
        level = str(row.get("level", "unknown"))
        update_summary_stats(by_task.setdefault(task, empty_summary_stats()), row)
        task_levels = by_task_level.setdefault(task, {})
        update_summary_stats(task_levels.setdefault(level, empty_summary_stats()), row)

    total = empty_summary_stats()
    for stats in by_task.values():
        for key in total:
            total[key] += stats[key]

    by_task = {key: finalize_summary_stats(by_task[key]) for key in sorted(by_task)}
    by_task_level = {
        task: {
            level: finalize_summary_stats(level_stats)
            for level, level_stats in sorted(
                levels.items(),
                key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
            )
        }
        for task, levels in sorted(by_task_level.items())
    }

    task_macro_avg_by_level = {
        task: macro_average_stats(levels)
        for task, levels in by_task_level.items()
    }
    all_tasks_macro_avg = macro_average_stats(by_task)

    return {
        "total": finalize_summary_stats(total),
        "by_task": by_task,
        "by_task_level": by_task_level,
        "task_macro_avg_by_level": task_macro_avg_by_level,
        "all_tasks_macro_avg": all_tasks_macro_avg,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
    records = load_records(
        args.dataset_root,
        args.frozenlake_bench_root,
        args.tasks,
        args.split,
        levels,
        args.start_index,
        args.samples_per_task,
    )

    if args.resume:
        done = done_case_ids(args.output)
        records = [record for record in records if record["_case_id"] not in done]

    print(f"[select] cases={len(records)} tasks={','.join(args.tasks)} split={args.split}", flush=True)
    if not records:
        return

    if args.dry_run:
        for record in records[:20]:
            image_path = render_record_image(record, args.image_cache, args.overwrite_images)
            print(
                f"[dry-run] case={record['_case_id']} task={record['_task']} "
                f"level={record_level(record)} image={image_path}",
                flush=True,
            )
        if len(records) > 20:
            print(f"[dry-run] omitted {len(records) - 20} more case(s)", flush=True)
        return

    endpoints: list[Endpoint] | None = None
    model = None
    if not args.render_only:
        endpoints = resolve_endpoints(args)
        model = args.model or get_default_model(
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
                f"success={eval_result.get('success')} optimal={eval_result.get('optimal_success')} "
                f"error={row.get('error')}",
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
                    f"success={eval_result.get('success')} optimal={eval_result.get('optimal_success')} "
                    f"error={row.get('error')}",
                    flush=True,
                )

    summary_rows = read_jsonl(args.output)
    summary = summarize_rows(summary_rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote rows to {args.output}", flush=True)
    print(f"[done] wrote summary to {summary_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

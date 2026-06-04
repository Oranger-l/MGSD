#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_CONFIG = REPO_ROOT / "api_config_files/api_config_vllm.json"
DEFAULT_DATASET_JSON = REPO_ROOT / "LlamaFactory/data/vsp_tasks_perception_sft.json"
DEFAULT_MODEL = "qwen3vl-vsp-perception"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/vsp_perception_eval.jsonl"
TASK_ALIASES = {
    "frozenlake": "frozenlake_perception_all",
    "frozenlake_perception_all": "frozenlake_perception_all",
    "maze": "maze_perception_all",
    "maze_perception_all": "maze_perception_all",
    "minibehaviour": "minibehaviour_perception_all",
    "minibehavior": "minibehaviour_perception_all",
    "mini": "minibehaviour_perception_all",
    "minibehaviour_perception_all": "minibehaviour_perception_all",
}
CANONICAL_TASKS = (
    "frozenlake_perception_all",
    "maze_perception_all",
    "minibehaviour_perception_all",
)
TASK_DISPLAY = {
    "frozenlake_perception_all": "FrozenLake",
    "maze_perception_all": "Maze",
    "minibehaviour_perception_all": "MiniBehaviour",
}
DEFAULT_TASK_LEVELS = {
    "frozenlake_perception_all": (3, 4, 5, 6, 7, 8),
    "maze_perception_all": (3, 4, 5, 6),
    "minibehaviour_perception_all": (5, 6),
}
QUESTION_KEYS = {
    "frozenlake_perception_all": ["map_size", "player_position", "goal_position", "h_positions"],
    "maze_perception_all": ["map_size", "player_position", "goal_position", "open_directions", "player_legal_actions"],
    "minibehaviour_perception_all": [
        "map_size",
        "agent_position",
        "printer_position",
        "table_cells",
        "printer_adjacent_cells",
        "table_adjacent_cells",
        "movement_legal_actions",
    ],
}
TASK_PROMPTS = {
    "frozenlake_perception_all": "\n".join(
        [
            "<image>Answer the following FrozenLake perception questions:",
            "1. What is the map size of this FrozenLake grid?",
            "2. Where is the player on this FrozenLake grid?",
            "3. Where is the goal on this FrozenLake grid?",
            "4. Where are the H tiles on this FrozenLake grid?",
        ]
    ),
    "maze_perception_all": "\n".join(
        [
            "<image>Answer the following Maze perception questions:",
            "",
            "Image color guide:",
            "1. The yellow dot is the Player.",
            "2. The blue dot is the Goal.",
            "3. White corridors are traversable.",
            "4. Black boundaries are walls and cannot be crossed.",
            "",
            "Questions:",
            "1. What is the map size?",
            "2. Where is the player position?",
            "3. Where is the goal position?",
            "4. For each cell, which movement directions are open?",
            "5. What are the legal movement actions from the player position?",
        ]
    ),
    "minibehaviour_perception_all": "\n".join(
        [
            "<image>Answer the following MiniBehaviour perception questions:",
            "",
            "Image color guide:",
            "1. The red object is the Agent.",
            "2. The white marker is the Printer.",
            "3. The tan block is the Table.",
            "4. Black cells are free floor cells.",
            "",
            "Questions:",
            "1. What is the map size?",
            "2. Where is the Agent?",
            "3. Where is the Printer?",
            "4. Which cells are occupied by the Table?",
            "5. Which cells are adjacent to the Printer?",
            "6. Which cells are adjacent to the Table?",
            "7. Which movement actions are legal from the red Agent position?",
        ]
    ),
}
ACTION_RE = re.compile(r"\b(up|right|down|left|pick|drop)\b", flags=re.IGNORECASE)
SIZE_RE = re.compile(r"\b(\d+)\s*[xX]\s*(\d+)\b")
COORD_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
NUMBERED_RE = re.compile(r"(?ms)^\s*(\d+)[\.\)]\s*(.*?)(?=^\s*\d+[\.\)]\s*|\Z)")


Endpoint = tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VSP-Tasks perception SFT checkpoints through OpenAI-compatible vLLM endpoints."
    )
    parser.add_argument("image", type=Path, nargs="?", help="Optional single image to query.")
    parser.add_argument("--single-task", choices=sorted(TASK_ALIASES), default="frozenlake", help="Task prompt for single-image mode.")
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Served model name. Use empty string to query /models.")
    parser.add_argument("--base-urls", nargs="+", default=None, help="Explicit OpenAI-compatible /v1 endpoint URLs.")
    parser.add_argument("--replicas", type=int, default=1, help="Build localhost endpoints from --base-port when >1.")
    parser.add_argument("--base-port", type=int, default=None, help="First local vLLM port. Defaults to --api-config port.")
    parser.add_argument("--host", default="localhost", help="Host used with --replicas.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tasks", nargs="+", default=["all"], help="all, frozenlake, maze, minibehaviour.")
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=None,
        help="Optional shared level filter that overrides the task-specific defaults.",
    )
    parser.add_argument(
        "--frozenlake-levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_TASK_LEVELS["frozenlake_perception_all"]),
        help="FrozenLake levels to evaluate when --levels is not set.",
    )
    parser.add_argument(
        "--maze-levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_TASK_LEVELS["maze_perception_all"]),
        help="Maze levels to evaluate when --levels is not set.",
    )
    parser.add_argument(
        "--minibehaviour-levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_TASK_LEVELS["minibehaviour_perception_all"]),
        help="MiniBehaviour levels to evaluate when --levels is not set.",
    )
    parser.add_argument("--samples-per-level", type=int, default=1, help="0 means all records per selected task/level.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--use-system-proxy",
        action="store_true",
        help="Use HTTP_PROXY/HTTPS_PROXY. Local vLLM calls bypass proxies by default.",
    )
    return parser.parse_args()


def load_api_config(path: Path) -> tuple[str, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    api_key = config.get("api_key", "")
    if isinstance(api_key, list):
        api_key = api_key[0] if api_key else ""
    return config["base_url"].rstrip("/"), api_key


def endpoint_port(base_url: str) -> int:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.port is not None:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def resolve_endpoints(args: argparse.Namespace) -> list[Endpoint]:
    config_base_url, config_api_key = load_api_config(args.api_config)
    api_key = config_api_key if args.api_key is None else args.api_key
    if args.base_urls:
        return [(url.rstrip("/"), api_key) for url in args.base_urls]
    if args.replicas > 1:
        base_port = args.base_port if args.base_port is not None else endpoint_port(config_base_url)
        return [(f"http://{args.host}:{base_port + offset}/v1", api_key) for offset in range(args.replicas)]
    return [(config_base_url, api_key)]


def request_json(
    url: str,
    api_key: str,
    payload: dict[str, Any] | None,
    timeout: float,
    use_system_proxy: bool,
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


def get_default_model(endpoint: Endpoint, timeout: float, use_system_proxy: bool) -> str:
    base_url, api_key = endpoint
    data = request_json(f"{base_url}/models", api_key, None, timeout, use_system_proxy)
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models returned by {base_url}/models.")
    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError(f"Invalid /models response from {base_url}: {data}")
    return model_id


def image_to_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(model: str, image_path: Path, prompt: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def extract_answer(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response format: {response}") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    raise RuntimeError(f"Unexpected message content format: {content}")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    return data


def first_message(record: dict[str, Any], role: str) -> str:
    for message in record.get("messages", []):
        if message.get("role") == role:
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return ""


def record_image(record: dict[str, Any]) -> Path:
    images = record.get("images") or []
    if not images:
        raise RuntimeError(f"Record has no image: {record}")
    return Path(images[0])


def canonical_task(value: str) -> str:
    key = value.lower()
    if key not in TASK_ALIASES:
        raise RuntimeError(f"Unknown task {value!r}.")
    return TASK_ALIASES[key]


def resolve_tasks(values: list[str]) -> list[str]:
    if not values or any(value.lower() == "all" for value in values):
        return list(CANONICAL_TASKS)
    tasks = []
    for value in values:
        task = canonical_task(value)
        if task not in tasks:
            tasks.append(task)
    return tasks


def levels_for_task(args: argparse.Namespace, task: str) -> set[int]:
    if args.levels is not None:
        return set(args.levels)
    if task == "frozenlake_perception_all":
        return set(args.frozenlake_levels)
    if task == "maze_perception_all":
        return set(args.maze_levels)
    if task == "minibehaviour_perception_all":
        return set(args.minibehaviour_levels)
    raise RuntimeError(f"Unsupported task {task!r}.")


def select_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = set(resolve_tasks(args.tasks))
    task_levels = {task: levels_for_task(args, task) for task in tasks}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task = record.get("task")
        if task not in tasks:
            continue
        try:
            level = int(record["level"])
        except (KeyError, TypeError, ValueError):
            continue
        if level not in task_levels[task]:
            continue
        grouped[(task, level)].append(record)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (CANONICAL_TASKS.index(item[0]), item[1])):
        candidates = sorted(grouped[key], key=lambda item: (str(item.get("source_text", "")), str(record_image(item))))
        if args.shuffle:
            import random

            random.Random(args.seed + hash(key)).shuffle(candidates)
        if args.start_index:
            candidates = candidates[args.start_index :]
        if args.samples_per_level > 0:
            candidates = candidates[: args.samples_per_level]
        selected.extend(candidates)

    if not selected:
        raise RuntimeError("No records selected. Check --tasks/--levels/--samples-per-level.")
    return selected


def split_numbered_answers(text: str) -> dict[int, str]:
    return {int(match.group(1)): match.group(2).strip() for match in NUMBERED_RE.finditer(text.strip())}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def parse_map_size(text: str) -> str | None:
    match = SIZE_RE.search(text)
    if match:
        return f"{int(match.group(1))}x{int(match.group(2))}"
    bare = re.search(r"\b(\d+)\b", text)
    if bare:
        size = int(bare.group(1))
        return f"{size}x{size}"
    return None


def parse_first_coord(text: str) -> tuple[int, int] | None:
    match = COORD_RE.search(text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_coord_set(text: str) -> set[tuple[int, int]]:
    return {(int(row), int(col)) for row, col in COORD_RE.findall(text)}


def parse_actions(text: str, include_pick_drop: bool = False) -> list[str]:
    allowed = {"up", "right", "down", "left", "pick", "drop"} if include_pick_drop else {"up", "right", "down", "left"}
    actions = []
    for match in ACTION_RE.finditer(text):
        action = match.group(1).lower()
        if action in allowed and action not in actions:
            actions.append(action)
    return actions


def parse_open_direction_map(text: str) -> dict[tuple[int, int], set[str]]:
    parsed: dict[tuple[int, int], set[str]] = {}
    for match in re.finditer(r"\((\d+)\s*,\s*(\d+)\)\s*:\s*([^\n]+)", text):
        coord = (int(match.group(1)), int(match.group(2)))
        parsed[coord] = set(parse_actions(match.group(3)))
    return parsed


def parse_yes_no(text: str) -> bool | None:
    lowered = normalize_text(text)
    if re.search(r"\bno\b", lowered) or "not currently legal" in lowered or "not legal" in lowered:
        return False
    if re.search(r"\byes\b", lowered) or "currently legal" in lowered:
        return True
    return None


def compare_coord(expected: str, answer: str) -> bool:
    return parse_first_coord(expected) == parse_first_coord(answer)


def compare_coord_set(expected: str, answer: str) -> bool:
    return parse_coord_set(expected) == parse_coord_set(answer)


def compare_actions(expected: str, answer: str) -> bool:
    return set(parse_actions(expected)) == set(parse_actions(answer))


def compare_open_directions(expected: str, answer: str) -> bool:
    return parse_open_direction_map(expected) == parse_open_direction_map(answer)


def compare_bool(expected: str, answer: str) -> bool:
    return parse_yes_no(expected) == parse_yes_no(answer)


def score_sections(task: str, expected: str, answer: str) -> dict[str, bool]:
    expected_sections = split_numbered_answers(expected)
    answer_sections = split_numbered_answers(answer)
    if not answer_sections:
        return {key: False for key in QUESTION_KEYS[task]}

    checks = []
    if task == "frozenlake_perception_all":
        checks = [
            ("map_size", lambda e, a: parse_map_size(e) == parse_map_size(a)),
            ("player_position", compare_coord),
            ("goal_position", compare_coord),
            ("h_positions", compare_coord_set),
        ]
    elif task == "maze_perception_all":
        checks = [
            ("map_size", lambda e, a: parse_map_size(e) == parse_map_size(a)),
            ("player_position", compare_coord),
            ("goal_position", compare_coord),
            ("open_directions", compare_open_directions),
            ("player_legal_actions", compare_actions),
        ]
    elif task == "minibehaviour_perception_all":
        checks = [
            ("map_size", lambda e, a: parse_map_size(e) == parse_map_size(a)),
            ("agent_position", compare_coord),
            ("printer_position", compare_coord),
            ("table_cells", compare_coord_set),
            ("printer_adjacent_cells", compare_coord_set),
            ("table_adjacent_cells", compare_coord_set),
            ("movement_legal_actions", compare_actions),
        ]
    else:
        raise RuntimeError(f"Unsupported task: {task}")

    scores: dict[str, bool] = {}
    for index, (key, check) in enumerate(checks, start=1):
        scores[key] = check(expected_sections.get(index, ""), answer_sections.get(index, ""))
    return scores


def build_case(record: dict[str, Any], index: int) -> dict[str, Any]:
    task = record["task"]
    source_text = Path(record.get("source_text", ""))
    source_id = source_text.stem if source_text.name else f"{index:04d}"
    return {
        "case_id": f"{task}_level{record.get('level')}_{source_id}",
        "task": task,
        "level": record.get("level"),
        "image": record_image(record),
        "source_text": str(source_text) if source_text else None,
        "prompt": TASK_PROMPTS.get(task) or first_message(record, "user"),
        "expected": first_message(record, "assistant"),
        "record_index": index,
    }


def run_single(args: argparse.Namespace) -> None:
    endpoints = resolve_endpoints(args)
    model = args.model or get_default_model(endpoints[0], args.timeout, args.use_system_proxy)
    task = canonical_task(args.single_task)
    prompt = TASK_PROMPTS[task]
    if args.dry_run:
        print(prompt)
        return
    base_url, api_key = endpoints[0]
    response = request_json(
        f"{base_url}/chat/completions",
        api_key,
        build_payload(model, args.image, prompt, args.temperature, args.max_tokens),
        args.timeout,
        args.use_system_proxy,
    )
    if args.print_json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(extract_answer(response))


def completed_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = row.get("case_id")
            if case_id:
                ids.add(case_id)
    return ids


def query_case(
    case: dict[str, Any],
    endpoint: Endpoint,
    model: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_url, api_key = endpoint
    response = request_json(
        f"{base_url}/chat/completions",
        api_key,
        build_payload(model, case["image"], case["prompt"], args.temperature, args.max_tokens),
        args.timeout,
        args.use_system_proxy,
    )
    answer = extract_answer(response)
    question_matches = score_sections(case["task"], case["expected"], answer)
    row = {
        "case_id": case["case_id"],
        "task": case["task"],
        "task_display": TASK_DISPLAY[case["task"]],
        "level": case["level"],
        "image": str(case["image"]),
        "source_text": case["source_text"],
        "endpoint": base_url,
        "prompt": case["prompt"],
        "expected": case["expected"],
        "answer": answer,
        "question_exact_match": question_matches,
        "all_exact_match": all(question_matches.values()),
    }
    if args.print_json:
        row["response"] = response
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"total": len(rows), "tasks": {}}
    for task in CANONICAL_TASKS:
        task_rows = [row for row in rows if row.get("task") == task]
        if not task_rows:
            continue
        question_counts = Counter()
        level_stats: dict[str, dict[str, Any]] = {}
        for row in task_rows:
            for key, value in row["question_exact_match"].items():
                question_counts[key] += int(value)
            level = str(row.get("level"))
            level_summary = level_stats.setdefault(
                level,
                {
                    "total": 0,
                    "all_exact_match": 0,
                    "questions": Counter(),
                },
            )
            level_summary["total"] += 1
            level_summary["all_exact_match"] += int(row["all_exact_match"])
            for key, value in row["question_exact_match"].items():
                level_summary["questions"][key] += int(value)
        levels = {
            level: {
                "total": value["total"],
                "all_exact_match": value["all_exact_match"],
                "questions": dict(value["questions"]),
            }
            for level, value in sorted(level_stats.items(), key=lambda item: int(item[0]))
        }
        summary["tasks"][task] = {
            "total": len(task_rows),
            "all_exact_match": sum(int(row["all_exact_match"]) for row in task_rows),
            "questions": dict(question_counts),
            "levels": levels,
        }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"Finished {summary['total']} samples.")
    for task, task_summary in summary["tasks"].items():
        total = task_summary["total"]
        all_count = task_summary["all_exact_match"]
        print(f"{TASK_DISPLAY[task]} all_exact_match={all_count}/{total} ({all_count / total:.2%})")
        for key in QUESTION_KEYS[task]:
            count = task_summary["questions"].get(key, 0)
            print(f"  {key}={count}/{total} ({count / total:.2%})")
        for level, level_summary in task_summary.get("levels", {}).items():
            level_total = level_summary["total"]
            level_all_count = level_summary["all_exact_match"]
            print(f"  level{level} all_exact_match={level_all_count}/{level_total} ({level_all_count / level_total:.2%})")


def run_batch(args: argparse.Namespace) -> None:
    records = load_dataset(args.dataset_json)
    cases = [build_case(record, index) for index, record in enumerate(select_records(args, records))]
    selected_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for case in cases:
        selected_counts[case["task"]][int(case["level"])] += 1
    print(f"[select] dataset={args.dataset_json} cases={len(cases)} tasks={','.join(resolve_tasks(args.tasks))}")
    for task in CANONICAL_TASKS:
        if task not in selected_counts:
            continue
        levels = " ".join(f"level{level}:{count}" for level, count in sorted(selected_counts[task].items()))
        print(f"[select] {TASK_DISPLAY[task]} {levels}")
    if args.resume:
        done = completed_case_ids(args.output)
        cases = [case for case in cases if case["case_id"] not in done]
    if args.dry_run:
        for case in cases:
            print(f"{case['case_id']} task={case['task']} level={case['level']} image={case['image']}")
            print(case["prompt"])
            print(case["expected"])
            print("-" * 80)
        print(f"Selected {len(cases)} cases.")
        return

    endpoints = resolve_endpoints(args)
    model = args.model or get_default_model(endpoints[0], args.timeout, args.use_system_proxy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    rows: list[dict[str, Any]] = []
    with args.output.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_case = {}
            for index, case in enumerate(cases):
                endpoint = endpoints[index % len(endpoints)]
                future = executor.submit(query_case, case, endpoint, model, args)
                future_to_case[future] = case

            for done_count, future in enumerate(as_completed(future_to_case), start=1):
                case = future_to_case[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "case_id": case["case_id"],
                        "task": case["task"],
                        "level": case["level"],
                        "image": str(case["image"]),
                        "source_text": case["source_text"],
                        "error": str(exc),
                        "all_exact_match": False,
                        "question_exact_match": {key: False for key in QUESTION_KEYS[case["task"]]},
                    }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{done_count}/{len(cases)}] {row['case_id']} "
                    f"all_exact={row.get('all_exact_match')} error={row.get('error')}"
                )

    summary = summarize(rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"Saved results to {args.output}")
    print(f"Saved summary to {summary_path}")


def main() -> None:
    args = parse_args()
    if args.image is not None:
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

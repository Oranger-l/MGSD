from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_CONFIG = REPO_ROOT / "api_config_files/api_config_vllm.json"
DEFAULT_DATASET_JSON = REPO_ROOT / "LlamaFactory/data/vsp_tasks_perception_sft.json"
DEFAULT_LEVELS = [3, 4, 5, 6, 7, 8]
DEFAULT_TASK = "frozenlake_perception_all"
QUESTION_KEYS = {
    1: "map_size",
    2: "player_position",
    3: "goal_position",
    4: "h_positions",
}
QUESTION_TEXTS = {
    "map_size": "What is the map size of this FrozenLake grid?",
    "player_position": "Where is the player on this FrozenLake grid?",
    "goal_position": "Where is the goal on this FrozenLake grid?",
    "h_positions": "Where are the H tiles on this FrozenLake grid?",
}
QUESTION_ALIASES = {
    "1": "map_size",
    "map": "map_size",
    "map_size": "map_size",
    "size": "map_size",
    "2": "player_position",
    "player": "player_position",
    "start": "player_position",
    "player_position": "player_position",
    "start_position": "player_position",
    "3": "goal_position",
    "goal": "goal_position",
    "goal_position": "goal_position",
    "4": "h_positions",
    "h": "h_positions",
    "hole": "h_positions",
    "holes": "h_positions",
    "h_positions": "h_positions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask a vLLM-served VLM FrozenLake perception questions. "
            "If no image is provided, sample records from the SFT JSON across all levels."
        )
    )
    parser.add_argument("image", type=Path, nargs="?", help="Optional single FrozenLake image to query.")
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG, help="Path to API config JSON.")
    parser.add_argument("--model", default=None, help="Served model name. If omitted, query /models first.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Question to ask the model. In batch mode, defaults to each record's training prompt.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--print-json", action="store_true", help="Print/store the raw response JSON.")
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON, help="SFT JSON used for batch tests.")
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS, help="Levels to test in batch mode.")
    parser.add_argument("--samples-per-level", type=int, default=1, help="Number of images to test per level.")
    parser.add_argument(
        "--questions",
        nargs="+",
        default=["all"],
        help=(
            "Questions to ask/evaluate. Use all, map_size, player_position, goal_position, h_positions, "
            "or aliases like 1, 2, 3, 4, map, player, goal, h."
        ),
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task name to select from the SFT JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL path to save batch results.")
    parser.add_argument("--dry-run", action="store_true", help="Only print selected samples and expected answers.")
    return parser.parse_args()


def load_api_config(path: Path) -> tuple[str, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    base_url = config["base_url"].rstrip("/")
    api_key = config.get("api_key", "")
    if isinstance(api_key, list):
        api_key = api_key[0] if api_key else ""
    return base_url, api_key


def request_json(url: str, api_key: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc}") from exc


def get_default_model(base_url: str, api_key: str, timeout: float) -> str:
    data = request_json(f"{base_url}/models", api_key, None, timeout)
    models = data.get("data") or []
    if not models:
        raise RuntimeError("No models returned by /models. Please pass --model explicitly.")
    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError(f"Invalid /models response: {data}")
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
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(texts).strip()
    raise RuntimeError(f"Unexpected message content format: {content}")


def load_dataset_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON list in {path}")
    return data


def first_message_content(record: dict[str, Any], role: str) -> str:
    for message in record.get("messages", []):
        if message.get("role") == role:
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                return "\n".join(parts)
    return ""


def record_image(record: dict[str, Any]) -> Path:
    images = record.get("images") or []
    if not images:
        raise RuntimeError(f"Record has no images field: {record}")
    return Path(images[0])


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def extract_map_size(text: str) -> str | None:
    match = re.search(r"\b(\d+)\s*[xX]\s*(\d+)\b", text)
    if match is not None:
        return f"{int(match.group(1))}x{int(match.group(2))}"

    bare_number = re.fullmatch(r"\s*(\d+)\s*\.?\s*", text)
    if bare_number is not None:
        size = int(bare_number.group(1))
        return f"{size}x{size}"

    return None


def question_exact_match(key: str, expected: str, answer: str) -> bool:
    if key == "map_size":
        expected_size = extract_map_size(expected)
        answer_size = extract_map_size(answer)
        if expected_size is not None and answer_size is not None:
            return expected_size == answer_size

    return normalize_answer(answer) == normalize_answer(expected)


def resolve_questions(values: list[str]) -> list[str]:
    if not values or any(value.lower() == "all" for value in values):
        return list(QUESTION_KEYS.values())

    requested = set()
    for value in values:
        key = QUESTION_ALIASES.get(value.lower())
        if key is None:
            valid = ", ".join(["all", *QUESTION_ALIASES])
            raise RuntimeError(f"Unknown question {value!r}. Valid values: {valid}")
        requested.add(key)

    return [key for key in QUESTION_KEYS.values() if key in requested]


def build_question_prompt(selected_keys: list[str]) -> str:
    lines = ["<image>Answer the following FrozenLake perception questions:"]
    for number, key in enumerate(selected_keys, start=1):
        lines.append(f"{number}. {QUESTION_TEXTS[key]}")
    return "\n".join(lines)


def split_numbered_answers(text: str, selected_keys: list[str] | None = None) -> dict[str, str]:
    matches = list(re.finditer(r"(?ms)^\s*([1-4])[\.\)]\s*(.*?)(?=^\s*[1-4][\.\)]\s*|\Z)", text.strip()))
    parsed: dict[str, str] = {}
    for match in matches:
        number = int(match.group(1))
        if selected_keys is not None:
            if number > len(selected_keys):
                continue
            key = selected_keys[number - 1]
        else:
            key = QUESTION_KEYS[number]
        parsed[key] = match.group(2).strip()
    return parsed


def build_expected_answer(record: dict[str, Any], selected_keys: list[str]) -> str:
    expected_parts = split_numbered_answers(first_message_content(record, "assistant"))
    lines = []
    for number, key in enumerate(selected_keys, start=1):
        lines.append(f"{number}. {expected_parts.get(key, '')}")
    return "\n".join(lines)


def score_question_matches(expected: str, answer: str, selected_keys: list[str]) -> dict[str, bool]:
    expected_parts = split_numbered_answers(expected, selected_keys)
    answer_parts = split_numbered_answers(answer, selected_keys)
    if len(selected_keys) == 1 and not answer_parts:
        key = selected_keys[0]
        return {key: question_exact_match(key, expected_parts.get(key, expected), answer)}

    return {
        key: question_exact_match(key, expected_parts.get(key, ""), answer_parts.get(key, ""))
        for key in selected_keys
    }


def select_records(
    records: list[dict[str, Any]],
    levels: list[int],
    samples_per_level: int,
    task: str,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if task and record.get("task") != task:
            continue
        try:
            level = int(record["level"])
        except (KeyError, TypeError, ValueError):
            continue
        if level in levels:
            grouped[level].append(record)

    selected: list[dict[str, Any]] = []
    for level in levels:
        candidates = sorted(
            grouped.get(level, []),
            key=lambda item: (str(item.get("source_text", "")), str(record_image(item))),
        )
        if not candidates:
            raise RuntimeError(f"No records found for level={level} task={task!r} in the dataset JSON.")
        if len(candidates) < samples_per_level:
            raise RuntimeError(
                f"Only found {len(candidates)} records for level={level} task={task!r}; "
                f"requested {samples_per_level}."
            )
        selected.extend(candidates[:samples_per_level])
    return selected


def print_result(
    index: int,
    total: int,
    record: dict[str, Any],
    prompt: str,
    expected: str,
    answer: str | None,
    question_matches: dict[str, bool] | None,
    selected_keys: list[str],
) -> None:
    level = record.get("level", "?")
    image = record_image(record)
    source_text = record.get("source_text", "")
    print("=" * 80)
    print(f"[{index}/{total}] level={level} image={image}")
    if source_text:
        print(f"source_text={source_text}")
    print("\nPrompt:")
    print(prompt)
    print("\nExpected:")
    print(expected)
    if answer is not None:
        print("\nModel answer (full assistant content):")
        print(answer)
        print("\nQuestion exact match:")
        for key in selected_keys:
            print(f"  {key}: {question_matches[key] if question_matches is not None else None}")


def run_single(args: argparse.Namespace) -> None:
    base_url, api_key = load_api_config(args.api_config)
    model = args.model or get_default_model(base_url, api_key, args.timeout)
    selected_keys = resolve_questions(args.questions)
    prompt = args.prompt or build_question_prompt(selected_keys)

    if args.dry_run:
        print(f"image={args.image}")
        print("\nPrompt:")
        print(prompt)
        return

    payload = build_payload(model, args.image, prompt, args.temperature, args.max_tokens)
    response = request_json(f"{base_url}/chat/completions", api_key, payload, args.timeout)

    if args.print_json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return

    answer = extract_answer(response)
    print("Model answer (full assistant content):")
    print(answer)


def run_batch(args: argparse.Namespace) -> None:
    if args.samples_per_level < 1:
        raise RuntimeError("--samples-per-level must be at least 1.")

    selected_keys = resolve_questions(args.questions)
    records = load_dataset_records(args.dataset_json)
    selected = select_records(records, args.levels, args.samples_per_level, args.task)
    total = len(selected)

    if args.dry_run:
        for index, record in enumerate(selected, start=1):
            prompt = args.prompt or build_question_prompt(selected_keys)
            expected = build_expected_answer(record, selected_keys)
            print_result(index, total, record, prompt, expected, answer=None, question_matches=None, selected_keys=selected_keys)
        return

    base_url, api_key = load_api_config(args.api_config)
    model = args.model or get_default_model(base_url, api_key, args.timeout)

    results: list[dict[str, Any]] = []
    exact_matches_by_question = {key: 0 for key in selected_keys}
    all_exact_matches = 0
    for index, record in enumerate(selected, start=1):
        prompt = args.prompt or build_question_prompt(selected_keys)
        expected = build_expected_answer(record, selected_keys)
        response = request_json(
            f"{base_url}/chat/completions",
            api_key,
            build_payload(model, record_image(record), prompt, args.temperature, args.max_tokens),
            args.timeout,
        )
        answer = extract_answer(response)
        question_matches = score_question_matches(expected, answer, selected_keys)
        all_exact_match = all(question_matches.values())
        all_exact_matches += int(all_exact_match)
        for key, matched in question_matches.items():
            exact_matches_by_question[key] += int(matched)

        result = {
            "level": record.get("level"),
            "task": record.get("task"),
            "image": str(record_image(record)),
            "source_text": record.get("source_text"),
            "questions": selected_keys,
            "prompt": prompt,
            "expected": expected,
            "answer": answer,
            "question_exact_match": question_matches,
            "all_exact_match": all_exact_match,
        }
        if args.print_json:
            result["response"] = response
        results.append(result)
        print_result(index, total, record, prompt, expected, answer, question_matches, selected_keys)

    print("=" * 80)
    print(f"Finished {total} samples.")
    for key in selected_keys:
        count = exact_matches_by_question[key]
        print(f"{key}_exact_match={count}/{total} ({count / total:.2%})")
    print(f"all_questions_exact_match={all_exact_matches}/{total} ({all_exact_matches / total:.2%})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Saved results to {args.output}")


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

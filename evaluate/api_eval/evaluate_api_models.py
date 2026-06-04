#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CKPT_EVAL_PATH = SCRIPT_DIR.parent / "ckpt_eval/evaluate_visualplanning_ckpt.py"

spec = importlib.util.spec_from_file_location("ckpt_eval_pipeline", CKPT_EVAL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load ckpt eval pipeline: {CKPT_EVAL_PATH}")
ckpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ckpt)


DEFAULT_API_KEY_FILE = REPO_ROOT / "api_config_files" / "api_config_openai.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_API_PROMPT_DIR = SCRIPT_DIR / "prompts"
DEFAULT_API_FROZENLAKE_PROMPT = DEFAULT_API_PROMPT_DIR / "opcd_student_Frozenlake_direct.txt"
DEFAULT_REQUESTED_MODELS = (
    "gpt-4o",
    "gpt-5",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
)
OFFICIAL_MODEL_IDS = {
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt5": "gpt-5",
    "gpt-5": "gpt-5",
    "gemini2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini3-flash": "gemini-3-flash-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "claude4.5-haiku": "claude-haiku-4-5-20251001",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
    "qwen3-vl-30b-thinking": "qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-30b-a3b-thinking": "qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-32b-thinking": "qwen3-vl-32b-thinking",
    "qwen3-vl-235b-a22b": "qwen3-vl-235b-a22b-instruct",
    "qwen3-vl-235b-a22b-instruct": "qwen3-vl-235b-a22b-instruct",
    "qwen3-vl-235b-a22b-thinking": "qwen3-vl-235b-a22b-thinking",
    "kimi-k2.5": "kimi-k2.5",
    "kimi-k2.5-thinking": "kimi-k2.5-thinking",
}

Endpoint = tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate closed-source VLM APIs with the same data and metrics as ckpt_eval."
    )
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible /v1 base URL. Overrides the key file.")
    parser.add_argument("--api-key", default=None, help="API key. Overrides the key file.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_REQUESTED_MODELS))
    parser.add_argument("--list-models", action="store_true", help="List models returned by /models and exit.")
    parser.add_argument("--resolve-models", action="store_true", help="Resolve requested aliases, print them, and exit.")
    parser.add_argument(
        "--allow-unlisted",
        action="store_true",
        help="Run models even when they are absent from the provider /models response.",
    )
    parser.add_argument(
        "--no-fetch-model-list",
        action="store_true",
        help="Do not call /models. Implies --allow-unlisted for requested model ids.",
    )
    parser.add_argument("--dataset-root", type=Path, default=ckpt.DEFAULT_DATASET_ROOT)
    parser.add_argument("--frozenlake-bench-root", type=Path, default=ckpt.DEFAULT_FROZENLAKE_BENCH_ROOT)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--tasks", nargs="+", choices=ckpt.TASKS, default=list(ckpt.TASKS))
    parser.add_argument("--levels", nargs="+", type=int, default=None)
    parser.add_argument("--samples-per-task", type=int, default=1, help="0 means all selected samples.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--image-cache", type=Path, default=SCRIPT_DIR / "rendered_images")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single output JSONL path. Only valid when exactly one model is resolved.",
    )
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_API_PROMPT_DIR)
    parser.add_argument("--frozenlake-prompt", type=Path, default=DEFAULT_API_FROZENLAKE_PROMPT)
    parser.add_argument("--prompt-image-1", type=Path, default=ckpt.DEFAULT_PROMPT_IMAGES[0])
    parser.add_argument("--prompt-image-2", type=Path, default=ckpt.DEFAULT_PROMPT_IMAGES[1])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-p", type=float, default=None, help="Omit by default for Claude compatibility.")
    parser.add_argument(
        "--token-field",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_tokens",
        help="Token budget field used in the OpenAI-compatible chat/completions payload.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=32, help="Concurrent requests per model.")
    parser.add_argument("--model-workers", type=int, default=3, help="Concurrent model evaluations.")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing JSONL, skip completed no-error cases, and retry missing/error cases.",
    )
    parser.add_argument("--store-prompt", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-system-proxy", action="store_true")
    return parser.parse_args()


def first_config_value(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value) if value is not None else ""


def load_api_key_file(path: Path) -> tuple[str, str]:
    api_key = ""
    base_url = ""
    if path.exists():
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        try:
            config = json.loads(raw_text)
        except json.JSONDecodeError:
            config = None

        if isinstance(config, dict):
            api_key = first_config_value(
                config.get("api_key")
                or config.get("key")
                or config.get("token")
                or config.get("openai_api_key")
                or config.get("azure_openai_api_key")
            )
            base_url = first_config_value(
                config.get("base_url")
                or config.get("api_base")
                or config.get("openai_base_url")
                or config.get("url")
                or config.get("azure_openai_endpoint")
            ).rstrip("/")
        else:
            for raw_line in raw_text.splitlines():
                line = raw_line.strip().replace("＝", "=")
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().lower().replace("-", "_").replace(" ", "")
                    value = value.strip().strip("\"'")
                    if key in {"apikey", "api_key", "key", "token"}:
                        api_key = value
                    elif key in {"baseurl", "base_url", "api_base", "openai_base_url", "url"}:
                        base_url = value.rstrip("/")
                elif not api_key:
                    api_key = line

    api_key = os.environ.get("OPENAI_API_KEY", api_key)
    base_url = os.environ.get("OPENAI_BASE_URL", base_url).rstrip("/")

    if not api_key:
        raise RuntimeError(f"No API key found in {path}")
    if not base_url:
        raise RuntimeError(f"No baseurl found in {path}")
    return base_url, api_key


def resolve_auth(args: argparse.Namespace) -> Endpoint:
    file_base_url, file_api_key = load_api_key_file(args.api_key_file)
    base_url = (args.base_url or file_base_url).rstrip("/")
    api_key = args.api_key if args.api_key is not None else file_api_key
    return base_url, api_key


def request_json(
    url: str,
    api_key: str,
    payload: dict[str, Any] | None,
    timeout: float,
    use_system_proxy: bool,
    retries: int = 0,
    retry_sleep: float = 0.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if payload is None else "POST")
    opener = None if use_system_proxy else urllib.request.build_opener(urllib.request.ProxyHandler({}))
    open_fn = urllib.request.urlopen if opener is None else opener.open

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with open_fn(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code} from {url}: {body}")
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"Failed to connect to {url}: {exc}")
        except TimeoutError as exc:
            last_error = RuntimeError(f"Timed out connecting to {url}: {exc}")
        if attempt < retries:
            time.sleep(retry_sleep)
    assert last_error is not None
    raise last_error


def fetch_model_ids(endpoint: Endpoint, args: argparse.Namespace) -> list[str]:
    base_url, api_key = endpoint
    data = request_json(
        f"{base_url}/models",
        api_key,
        None,
        args.timeout,
        args.use_system_proxy,
        retries=1,
        retry_sleep=args.retry_sleep,
    )
    ids: list[str] = []
    items = data.get("data", data if isinstance(data, list) else [])
    for item in items:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("name") or item.get("model")
        else:
            model_id = str(item)
        if model_id:
            ids.append(str(model_id))
    return ids


def canonical_model_name(value: str) -> str:
    return OFFICIAL_MODEL_IDS.get(value, value)


def resolve_requested_models(
    requested: list[str],
    available: list[str] | None,
    allow_unlisted: bool,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    available_set = set(available or [])
    resolved: list[str] = []
    aliases: dict[str, str] = {}
    missing: dict[str, str] = {}

    for raw_name in requested:
        model_id = canonical_model_name(raw_name)
        if available is None or model_id in available_set or raw_name in available_set:
            chosen = raw_name if raw_name in available_set else model_id
        elif allow_unlisted:
            chosen = model_id
        else:
            missing[raw_name] = model_id
            continue
        if chosen not in resolved:
            resolved.append(chosen)
        aliases[raw_name] = chosen
    return resolved, aliases, missing


def safe_model_filename(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_")


def output_path_for_model(args: argparse.Namespace, model: str, model_count: int) -> Path:
    if args.output is not None:
        if model_count != 1:
            raise RuntimeError("--output can only be used when exactly one model is resolved.")
        return args.output
    return args.output_dir / f"api_eval_{safe_model_filename(model)}.jsonl"


def latest_rows_by_case_id(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = row.get("case_id")
            if isinstance(case_id, str):
                latest[case_id] = row
    return latest


def is_completed_row(row: dict[str, Any]) -> bool:
    if row.get("error") or not isinstance(row.get("eval_result"), dict):
        return False
    return bool(str(row.get("generated_text") or "").strip())


def completed_case_ids(path: Path) -> set[str]:
    return {
        case_id
        for case_id, row in latest_rows_by_case_id(path).items()
        if is_completed_row(row)
    }


def summarize_output(path: Path) -> dict[str, Any]:
    rows = []
    for row in latest_rows_by_case_id(path).values():
        if (
            not row.get("error")
            and isinstance(row.get("eval_result"), dict)
            and not str(row.get("generated_text") or "").strip()
        ):
            row = dict(row)
            row["error"] = "empty_assistant_content"
        rows.append(row)
    summary = ckpt.summarize_rows(rows)
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def response_diagnostic(response: dict[str, Any]) -> str:
    choice = {}
    try:
        choice = response.get("choices", [{}])[0]
    except (IndexError, TypeError, AttributeError):
        choice = {}
    usage = response.get("usage") if isinstance(response, dict) else None
    return f"finish_reason={choice.get('finish_reason')!r} usage={usage!r}"


def build_api_payload(
    model: str,
    prompt: str,
    image_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": ckpt.build_content_from_prompt(
                    prompt,
                    image_path,
                    args.prompt_image_1,
                    args.prompt_image_2,
                ),
            }
        ],
        args.token_field: args.max_tokens,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    return payload


def evaluate_one(
    record: dict[str, Any],
    model: str,
    endpoint: Endpoint,
    prompts: dict[str, tuple[Path, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_path = ckpt.render_record_image(record, args.image_cache, args.overwrite_images)
    prompt_path, prompt = prompts[record["_task"]]
    result = ckpt.base_result(record, image_path, prompt_path)
    result["model"] = model
    if args.store_prompt:
        result["prompt"] = prompt

    if args.render_only or args.dry_run:
        result["render_only"] = True
        return result

    base_url, api_key = endpoint
    result["endpoint"] = base_url
    payload = build_api_payload(model, prompt, image_path, args)
    response: dict[str, Any] | None = None
    generated_text = ""
    for attempt in range(args.retries + 1):
        response = request_json(
            f"{base_url}/chat/completions",
            api_key,
            payload,
            args.timeout,
            args.use_system_proxy,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        generated_text = ckpt.extract_model_text(response)
        if generated_text:
            break
        if attempt < args.retries:
            time.sleep(args.retry_sleep)
    if not generated_text:
        assert response is not None
        raise RuntimeError(f"Empty assistant content from API response ({response_diagnostic(response)})")
    actions, answer_text, eval_result = ckpt.evaluate_actions(record, generated_text)
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
    model: str,
    endpoint: Endpoint,
    prompts: dict[str, tuple[Path, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_path: Path | None = None
    prompt_path: Path | None = None
    try:
        prompt_path = prompts[record["_task"]][0]
        image_path = ckpt.render_record_image(record, args.image_cache, args.overwrite_images)
        return evaluate_one(record, model, endpoint, prompts, args)
    except Exception as exc:
        result = ckpt.base_result(record, image_path, prompt_path)
        result["model"] = model
        result["error"] = str(exc)
        return result


def evaluate_model(
    model: str,
    records: list[dict[str, Any]],
    endpoint: Endpoint,
    prompts: dict[str, tuple[Path, str]],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_records = [dict(record) for record in records]

    if args.resume:
        done = completed_case_ids(output_path)
        model_records = [record for record in model_records if record["_case_id"] not in done]
        if output_path.exists():
            latest = latest_rows_by_case_id(output_path)
            error_count = sum(1 for row in latest.values() if row.get("error"))
            print(
                f"[resume] output={output_path} completed={len(done)} "
                f"retry_or_missing={len(model_records)} "
                f"existing_error_latest={error_count}",
                flush=True,
            )
    elif output_path.exists():
        output_path.unlink()

    for eval_index, record in enumerate(model_records):
        record["_eval_index"] = eval_index

    print(f"[model] {model} cases={len(model_records)} output={output_path}", flush=True)
    if not model_records:
        if output_path.exists():
            summary = summarize_output(output_path)
            summary_path = output_path.with_suffix(".summary.json")
            print(f"[done] model={model} no remaining cases; reused {output_path}", flush=True)
            print(f"[done] model={model} summary={summary_path}", flush=True)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for index, record in enumerate(model_records, start=1):
            row = evaluate_one_safe(record, model, endpoint, prompts, args)
            rows.append(row)
            ckpt.write_jsonl_row(output_path, row)
            eval_result = row.get("eval_result") or {}
            print(
                f"[{index}/{len(model_records)}] model={model} case={row['case_id']} "
                f"task={row['task']} success={eval_result.get('success')} "
                f"optimal={eval_result.get('optimal_success')} error={row.get('error')}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(evaluate_one_safe, record, model, endpoint, prompts, args)
                for record in model_records
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                ckpt.write_jsonl_row(output_path, row)
                eval_result = row.get("eval_result") or {}
                print(
                    f"[{index}/{len(model_records)}] model={model} case={row['case_id']} "
                    f"task={row['task']} success={eval_result.get('success')} "
                    f"optimal={eval_result.get('optimal_success')} error={row.get('error')}",
                    flush=True,
                )

    summary = summarize_output(output_path)
    summary_path = output_path.with_suffix(".summary.json")
    print(f"[done] model={model} rows={output_path}", flush=True)
    print(f"[done] model={model} summary={summary_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.samples_per_task < 0:
        raise RuntimeError("--samples-per-task must be >= 0.")
    if args.start_index < 0:
        raise RuntimeError("--start-index must be >= 0.")
    if args.workers < 1:
        raise RuntimeError("--workers must be >= 1.")
    if args.model_workers < 1:
        raise RuntimeError("--model-workers must be >= 1.")

    endpoint = resolve_auth(args)
    available: list[str] | None = None
    if not args.no_fetch_model_list:
        available = fetch_model_ids(endpoint, args)
        if args.list_models:
            for model_id in available:
                print(model_id)
            return
    elif args.list_models:
        raise RuntimeError("--list-models requires fetching /models.")

    resolved_models, aliases, missing = resolve_requested_models(
        args.models,
        available,
        allow_unlisted=args.allow_unlisted or args.no_fetch_model_list,
    )

    print("[models] requested_to_resolved=" + json.dumps(aliases, ensure_ascii=False), flush=True)
    if missing:
        print("[models] unavailable_skipped=" + json.dumps(missing, ensure_ascii=False), flush=True)
    if args.resolve_models:
        return
    if not resolved_models:
        raise RuntimeError("No runnable models resolved. Use --allow-unlisted if your API gateway hides models from /models.")

    levels = set(args.levels) if args.levels else None
    records = ckpt.load_records(
        args.dataset_root,
        args.frozenlake_bench_root,
        args.tasks,
        args.split,
        levels,
        args.start_index,
        args.samples_per_task,
    )
    print(f"[select] cases={len(records)} tasks={','.join(args.tasks)} split={args.split}", flush=True)
    if not records:
        return

    if args.dry_run:
        for record in records[:20]:
            image_path = ckpt.render_record_image(record, args.image_cache, args.overwrite_images)
            print(
                f"[dry-run] case={record['_case_id']} task={record['_task']} "
                f"level={ckpt.record_level(record)} image={image_path}",
                flush=True,
            )
        if len(records) > 20:
            print(f"[dry-run] omitted {len(records) - 20} more case(s)", flush=True)
        return

    prompts = ckpt.load_prompts(args)
    if args.model_workers == 1 or len(resolved_models) == 1:
        for model in resolved_models:
            output_path = output_path_for_model(args, model, len(resolved_models))
            evaluate_model(model, records, endpoint, prompts, args, output_path)
    else:
        max_workers = min(args.model_workers, len(resolved_models))
        print(
            f"[models] concurrent_models={max_workers} workers_per_model={args.workers} "
            f"max_inflight_requests={max_workers * args.workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for model in resolved_models:
                output_path = output_path_for_model(args, model, len(resolved_models))
                futures.append(executor.submit(evaluate_model, model, records, endpoint, prompts, args, output_path))
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()

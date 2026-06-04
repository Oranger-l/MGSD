from __future__ import annotations

import argparse
import json
import re
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym


ACTION_TO_ID = {
    "LEFT": 0,
    "DOWN": 1,
    "RIGHT": 2,
    "UP": 3,
}

ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}

ACTION_ALIASES = {
    "L": "LEFT",
    "LEFT": "LEFT",
    "LEFTWARD": "LEFT",
    "LEFTWARDS": "LEFT",
    "R": "RIGHT",
    "RIGHT": "RIGHT",
    "RIGHTWARD": "RIGHT",
    "RIGHTWARDS": "RIGHT",
    "U": "UP",
    "UP": "UP",
    "UPWARD": "UP",
    "UPWARDS": "UP",
    "D": "DOWN",
    "DOWN": "DOWN",
    "DOWNWARD": "DOWN",
    "DOWNWARDS": "DOWN",
    "0": "LEFT",
    "1": "DOWN",
    "2": "RIGHT",
    "3": "UP",
    "←": "LEFT",
    "↓": "DOWN",
    "→": "RIGHT",
    "↑": "UP",
}

ACTION_PATTERN = re.compile(
    r"\b(?:LEFT|RIGHT|UP|DOWN|L|R|U|D|0|1|2|3)\b|[←↓→↑]",
    flags=re.IGNORECASE,
)
COMPACT_ACTION_PATTERN = re.compile(r"^[LRUD0123←↓→↑]+$", flags=re.IGNORECASE)


@dataclass
class FrozenLakeTask:
    split: str
    level: int
    index: int
    seed: int
    frozen_probability: float
    is_slippery: bool
    desc: list[str]
    source_path: str


@dataclass
class FrozenLakeEvalResult:
    parse_success: bool
    success: bool
    reached_goal: bool
    fell_into_hole: bool
    exhausted_actions: bool
    terminated: bool
    truncated: bool
    reward: float
    steps_executed: int
    action_count: int
    normalized_actions: list[str]
    invalid_action_tokens: list[str]
    final_state: int
    final_row: int
    final_col: int
    final_tile: str
    shortest_path_length: int | None
    shortest_path_actions: list[str] | None
    error: str | None = None


def load_task_from_text(text_path: str | Path) -> FrozenLakeTask:
    text_path = Path(text_path)
    lines = text_path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, str] = {}
    desc: list[str] = []

    in_map = False
    for line in lines:
        if in_map:
            if line.strip():
                desc.append(line.strip())
            continue
        if line.strip() == "map:":
            in_map = True
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()

    return FrozenLakeTask(
        split=metadata.get("split", "unknown"),
        level=int(metadata["level"]),
        index=int(metadata["index"]),
        seed=int(metadata["seed"]),
        frozen_probability=float(metadata["frozen_probability"]),
        is_slippery=metadata.get("is_slippery", "false").lower() == "true",
        desc=desc,
        source_path=str(text_path),
    )


def extract_action_text(prediction_text: str) -> str:
    match = re.search(r"<action>(.*?)</action>", prediction_text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return prediction_text.strip()


def parse_action_sequence(prediction_text: str) -> tuple[list[str], list[str]]:
    action_text = extract_action_text(prediction_text)
    raw_tokens = ACTION_PATTERN.findall(action_text)
    normalized_actions: list[str] = []
    invalid_tokens: list[str] = []

    for token in raw_tokens:
        normalized = ACTION_ALIASES.get(token.upper(), ACTION_ALIASES.get(token, None))
        if normalized is None:
            invalid_tokens.append(token)
        else:
            normalized_actions.append(normalized)

    if not normalized_actions:
        # Last fallback: split the <action> block by separators and inspect tokens one by one.
        fallback_tokens = re.split(r"[\s,;\-\>\[\]\(\)\|/]+", action_text)
        for token in fallback_tokens:
            token = token.strip()
            if not token:
                continue
            normalized = ACTION_ALIASES.get(token.upper(), ACTION_ALIASES.get(token, None))
            if normalized is None:
                invalid_tokens.append(token)
            else:
                normalized_actions.append(normalized)

    if not normalized_actions:
        compact_text = re.sub(r"\s+", "", action_text)
        if compact_text and COMPACT_ACTION_PATTERN.fullmatch(compact_text):
            invalid_tokens = []
            for token in compact_text:
                normalized = ACTION_ALIASES.get(token.upper(), ACTION_ALIASES.get(token, None))
                if normalized is None:
                    invalid_tokens.append(token)
                else:
                    normalized_actions.append(normalized)

    return normalized_actions, invalid_tokens


def _find_tile(desc: list[str], tile: str) -> tuple[int, int]:
    for row_idx, row in enumerate(desc):
        col_idx = row.find(tile)
        if col_idx != -1:
            return row_idx, col_idx
    raise ValueError(f"Tile {tile!r} not found in FrozenLake map.")


def shortest_safe_path(desc: list[str]) -> list[str] | None:
    nrow = len(desc)
    ncol = len(desc[0])
    start = _find_tile(desc, "S")
    goal = _find_tile(desc, "G")
    deltas = {
        "LEFT": (0, -1),
        "DOWN": (1, 0),
        "RIGHT": (0, 1),
        "UP": (-1, 0),
    }

    queue = deque([(start, [])])
    visited = {start}
    while queue:
        (row, col), path = queue.popleft()
        if (row, col) == goal:
            return path
        for action in ("LEFT", "DOWN", "RIGHT", "UP"):
            dr, dc = deltas[action]
            nr, nc = row + dr, col + dc
            if not (0 <= nr < nrow and 0 <= nc < ncol):
                continue
            if desc[nr][nc] == "H":
                continue
            node = (nr, nc)
            if node in visited:
                continue
            visited.add(node)
            queue.append((node, path + [action]))
    return None


def evaluate_prediction_text(
    task: FrozenLakeTask,
    prediction_text: str,
) -> FrozenLakeEvalResult:
    normalized_actions, invalid_tokens = parse_action_sequence(prediction_text)
    shortest_actions = shortest_safe_path(task.desc)

    if not normalized_actions:
        start_row, start_col = _find_tile(task.desc, "S")
        return FrozenLakeEvalResult(
            parse_success=False,
            success=False,
            reached_goal=False,
            fell_into_hole=False,
            exhausted_actions=True,
            terminated=False,
            truncated=False,
            reward=0.0,
            steps_executed=0,
            action_count=0,
            normalized_actions=[],
            invalid_action_tokens=invalid_tokens,
            final_state=start_row * task.level + start_col,
            final_row=start_row,
            final_col=start_col,
            final_tile="S",
            shortest_path_length=None if shortest_actions is None else len(shortest_actions),
            shortest_path_actions=shortest_actions,
            error="No valid action sequence found. Prefer wrapping actions in <action>...</action>.",
        )

    env = gym.make(
        "FrozenLake-v1",
        desc=task.desc,
        map_name=None,
        is_slippery=task.is_slippery,
    )
    try:
        state, _ = env.reset(seed=task.seed)
        reward = 0.0
        terminated = False
        truncated = False
        steps_executed = 0
        for action_name in normalized_actions:
            state, reward, terminated, truncated, _ = env.step(ACTION_TO_ID[action_name])
            steps_executed += 1
            if terminated or truncated:
                break

        final_row, final_col = divmod(state, task.level)
        final_tile = task.desc[final_row][final_col]
        reached_goal = final_tile == "G"
        fell_into_hole = final_tile == "H"
        exhausted_actions = steps_executed == len(normalized_actions) and not (terminated or truncated)

        return FrozenLakeEvalResult(
            parse_success=True,
            success=reached_goal,
            reached_goal=reached_goal,
            fell_into_hole=fell_into_hole,
            exhausted_actions=exhausted_actions,
            terminated=terminated,
            truncated=truncated,
            reward=float(reward),
            steps_executed=steps_executed,
            action_count=len(normalized_actions),
            normalized_actions=normalized_actions,
            invalid_action_tokens=invalid_tokens,
            final_state=int(state),
            final_row=int(final_row),
            final_col=int(final_col),
            final_tile=final_tile,
            shortest_path_length=None if shortest_actions is None else len(shortest_actions),
            shortest_path_actions=shortest_actions,
            error=None,
        )
    finally:
        env.close()


def evaluate_prediction_file(
    task_text_path: str | Path,
    prediction_file: str | Path,
) -> FrozenLakeEvalResult:
    task = load_task_from_text(task_text_path)
    prediction_text = Path(prediction_file).read_text(encoding="utf-8")
    return evaluate_prediction_text(task, prediction_text)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a FrozenLake action sequence on a generated board.")
    parser.add_argument("--task-text", required=True, help="Path to the FrozenLake .txt sample metadata file.")
    parser.add_argument(
        "--prediction-text",
        default=None,
        help="Model output text containing an <action>...</action> block or a plain action sequence.",
    )
    parser.add_argument(
        "--prediction-file",
        default=None,
        help="Optional file containing the model output text. Use this instead of --prediction-text.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if bool(args.prediction_text) == bool(args.prediction_file):
        parser.error("Provide exactly one of --prediction-text or --prediction-file.")

    task = load_task_from_text(args.task_text)
    if args.prediction_file:
        result = evaluate_prediction_file(args.task_text, args.prediction_file)
    else:
        result = evaluate_prediction_text(task, args.prediction_text)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

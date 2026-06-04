"""Unified rule-based reward for mixed VSP-Tasks OPCD training."""

from __future__ import annotations

import json
import re
from typing import Any


REWARD_NAME = "vsp_tasks"
REWARD_TYPE = "batch"

MOVE_ALIASES = {
    "L": "L",
    "LEFT": "L",
    "LEFTWARD": "L",
    "LEFTWARDS": "L",
    "R": "R",
    "RIGHT": "R",
    "RIGHTWARD": "R",
    "RIGHTWARDS": "R",
    "U": "U",
    "UP": "U",
    "UPWARD": "U",
    "UPWARDS": "U",
    "D": "D",
    "DOWN": "D",
    "DOWNWARD": "D",
    "DOWNWARDS": "D",
    "0": "L",
    "1": "D",
    "2": "R",
    "3": "U",
    "←": "L",
    "↓": "D",
    "→": "R",
    "↑": "U",
}

INTERACTION_ALIASES = {
    "PICK": "PICK",
    "PICKUP": "PICK",
    "PICK_UP": "PICK",
    "PICK-UP": "PICK",
    "GRAB": "PICK",
    "DROP": "DROP",
    "PUT": "DROP",
    "PUTDOWN": "DROP",
    "PUT_DOWN": "DROP",
    "PUT-DOWN": "DROP",
}

DELTAS = {
    "L": (0, -1),
    "D": (1, 0),
    "R": (0, 1),
    "U": (-1, 0),
}

TOKEN_PATTERN = re.compile(
    r"\b(?:PICK[_-]?UP|PUT[_-]?DOWN|PICKUP|PUTDOWN|PICK|DROP|GRAB|LEFT|RIGHT|UP|DOWN|L|R|U|D|0|1|2|3)\b|[←↓→↑]",
    flags=re.IGNORECASE,
)
COMPACT_MOVE_PATTERN = re.compile(r"^[LRUD0123←↓→↑]+$", flags=re.IGNORECASE)


def _extract_answer_content(text: str) -> tuple[str, bool]:
    if text is None:
        return "", False
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return "", False
    return match.group(1).strip(), True


def _normalize_token(token: str) -> str | None:
    key = token.upper()
    if key in MOVE_ALIASES:
        return MOVE_ALIASES[key]
    if key in INTERACTION_ALIASES:
        return INTERACTION_ALIASES[key]
    return None


def _extract_actions(response: str, allow_interactions: bool) -> tuple[list[str], bool]:
    content, has_answer_tag = _extract_answer_content(response)
    if not has_answer_tag:
        return [], False

    compact = re.sub(r"\s+", "", content)
    if compact and COMPACT_MOVE_PATTERN.fullmatch(compact):
        return [_normalize_token(token) for token in compact if _normalize_token(token) is not None], True

    actions: list[str] = []
    for token in TOKEN_PATTERN.findall(content):
        action = _normalize_token(token)
        if action is None:
            continue
        if action in {"PICK", "DROP"} and not allow_interactions:
            continue
        actions.append(action)
    return actions, True


def _parse_task_text(task_text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    metadata: dict[str, str] = {}
    blocks: dict[str, list[str]] = {}
    current_block: str | None = None
    for raw_line in task_text.splitlines():
        line = raw_line.strip()
        if current_block is not None:
            if line:
                blocks[current_block].append(line)
            continue
        if line in {"map:", "layout:", "grid:"}:
            current_block = line[:-1]
            blocks[current_block] = []
            continue
        if "=" in raw_line:
            key, value = raw_line.split("=", 1)
            metadata[key.strip()] = value.strip()
    return metadata, blocks


def _json_field(metadata: dict[str, str], key: str):
    return json.loads(metadata[key])


def _find_tile(desc: list[str], tile: str) -> tuple[int, int]:
    for row_idx, row in enumerate(desc):
        col_idx = row.find(tile)
        if col_idx != -1:
            return row_idx, col_idx
    raise ValueError(f"Tile {tile!r} not found.")


def _simulate_frozenlake(metadata: dict[str, str], blocks: dict[str, list[str]], actions: list[str]) -> dict[str, float]:
    desc = blocks.get("map", [])
    if not desc:
        return {"success": 0.0, "illegal_action": 1.0}
    row, col = _find_tile(desc, "S")
    nrow = len(desc)
    ncol = len(desc[0])
    for action in actions:
        dr, dc = DELTAS[action]
        next_row = row + dr
        next_col = col + dc
        if 0 <= next_row < nrow and 0 <= next_col < ncol:
            row, col = next_row, next_col
        tile = desc[row][col]
        if tile == "H":
            return {"success": 0.0, "illegal_action": 0.0}
        if tile == "G":
            return {"success": 1.0, "illegal_action": 0.0}
    return {"success": 0.0, "illegal_action": 0.0}


def _simulate_maze(metadata: dict[str, str], actions: list[str]) -> dict[str, float]:
    layout = _json_field(metadata, "layout_json")
    level = int(metadata["level"])
    row = int(metadata["start_row"])
    col = int(metadata["start_col"])
    target = (int(metadata["target_row"]), int(metadata["target_col"]))
    wall_key = {"U": "north", "D": "south", "R": "east", "L": "west"}

    for action in actions:
        if bool(layout[row][col][wall_key[action]]):
            return {"success": 0.0, "illegal_action": 1.0}
        dr, dc = DELTAS[action]
        row += dr
        col += dc
        if not (0 <= row < level and 0 <= col < level):
            return {"success": 0.0, "illegal_action": 1.0}
        if (row, col) == target:
            return {"success": 1.0, "illegal_action": 0.0}
    return {"success": 0.0, "illegal_action": 0.0}


def _is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _simulate_minibehaviour(metadata: dict[str, str], actions: list[str]) -> dict[str, float]:
    level = int(metadata["level"])
    row = int(metadata["start_row"])
    col = int(metadata["start_col"])
    printer = (int(metadata["printer_row"]), int(metadata["printer_col"]))
    table_cells = {tuple(coord) for coord in _json_field(metadata, "table_pos")}
    carrying = False

    for action in actions:
        coord = (row, col)
        if action in DELTAS:
            dr, dc = DELTAS[action]
            next_coord = (row + dr, col + dc)
            if not (0 <= next_coord[0] < level and 0 <= next_coord[1] < level):
                return {"success": 0.0, "illegal_action": 1.0}
            if next_coord in table_cells:
                return {"success": 0.0, "illegal_action": 1.0}
            if next_coord == printer and not carrying:
                return {"success": 0.0, "illegal_action": 1.0}
            row, col = next_coord
        elif action == "PICK":
            if carrying or not _is_adjacent(coord, printer):
                return {"success": 0.0, "illegal_action": 1.0}
            carrying = True
        elif action == "DROP":
            if not carrying or not any(_is_adjacent(coord, table_cell) for table_cell in table_cells):
                return {"success": 0.0, "illegal_action": 1.0}
            return {"success": 1.0, "illegal_action": 0.0}
        else:
            return {"success": 0.0, "illegal_action": 1.0}
    return {"success": 0.0, "illegal_action": 0.0}


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        task_text = reward_input.get("vsp_task_text") or reward_input.get("frozenlake_task_text", "")
        metadata, blocks = _parse_task_text(task_text)
        task = str(reward_input.get("task") or metadata.get("task", "")).lower()
        if task == "minibehavior":
            task = "minibehaviour"

        actions, has_answer_tag = _extract_actions(response, allow_interactions=(task == "minibehaviour"))
        format_score = 1.0 if has_answer_tag else 0.0
        parse_success = 1.0 if actions else 0.0

        if not actions:
            result = {"success": 0.0, "illegal_action": 0.0}
        elif task == "frozenlake":
            result = _simulate_frozenlake(metadata, blocks, [action for action in actions if action in DELTAS])
        elif task == "maze":
            result = _simulate_maze(metadata, [action for action in actions if action in DELTAS])
        elif task == "minibehaviour":
            result = _simulate_minibehaviour(metadata, actions)
        else:
            result = {"success": 0.0, "illegal_action": 1.0}

        success = float(result["success"])
        scores.append(
            {
                "overall": success,
                "accuracy": success,
                "success": success,
                "format": format_score,
                "parse_success": parse_success,
                "illegal_action": float(result["illegal_action"]),
            }
        )
    return scores

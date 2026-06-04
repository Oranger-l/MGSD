from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from PIL import Image

from .renderer import render_desc_to_image


@dataclass
class FrozenLakeSample:
    split: str
    level: int
    index: int
    seed: int
    frozen_probability: float
    desc: List[str]
    solution_actions: list[str]
    trajectory_states: list[list[int]]
    image_path: Path
    text_path: Path


def _normalize_desc(desc) -> List[str]:
    rows = []
    for row in desc:
        if isinstance(row, bytes):
            rows.append(row.decode("utf-8"))
        elif isinstance(row, np.ndarray):
            rows.append("".join(x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in row.tolist()))
        elif isinstance(row, (list, tuple)):
            rows.append("".join(x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in row))
        else:
            rows.append(str(row))
    return rows


def generate_random_desc(size: int, seed: int, frozen_probability: float = 0.9) -> List[str]:
    try:
        desc = generate_random_map(size=size, p=frozen_probability, seed=seed)
    except TypeError:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            desc = generate_random_map(size=size, p=frozen_probability)
        finally:
            np.random.set_state(state)
    return _normalize_desc(desc)


def _find_tile(desc: List[str], tile: str) -> tuple[int, int]:
    for row_idx, row in enumerate(desc):
        col_idx = row.find(tile)
        if col_idx != -1:
            return row_idx, col_idx
    raise ValueError(f"Tile {tile!r} not found in desc.")


def shortest_safe_path_length(desc: List[str]) -> int | None:
    path = shortest_safe_path(desc)
    if path is None:
        return None
    return len(path) - 1


def shortest_safe_path(desc: List[str]) -> list[tuple[int, int]] | None:
    from collections import deque

    start = _find_tile(desc, "S")
    goal = _find_tile(desc, "G")
    nrow = len(desc)
    ncol = len(desc[0])
    deltas = ((0, -1), (1, 0), (0, 1), (-1, 0))
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        (row, col), path = queue.popleft()
        if (row, col) == goal:
            return path
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < nrow and 0 <= nc < ncol):
                continue
            if desc[nr][nc] == "H":
                continue
            node = (nr, nc)
            if node in visited:
                continue
            visited.add(node)
            queue.append((node, path + [node]))
    return None


def path_to_actions(path: list[tuple[int, int]]) -> list[str]:
    actions: list[str] = []
    for (row, col), (next_row, next_col) in zip(path, path[1:]):
        delta = (next_row - row, next_col - col)
        if delta == (-1, 0):
            actions.append("up")
        elif delta == (1, 0):
            actions.append("down")
        elif delta == (0, -1):
            actions.append("left")
        elif delta == (0, 1):
            actions.append("right")
        else:
            raise ValueError(f"Non-adjacent path step: {(row, col)} -> {(next_row, next_col)}")
    return actions


def randomize_start_goal_positions(
    desc: List[str],
    seed: int,
    min_path_length: int = 2,
    max_tries: int = 512,
) -> List[str]:
    rows = [list(row) for row in desc]
    for r, row in enumerate(rows):
        for c, tile in enumerate(row):
            if tile in {"S", "G"}:
                rows[r][c] = "F"

    open_cells = [(r, c) for r, row in enumerate(rows) for c, tile in enumerate(row) if tile != "H"]
    if len(open_cells) < 2:
        raise ValueError("Need at least two non-hole cells to place start and goal.")

    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        start_idx, goal_idx = rng.choice(len(open_cells), size=2, replace=False)
        start = open_cells[int(start_idx)]
        goal = open_cells[int(goal_idx)]
        candidate_rows = [row[:] for row in rows]
        candidate_rows[start[0]][start[1]] = "S"
        candidate_rows[goal[0]][goal[1]] = "G"
        candidate_desc = ["".join(row) for row in candidate_rows]
        path_length = shortest_safe_path_length(candidate_desc)
        if path_length is None:
            continue
        if path_length < min_path_length:
            continue
        return candidate_desc

    raise RuntimeError("Failed to sample valid randomized start/goal positions.")


def choose_cell_size(level: int) -> int:
    return min(64, 512 // level)


def validate_desc(desc: List[str], seed: int, is_slippery: bool) -> None:
    env = gym.make("FrozenLake-v1", desc=desc, map_name=None, is_slippery=is_slippery)
    try:
        env.reset(seed=seed)
    finally:
        env.close()


def write_text_sample(
    text_path: Path,
    split: str,
    level: int,
    index: int,
    seed: int,
    desc: List[str],
    is_slippery: bool,
    frozen_probability: float,
) -> None:
    start_row, start_col = _find_tile(desc, "S")
    goal_row, goal_col = _find_tile(desc, "G")
    path = shortest_safe_path(desc)
    if path is None:
        raise ValueError("Cannot write an unsolved FrozenLake sample.")
    solution_actions = path_to_actions(path)
    trajectory_states = [[row, col] for row, col in path]
    content = [
        "task=frozenlake",
        f"split={split}",
        f"level={level}",
        f"index={index}",
        f"seed={seed}",
        f"frozen_probability={frozen_probability}",
        f"is_slippery={str(is_slippery).lower()}",
        f"start_row={start_row}",
        f"start_col={start_col}",
        f"goal_row={goal_row}",
        f"goal_col={goal_col}",
        f"path_length={len(solution_actions)}",
        f"solution_actions={json.dumps(solution_actions)}",
        f"trajectory_states={json.dumps(trajectory_states)}",
        "map:",
        *desc,
    ]
    text_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def generate_level_dataset(
    output_root: str | Path,
    level: int,
    sample_count: int,
    base_seed: int = 42,
    frozen_probability: float = 0.9,
    is_slippery: bool = False,
    start_index: int = 0,
    seen_desc: set[str] | None = None,
    output_subdir_name: str | None = None,
    split: str = "train",
    seed_offset: int = 0,
    randomize_start_goal: bool = False,
    min_start_goal_path_length: int = 2,
    image_size: int | None = 256,
) -> list[FrozenLakeSample]:
    output_root = Path(output_root)
    level_dir_name = output_subdir_name or f"level{level}"
    level_root = output_root / level_dir_name
    image_dir = level_root / "image"
    text_dir = level_root / "text"
    image_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    samples: list[FrozenLakeSample] = []
    if seen_desc is None:
        seen_desc = set()
    attempt = 0
    while len(samples) < sample_count:
        sample_seed = (
            seed_offset
            + base_seed
            + level * 100_000
            + int(frozen_probability * 1_000) * 1_000
            + attempt
        )
        desc = generate_random_desc(level, sample_seed, frozen_probability=frozen_probability)
        if randomize_start_goal:
            desc = randomize_start_goal_positions(
                desc,
                seed=sample_seed + 17,
                min_path_length=min_start_goal_path_length,
            )
        desc_key = "\n".join(desc)
        attempt += 1
        if desc_key in seen_desc:
            continue
        path = shortest_safe_path(desc)
        if path is None:
            continue
        validate_desc(desc, sample_seed, is_slippery=is_slippery)
        seen_desc.add(desc_key)

        index = start_index + len(samples)
        image_path = image_dir / f"{index:04d}.png"
        text_path = text_dir / f"{index:04d}.txt"
        image = render_desc_to_image(desc, cell_size=choose_cell_size(level))
        if image_size is not None and image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), resample=Image.Resampling.LANCZOS)
        image.save(image_path)
        write_text_sample(
            text_path,
            split,
            level,
            index,
            sample_seed,
            desc,
            is_slippery=is_slippery,
            frozen_probability=frozen_probability,
        )
        samples.append(
            FrozenLakeSample(
                split=split,
                level=level,
                index=index,
                seed=sample_seed,
                frozen_probability=frozen_probability,
                desc=desc,
                solution_actions=path_to_actions(path),
                trajectory_states=[[row, col] for row, col in path],
                image_path=image_path,
                text_path=text_path,
            )
        )

    return samples

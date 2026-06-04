from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .renderer import render_maze


Direction = tuple[str, int, int]
Grid = list[list[dict[str, bool]]]

DIRECTIONS: tuple[Direction, ...] = (
    ("N", -1, 0),
    ("E", 0, 1),
    ("S", 1, 0),
    ("W", 0, -1),
)
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
ACTION_BY_DELTA = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}


@dataclass
class MazeSample:
    split: str
    level: int
    index: int
    seed: int
    start: tuple[int, int]
    target: tuple[int, int]
    solution_actions: list[str]
    trajectory_states: list[list[int]]
    image_path: Path
    text_path: Path


def make_empty_grid(level: int) -> Grid:
    return [[{"N": True, "E": True, "S": True, "W": True} for _ in range(level)] for _ in range(level)]


def neighbors_of(row: int, col: int, level: int) -> list[tuple[str, int, int]]:
    neighbors: list[tuple[str, int, int]] = []
    for direction, drow, dcol in DIRECTIONS:
        next_row = row + drow
        next_col = col + dcol
        if 0 <= next_row < level and 0 <= next_col < level:
            neighbors.append((direction, next_row, next_col))
    return neighbors


def remove_wall(grid: Grid, row: int, col: int, direction: str) -> None:
    drow, dcol = next((dr, dc) for d, dr, dc in DIRECTIONS if d == direction)
    next_row = row + drow
    next_col = col + dcol
    grid[row][col][direction] = False
    grid[next_row][next_col][OPPOSITE[direction]] = False


def generate_perfect_maze(level: int, rng: random.Random) -> Grid:
    grid = make_empty_grid(level)
    visited = [[False for _ in range(level)] for _ in range(level)]
    start = (rng.randrange(level), rng.randrange(level))
    stack = [start]
    visited[start[0]][start[1]] = True

    while stack:
        row, col = stack[-1]
        candidates = [
            (direction, next_row, next_col)
            for direction, next_row, next_col in neighbors_of(row, col, level)
            if not visited[next_row][next_col]
        ]
        if not candidates:
            stack.pop()
            continue
        direction, next_row, next_col = rng.choice(candidates)
        remove_wall(grid, row, col, direction)
        visited[next_row][next_col] = True
        stack.append((next_row, next_col))

    return grid


def open_neighbors(grid: Grid, coord: tuple[int, int]) -> list[tuple[int, int]]:
    level = len(grid)
    row, col = coord
    neighbors: list[tuple[int, int]] = []
    for direction, drow, dcol in DIRECTIONS:
        if grid[row][col][direction]:
            continue
        next_row = row + drow
        next_col = col + dcol
        if 0 <= next_row < level and 0 <= next_col < level:
            neighbors.append((next_row, next_col))
    return neighbors


def shortest_path(grid: Grid, start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        coord, path = queue.popleft()
        if coord == target:
            return path
        for next_coord in open_neighbors(grid, coord):
            if next_coord in visited:
                continue
            visited.add(next_coord)
            queue.append((next_coord, path + [next_coord]))
    raise ValueError("No path exists in generated maze.")


def distance_map_to_target(grid: Grid, target: tuple[int, int]) -> dict[str, int]:
    level = len(grid)
    queue = deque([(target, 0)])
    distances = {target: 0}
    while queue:
        coord, distance = queue.popleft()
        for next_coord in open_neighbors(grid, coord):
            if next_coord in distances:
                continue
            distances[next_coord] = distance + 1
            queue.append((next_coord, distance + 1))
    return {str(row * level + col): distance for (row, col), distance in sorted(distances.items())}


def path_to_actions(path: list[tuple[int, int]]) -> list[str]:
    actions: list[str] = []
    for (row, col), (next_row, next_col) in zip(path, path[1:]):
        actions.append(ACTION_BY_DELTA[(next_row - row, next_col - col)])
    return actions


def grid_to_wall_masks(grid: Grid) -> list[list[int]]:
    masks: list[list[int]] = []
    for row in grid:
        mask_row: list[int] = []
        for cell in row:
            value = 0
            if cell["N"]:
                value |= 1
            if cell["S"]:
                value |= 2
            if cell["W"]:
                value |= 4
            if cell["E"]:
                value |= 8
            mask_row.append(value)
        masks.append(mask_row)
    return masks


def grid_to_visual_planning_layout(grid: Grid) -> list[list[dict[str, bool]]]:
    return [
        [
            {
                "north": cell["N"],
                "south": cell["S"],
                "east": cell["E"],
                "west": cell["W"],
            }
            for cell in row
        ]
        for row in grid
    ]


def classify_path(path_length: int, level: int) -> str:
    if path_length <= max(1, level // 2):
        return "easy"
    if path_length <= level:
        return "medium"
    return "hard"


def sample_maze_for_bucket(
    level: int,
    seed: int,
    bucket: str,
    max_tries: int = 4096,
) -> tuple[Grid, tuple[int, int], tuple[int, int], list[tuple[int, int]], str]:
    rng = random.Random(seed)
    fallback: tuple[Grid, tuple[int, int], tuple[int, int], list[tuple[int, int]], str] | None = None
    nodes = [(row, col) for row in range(level) for col in range(level)]

    for _ in range(max_tries):
        grid = generate_perfect_maze(level, rng)
        start, target = rng.sample(nodes, 2)
        path = shortest_path(grid, start, target)
        path_bucket = classify_path(len(path) - 1, level)
        candidate = (grid, start, target, path, path_bucket)
        if fallback is None:
            fallback = candidate
        if path_bucket == bucket:
            return candidate

    if fallback is None:
        raise RuntimeError("Failed to generate a maze candidate.")
    return fallback


def write_text_sample(
    text_path: Path,
    split: str,
    level: int,
    index: int,
    seed: int,
    grid: Grid,
    start: tuple[int, int],
    target: tuple[int, int],
    path: list[tuple[int, int]],
    difficulty_bucket: str,
) -> None:
    actions = path_to_actions(path)
    trajectory_states = [[row, col] for row, col in path]
    wall_masks = grid_to_wall_masks(grid)
    layout = grid_to_visual_planning_layout(grid)
    distance_map = distance_map_to_target(grid, target)
    content = [
        "task=maze",
        f"split={split}",
        f"level={level}",
        f"index={index}",
        f"seed={seed}",
        f"difficulty_bucket={difficulty_bucket}",
        f"start_row={start[0]}",
        f"start_col={start[1]}",
        f"target_row={target[0]}",
        f"target_col={target[1]}",
        f"start_pos={start[0] * level + start[1]}",
        f"target_pos={target[0] * level + target[1]}",
        f"path_length={len(actions)}",
        f"solution_actions={json.dumps(actions)}",
        f"trajectory_states={json.dumps(trajectory_states)}",
        f"layout_json={json.dumps(layout, separators=(',', ':'))}",
        f"distance_map={json.dumps(distance_map, sort_keys=True, separators=(',', ':'))}",
        "layout:",
        *(" ".join(str(value) for value in row) for row in wall_masks),
    ]
    text_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def maze_key(grid: Grid, start: tuple[int, int], target: tuple[int, int]) -> str:
    return json.dumps(grid_to_wall_masks(grid), separators=(",", ":")) + f"|{start}|{target}"


def generate_level_dataset(
    output_root: str | Path,
    level: int,
    sample_count: int,
    base_seed: int = 42,
    start_index: int = 0,
    split: str = "train",
    seed_offset: int = 0,
    seen_keys: set[str] | None = None,
    image_size: int = 256,
    difficulty_buckets: tuple[str, ...] = ("easy", "medium", "hard"),
) -> list[MazeSample]:
    output_root = Path(output_root)
    level_root = output_root / f"level{level}"
    image_dir = level_root / "image"
    text_dir = level_root / "text"
    image_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    if seen_keys is None:
        seen_keys = set()

    samples: list[MazeSample] = []
    attempt = 0
    while len(samples) < sample_count:
        bucket = difficulty_buckets[(start_index + len(samples)) % len(difficulty_buckets)]
        sample_seed = seed_offset + base_seed + level * 100_000 + attempt
        attempt += 1
        grid, start, target, path, actual_bucket = sample_maze_for_bucket(level, sample_seed, bucket)
        key = maze_key(grid, start, target)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        index = start_index + len(samples)
        image_path = image_dir / f"{index:04d}.png"
        text_path = text_dir / f"{index:04d}.txt"
        render_maze(grid, start, target, image_size=image_size).save(image_path)
        write_text_sample(
            text_path,
            split,
            level,
            index,
            sample_seed,
            grid,
            start,
            target,
            path,
            actual_bucket,
        )
        samples.append(
            MazeSample(
                split=split,
                level=level,
                index=index,
                seed=sample_seed,
                start=start,
                target=target,
                solution_actions=path_to_actions(path),
                trajectory_states=[[row, col] for row, col in path],
                image_path=image_path,
                text_path=text_path,
            )
        )

    return samples

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .renderer import render_minibehaviour


Coord = tuple[int, int]
ACTION_BY_DELTA = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}


@dataclass
class MiniBehaviourSample:
    split: str
    level: int
    index: int
    seed: int
    start: Coord
    printer_pos: Coord
    table_pos: list[Coord]
    solution_actions: list[str]
    trajectory_states: list[dict[str, object]]
    image_path: Path
    text_path: Path


def neighbors(coord: Coord, level: int) -> list[Coord]:
    row, col = coord
    result: list[Coord] = []
    for drow, dcol in ACTION_BY_DELTA:
        next_row = row + drow
        next_col = col + dcol
        if 0 <= next_row < level and 0 <= next_col < level:
            result.append((next_row, next_col))
    return result


def unique_sorted(coords: Iterable[Coord]) -> list[Coord]:
    return sorted(set(coords))


def make_table_positions(top_left: Coord) -> list[Coord]:
    row, col = top_left
    return [(row + drow, col + dcol) for drow in range(2) for dcol in range(3)]


def free_cells(level: int, blocked: set[Coord]) -> list[Coord]:
    return [(row, col) for row in range(level) for col in range(level) if (row, col) not in blocked]


def object_neighbors(level: int, object_cells: Iterable[Coord], blocked: set[Coord]) -> list[Coord]:
    return unique_sorted(
        neighbor
        for cell in object_cells
        for neighbor in neighbors(cell, level)
        if neighbor not in blocked
    )


def bfs_path(level: int, blocked: set[Coord], start: Coord, target: Coord) -> list[Coord] | None:
    if start in blocked or target in blocked:
        return None
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        coord, path = queue.popleft()
        if coord == target:
            return path
        for next_coord in neighbors(coord, level):
            if next_coord in blocked or next_coord in visited:
                continue
            visited.add(next_coord)
            queue.append((next_coord, path + [next_coord]))
    return None


def distance_map(level: int, blocked: set[Coord], target: Coord) -> dict[str, int]:
    queue = deque([(target, 0)])
    distances = {target: 0}
    while queue:
        coord, distance = queue.popleft()
        for next_coord in neighbors(coord, level):
            if next_coord in blocked or next_coord in distances:
                continue
            distances[next_coord] = distance + 1
            queue.append((next_coord, distance + 1))
    return {str(coord): distance for coord, distance in sorted(distances.items())}


def path_to_actions(path: list[Coord]) -> list[str]:
    actions: list[str] = []
    for (row, col), (next_row, next_col) in zip(path, path[1:]):
        actions.append(ACTION_BY_DELTA[(next_row - row, next_col - col)])
    return actions


def build_trajectory(
    start: Coord,
    path_to_printer: list[Coord],
    path_to_table: list[Coord],
) -> tuple[list[str], list[dict[str, object]]]:
    actions: list[str] = []
    states: list[dict[str, object]] = [{"coord": [start[0], start[1]], "carrying": False}]

    for action, coord in zip(path_to_actions(path_to_printer), path_to_printer[1:]):
        actions.append(action)
        states.append({"coord": [coord[0], coord[1]], "carrying": False})

    pickup_coord = path_to_printer[-1]
    actions.append("pick")
    states.append({"coord": [pickup_coord[0], pickup_coord[1]], "carrying": True})

    for action, coord in zip(path_to_actions(path_to_table), path_to_table[1:]):
        actions.append(action)
        states.append({"coord": [coord[0], coord[1]], "carrying": True})

    drop_coord = path_to_table[-1]
    actions.append("drop")
    states.append({"coord": [drop_coord[0], drop_coord[1]], "carrying": False})
    return actions, states


def classify_path(path_length: int, level: int) -> str:
    if path_length <= level - 1:
        return "easy"
    if path_length <= level + 1:
        return "medium"
    return "hard"


def sample_candidate(
    level: int,
    rng: random.Random,
) -> tuple[Coord, Coord, list[Coord], list[Coord], list[Coord], list[list[Coord]], list[str], list[dict[str, object]], str] | None:
    table_top_lefts = [(row, col) for row in range(level - 1) for col in range(level - 2)]
    table_pos = make_table_positions(rng.choice(table_top_lefts))
    table_set = set(table_pos)

    printer_choices = free_cells(level, table_set)
    if not printer_choices:
        return None
    printer_pos = rng.choice(printer_choices)
    pre_pick_blocked = set(table_pos)
    pre_pick_blocked.add(printer_pos)
    post_pick_blocked = set(table_pos)

    printer_neighbors = object_neighbors(level, [printer_pos], pre_pick_blocked)
    table_neighbors = object_neighbors(level, table_pos, post_pick_blocked)
    if not printer_neighbors or not table_neighbors:
        return None

    start_choices = free_cells(level, pre_pick_blocked)
    if not start_choices:
        return None
    start = rng.choice(start_choices)

    distance_to_printer = {target: distance_map(level, pre_pick_blocked, target) for target in printer_neighbors}
    distance_to_table = {target: distance_map(level, post_pick_blocked, target) for target in table_neighbors}

    best_distance = None
    best_paths: list[list[Coord]] = []
    for printer_neighbor in printer_neighbors:
        start_distance = distance_to_printer[printer_neighbor].get(str(start))
        if start_distance is None:
            continue
        for table_neighbor in table_neighbors:
            bridge_distance = distance_to_table[table_neighbor].get(str(printer_neighbor))
            if bridge_distance is None:
                continue
            total_distance = start_distance + bridge_distance
            if best_distance is None or total_distance < best_distance:
                best_distance = total_distance
                best_paths = [[printer_neighbor, table_neighbor]]
            elif total_distance == best_distance:
                best_paths.append([printer_neighbor, table_neighbor])

    if best_distance is None or not best_paths:
        return None

    printer_target, table_target = rng.choice(best_paths)
    path_to_printer = bfs_path(level, pre_pick_blocked, start, printer_target)
    path_to_table = bfs_path(level, post_pick_blocked, printer_target, table_target)
    if path_to_printer is None or path_to_table is None:
        return None

    actions, trajectory_states = build_trajectory(start, path_to_printer, path_to_table)
    if not actions or actions[-1] != "drop" or "pick" not in actions:
        return None
    bucket = classify_path(len(actions), level)
    return (
        start,
        printer_pos,
        table_pos,
        printer_neighbors,
        table_neighbors,
        best_paths,
        actions,
        trajectory_states,
        bucket,
    )


def sample_for_bucket(
    level: int,
    seed: int,
    bucket: str | None,
    max_tries: int = 20000,
) -> tuple[Coord, Coord, list[Coord], list[Coord], list[Coord], list[list[Coord]], list[str], list[dict[str, object]], str]:
    rng = random.Random(seed)
    for _ in range(max_tries):
        candidate = sample_candidate(level, rng)
        if candidate is None:
            continue
        if bucket is None or candidate[-1] == bucket:
            return candidate
    bucket_message = "any bucket" if bucket is None else f"{bucket!r} bucket"
    raise RuntimeError(f"Failed to generate MiniBehaviour level={level} {bucket_message} sample.")


def grid_rows(level: int, start: Coord, printer_pos: Coord, table_pos: list[Coord]) -> list[str]:
    table_set = set(table_pos)
    rows: list[str] = []
    for row in range(level):
        chars: list[str] = []
        for col in range(level):
            coord = (row, col)
            if coord == start:
                chars.append("A")
            elif coord == printer_pos:
                chars.append("P")
            elif coord in table_set:
                chars.append("T")
            else:
                chars.append(".")
        rows.append("".join(chars))
    return rows


def coord_list(coords: list[Coord]) -> list[list[int]]:
    return [[row, col] for row, col in coords]


def nested_coord_list(paths: list[list[Coord]]) -> list[list[list[int]]]:
    return [[[row, col] for row, col in path] for path in paths]


def write_text_sample(
    text_path: Path,
    split: str,
    level: int,
    index: int,
    seed: int,
    start: Coord,
    printer_pos: Coord,
    table_pos: list[Coord],
    printer_neighbors: list[Coord],
    table_neighbors: list[Coord],
    best_paths: list[list[Coord]],
    solution_actions: list[str],
    trajectory_states: list[dict[str, object]],
    difficulty_bucket: str,
) -> None:
    pre_pick_blocked = set(table_pos)
    pre_pick_blocked.add(printer_pos)
    post_pick_blocked = set(table_pos)
    distance_to_printer = {
        str(target): distance_map(level, pre_pick_blocked, target)
        for target in printer_neighbors
    }
    distance_to_table = {
        str(target): distance_map(level, post_pick_blocked, target)
        for target in table_neighbors
    }
    content = [
        "task=minibehaviour",
        f"split={split}",
        f"level={level}",
        f"grid_rows={level}",
        f"grid_cols={level}",
        "outer_ring=false",
        f"index={index}",
        f"seed={seed}",
        f"difficulty_bucket={difficulty_bucket}",
        f"start_row={start[0]}",
        f"start_col={start[1]}",
        "start_dir=0",
        f"printer_row={printer_pos[0]}",
        f"printer_col={printer_pos[1]}",
        "printer_traversable_after_pick=true",
        f"table_pos={json.dumps(coord_list(table_pos))}",
        f"printer_neighbors={json.dumps(coord_list(printer_neighbors))}",
        f"table_neighbors={json.dumps(coord_list(table_neighbors))}",
        f"best_paths={json.dumps(nested_coord_list(best_paths))}",
        f"path_length={len(solution_actions)}",
        f"solution_actions={json.dumps(solution_actions)}",
        f"trajectory_states={json.dumps(trajectory_states, separators=(',', ':'))}",
        f"distance_map_to_printer={json.dumps(distance_to_printer, sort_keys=True, separators=(',', ':'))}",
        f"distance_map_to_table={json.dumps(distance_to_table, sort_keys=True, separators=(',', ':'))}",
        "grid:",
        *grid_rows(level, start, printer_pos, table_pos),
    ]
    text_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def sample_key(start: Coord, printer_pos: Coord, table_pos: list[Coord]) -> str:
    return f"start={start}|printer={printer_pos}|table={tuple(table_pos)}"


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
) -> list[MiniBehaviourSample]:
    output_root = Path(output_root)
    level_root = output_root / f"level{level}"
    image_dir = level_root / "image"
    text_dir = level_root / "text"
    image_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    if seen_keys is None:
        seen_keys = set()

    samples: list[MiniBehaviourSample] = []
    attempt = 0
    rejected_duplicates = 0
    max_rejected_duplicates = max(100_000, sample_count * 1000)
    while len(samples) < sample_count:
        bucket = difficulty_buckets[(start_index + len(samples)) % len(difficulty_buckets)]
        sample_seed = seed_offset + base_seed + level * 100_000 + attempt
        attempt += 1
        accepted = sample_for_bucket(level, sample_seed, bucket)
        key = sample_key(accepted[0], accepted[1], accepted[2])
        if key in seen_keys:
            rejected_duplicates += 1
            if rejected_duplicates > max_rejected_duplicates:
                raise RuntimeError(
                    f"Too many duplicate MiniBehaviour samples for level={level}; "
                    f"accepted={len(samples)}, requested={sample_count}."
                )
            continue
        rejected_duplicates = 0
        seen_keys.add(key)

        (
            start,
            printer_pos,
            table_pos,
            printer_neighbors,
            table_neighbors,
            best_paths,
            solution_actions,
            trajectory_states,
            actual_bucket,
        ) = accepted

        index = start_index + len(samples)
        image_path = image_dir / f"{index:04d}.png"
        text_path = text_dir / f"{index:04d}.txt"
        render_minibehaviour(level, start, printer_pos, table_pos, image_size=image_size).save(image_path)
        write_text_sample(
            text_path,
            split,
            level,
            index,
            sample_seed,
            start,
            printer_pos,
            table_pos,
            printer_neighbors,
            table_neighbors,
            best_paths,
            solution_actions,
            trajectory_states,
            actual_bucket,
        )
        samples.append(
            MiniBehaviourSample(
                split=split,
                level=level,
                index=index,
                seed=sample_seed,
                start=start,
                printer_pos=printer_pos,
                table_pos=table_pos,
                solution_actions=solution_actions,
                trajectory_states=trajectory_states,
                image_path=image_path,
                text_path=text_path,
            )
        )

    return samples

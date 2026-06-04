"""Preprocess mixed VSP-Tasks raw folders into normalized OPCD JSON.

The generated JSON is consumed by ``examples/data_preprocess/opcd_json.py``.
Each row keeps the task-specific student prompt inside ``messages`` and carries
the task-specific text-only teacher prompt/context in ``teacher_system_prompt``
and ``teacher_text_context``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


TASK_DIRS = {
    "frozenlake": "FrozenLake",
    "maze": "Maze",
    "minibehaviour": "MiniBehaviour",
}
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]

MOVE_TO_LETTER = {
    "left": "L",
    "down": "D",
    "right": "R",
    "up": "U",
    "l": "L",
    "d": "D",
    "r": "R",
    "u": "U",
}

DIR_ORDER = (
    ("up", "north"),
    ("right", "east"),
    ("down", "south"),
    ("left", "west"),
)

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[1] / "system_prompt"
DEFAULT_STUDENT_PROMPT_PATHS = {
    "frozenlake": DEFAULT_PROMPT_DIR / "opcd_student_Frozenlake.txt",
    "maze": DEFAULT_PROMPT_DIR / "opcd_student_Maze.txt",
    "minibehaviour": DEFAULT_PROMPT_DIR / "opcd_student_MiniBehaviour.txt",
}
DEFAULT_TEACHER_PROMPT_PATHS = {
    "frozenlake": DEFAULT_PROMPT_DIR / "opcd_teacher_text_FrozenLake.txt",
    "maze": DEFAULT_PROMPT_DIR / "opcd_teacher_text_Maze.txt",
    "minibehaviour": DEFAULT_PROMPT_DIR / "opcd_teacher_text_MiniBehaviour.txt",
}


def parse_level_names(raw: str) -> list[str]:
    names: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        names.append(item if item.startswith("level") else f"level{item}")
    return names


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def load_student_prompts(prompt_paths: dict[str, Path]) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for task, path in prompt_paths.items():
        prompt = path.read_text(encoding="utf-8").strip()
        prompt = prompt.replace("<TEST-IMAGE>", "<image>")
        if "<image>" not in prompt:
            prompt = f"{prompt}\n\n<image>"
        prompts[task] = prompt
    return prompts


def load_teacher_prompts(prompt_paths: dict[str, Path]) -> dict[str, str]:
    return {task: path.read_text(encoding="utf-8").strip() for task, path in prompt_paths.items()}


def read_task_text(text_path: Path) -> tuple[dict[str, str], dict[str, list[str]], str]:
    text = text_path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    blocks: dict[str, list[str]] = {}
    current_block: str | None = None
    for raw_line in text.splitlines():
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
    if "task" not in metadata:
        raise ValueError(f"Missing task field in {text_path}")
    return metadata, blocks, text


def json_field(metadata: dict[str, str], key: str):
    return json.loads(metadata[key])


def coord_text(coord: Iterable[int]) -> str:
    row, col = coord
    return f"({int(row)},{int(col)})"


def coord_list_text(coords: Iterable[Iterable[int]]) -> str:
    values = [coord_text(coord) for coord in coords]
    return ", ".join(values) if values else "none"


def portable_path(path: Path, root: Path) -> str:
    resolved_path = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def answer_from_metadata(metadata: dict[str, str], task: str) -> str:
    raw_actions = json_field(metadata, "solution_actions")
    normalized: list[str] = []
    for action in raw_actions:
        action_key = str(action).strip().lower()
        if task == "minibehaviour" and action_key in {"pick", "drop"}:
            normalized.append(action_key.upper())
        else:
            normalized.append(MOVE_TO_LETTER[action_key])
    return ",".join(normalized) if task == "minibehaviour" else "".join(normalized)


def frozenlake_context(
    metadata: dict[str, str],
    blocks: dict[str, list[str]],
    answer: str,
    include_teacher_ground_truth: bool = True,
) -> str:
    desc = blocks.get("map", [])
    holes = []
    for row_idx, row in enumerate(desc):
        for col_idx, tile in enumerate(row):
            if tile == "H":
                holes.append((row_idx, col_idx))
    lines = [
        "FrozenLake teacher text context.",
        (
            "Use the fully observable symbolic map below to solve the task. Do not mention that a reference plan is provided."
            if include_teacher_ground_truth
            else "Use the fully observable symbolic map below to solve the task."
        ),
        "",
        "Task: FrozenLake",
        f"Map size: {metadata['level']}x{metadata['level']}",
        f"Start: ({metadata['start_row']},{metadata['start_col']})",
        f"Goal: ({metadata['goal_row']},{metadata['goal_col']})",
        f"Holes: {coord_list_text(holes)}",
        "Text map:",
        *desc,
        "",
        "Teacher reasoning focus:",
        "- S is the player start, G is the goal, H is unsafe, and F is safe frozen land.",
        "- Produce a safe route from S to G without stepping on H.",
    ]
    if include_teacher_ground_truth:
        lines.append(f"Reference action plan: {answer}")
    lines.append("Solve the task and end with exactly one <answer>...</answer> block.")
    return "\n".join(lines)


def maze_context(
    metadata: dict[str, str],
    blocks: dict[str, list[str]],
    answer: str,
    include_teacher_ground_truth: bool = True,
) -> str:
    level = int(metadata["level"])
    layout = json_field(metadata, "layout_json")
    lines = [
        "Maze teacher text context.",
        (
            "Use the fully observable wall/open-direction table below to solve the task. Do not mention that a reference plan is provided."
            if include_teacher_ground_truth
            else "Use the fully observable wall/open-direction table below to solve the task."
        ),
        "",
        "Task: Maze",
        f"Map size: {level}x{level}",
        f"Start: ({metadata['start_row']},{metadata['start_col']})",
        f"Target: ({metadata['target_row']},{metadata['target_col']})",
        "Open directions by cell:",
    ]
    for row_idx, row in enumerate(layout):
        for col_idx, cell in enumerate(row):
            open_dirs = [name for name, key in DIR_ORDER if not bool(cell[key])]
            lines.append(f"({row_idx},{col_idx}): {', '.join(open_dirs) if open_dirs else 'none'}")
    if blocks.get("layout"):
        lines.extend(["Wall-mask table:", *blocks["layout"]])
    lines.extend([
        "",
        "Teacher reasoning focus:",
        "- A listed open direction means the move to the adjacent cell is legal.",
        "- A missing direction means a wall blocks that move.",
        "- Produce a route from Start to Target without crossing walls.",
    ])
    if include_teacher_ground_truth:
        lines.append(f"Reference action plan: {answer}")
    lines.append("Solve the task and end with exactly one <answer>...</answer> block.")
    return "\n".join(lines)


def legal_neighbors(row: int, col: int, level: int) -> list[tuple[int, int, str]]:
    candidates = [
        (row - 1, col, "U"),
        (row, col + 1, "R"),
        (row + 1, col, "D"),
        (row, col - 1, "L"),
    ]
    return [(r, c, action) for r, c, action in candidates if 0 <= r < level and 0 <= c < level]


def minibehaviour_context(
    metadata: dict[str, str],
    blocks: dict[str, list[str]],
    answer: str,
    include_teacher_ground_truth: bool = True,
) -> str:
    level = int(metadata["level"])
    start = (int(metadata["start_row"]), int(metadata["start_col"]))
    printer = (int(metadata["printer_row"]), int(metadata["printer_col"]))
    table_pos = [tuple(coord) for coord in json_field(metadata, "table_pos")]
    printer_neighbors = [tuple(coord) for coord in json_field(metadata, "printer_neighbors")]
    table_neighbors = [tuple(coord) for coord in json_field(metadata, "table_neighbors")]
    blocked = set(table_pos)
    blocked.add(printer)
    legal_moves = [
        action
        for next_row, next_col, action in legal_neighbors(start[0], start[1], level)
        if (next_row, next_col) not in blocked
    ]
    grid = blocks.get("grid", [])
    lines = [
        "MiniBehaviour teacher text context.",
        (
            "Use the fully observable object grid and adjacency lists below to solve the task. Do not mention that a reference plan is provided."
            if include_teacher_ground_truth
            else "Use the fully observable object grid and adjacency lists below to solve the task."
        ),
        "",
        "Task: MiniBehaviour",
        f"Grid size: {level}x{level}",
        f"Agent: {coord_text(start)}",
        f"Printer: {coord_text(printer)}",
        f"Table cells: {coord_list_text(table_pos)}",
        f"Printer-adjacent cells: {coord_list_text(printer_neighbors)}",
        f"Table-adjacent cells: {coord_list_text(table_neighbors)}",
        f"Pick legal initially: {'yes' if start in printer_neighbors else 'no'}",
        "Drop legal initially: no",
        f"Legal movement actions from agent: {', '.join(legal_moves) if legal_moves else 'none'}",
        "Grid legend: A=agent, P=printer, T=table, .=free floor",
        "Grid:",
        *grid,
        "",
        "Teacher reasoning focus:",
        "- Table cells are always blocked.",
        "- Before PICK, the printer cell is blocked; after PICK, the printer is removed and that cell becomes traversable.",
        "- First move to any printer-adjacent cell and execute PICK.",
        "- Then move to any table-adjacent cell and execute DROP.",
    ]
    if include_teacher_ground_truth:
        lines.append(f"Reference action plan: {answer}")
    lines.append("Solve the task and end with exactly one <answer>...</answer> block.")
    return "\n".join(lines)


def teacher_context(
    task: str,
    metadata: dict[str, str],
    blocks: dict[str, list[str]],
    answer: str,
    include_teacher_ground_truth: bool = True,
) -> str:
    if task == "frozenlake":
        return frozenlake_context(metadata, blocks, answer, include_teacher_ground_truth)
    if task == "maze":
        return maze_context(metadata, blocks, answer, include_teacher_ground_truth)
    if task == "minibehaviour":
        return minibehaviour_context(metadata, blocks, answer, include_teacher_ground_truth)
    raise ValueError(f"Unsupported task: {task}")


def build_example(
    task: str,
    image_path: Path,
    text_path: Path,
    student_prompts: dict[str, str],
    teacher_prompts: dict[str, str],
    include_teacher_ground_truth: bool = True,
    path_root: Path = DEFAULT_REPO_ROOT,
) -> dict:
    metadata, blocks, task_text = read_task_text(text_path)
    raw_task = metadata["task"].lower()
    if raw_task == "minibehavior":
        raw_task = "minibehaviour"
    if raw_task != task:
        raise ValueError(f"Task mismatch for {text_path}: expected {task}, got {metadata['task']}")
    answer = answer_from_metadata(metadata, task)
    return {
        "sample_id": f"{task}_{text_path.parent.parent.name}_{text_path.stem}",
        "task": task,
        "level": int(metadata["level"]),
        "messages": [
            {"role": "user", "content": student_prompts[task]},
            {"role": "assistant", "content": f"<answer>{answer}</answer>"},
        ],
        "images": [portable_path(image_path, path_root)],
        "teacher_images": [],
        "vsp_task_text": task_text,
        "teacher_system_prompt": teacher_prompts[task],
        "teacher_text_context": teacher_context(task, metadata, blocks, answer, include_teacher_ground_truth),
    }


def collect_level(
    task_root: Path,
    task: str,
    level_name: str,
    limit_per_level: int | None,
    sample_per_level: int | None,
    rng: random.Random,
    student_prompts: dict[str, str],
    teacher_prompts: dict[str, str],
    include_teacher_ground_truth: bool,
    path_root: Path,
) -> list[dict]:
    text_dir = task_root / level_name / "text"
    image_dir = task_root / level_name / "image"
    if not text_dir.exists() or not image_dir.exists():
        raise FileNotFoundError(f"Missing image/text folders under {task_root / level_name}")
    text_paths = sorted(text_dir.glob("*.txt"))
    if sample_per_level is not None:
        if sample_per_level > len(text_paths):
            raise ValueError(f"Cannot sample {sample_per_level} rows from {text_dir}; only {len(text_paths)} exist.")
        text_paths = sorted(rng.sample(text_paths, sample_per_level))
    elif limit_per_level is not None:
        text_paths = text_paths[:limit_per_level]

    rows = []
    for text_path in text_paths:
        image_path = image_dir / f"{text_path.stem}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {text_path}: {image_path}")
        rows.append(
            build_example(
                task,
                image_path,
                text_path,
                student_prompts,
                teacher_prompts,
                include_teacher_ground_truth,
                path_root=path_root,
            )
        )
    return rows


def collect_rows(
    data_root: Path,
    levels: dict[str, list[str]],
    limit_per_level: int | None,
    sample_per_level: int | None,
    seed: int,
    student_prompts: dict[str, str],
    teacher_prompts: dict[str, str],
    include_teacher_ground_truth: bool,
    path_root: Path,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for task, task_dir_name in TASK_DIRS.items():
        task_root = data_root / task_dir_name
        for level_name in levels[task]:
            rows.extend(
                collect_level(
                    task_root,
                    task,
                    level_name,
                    limit_per_level,
                    sample_per_level,
                    rng,
                    student_prompts,
                    teacher_prompts,
                    include_teacher_ground_truth,
                    path_root,
                )
            )
    return rows


def save_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(DEFAULT_REPO_ROOT / "data" / "VSP-Tasks"))
    parser.add_argument("--test_data_root", default=str(DEFAULT_REPO_ROOT / "data" / "VSP-Tasks-OPCD-Val"))
    parser.add_argument("--train_save_path", required=True)
    parser.add_argument("--test_save_path", required=True)
    parser.add_argument(
        "--path_root",
        default=str(DEFAULT_REPO_ROOT),
        help="Root used to write portable relative image paths.",
    )
    parser.add_argument("--frozenlake_train_levels", default="3,4,5,6,7,8")
    parser.add_argument("--maze_train_levels", default="3,4,5,6")
    parser.add_argument("--minibehaviour_train_levels", default="5,6")
    parser.add_argument("--frozenlake_test_levels", default="3,4,5,6,7,8")
    parser.add_argument("--maze_test_levels", default="3,4,5,6")
    parser.add_argument("--minibehaviour_test_levels", default="5,6")
    parser.add_argument("--limit_per_level", type=int, default=None)
    parser.add_argument("--sample_per_level", type=int, default=None)
    parser.add_argument("--test_sample_per_level", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--frozenlake_student_prompt", default=str(DEFAULT_STUDENT_PROMPT_PATHS["frozenlake"]))
    parser.add_argument("--maze_student_prompt", default=str(DEFAULT_STUDENT_PROMPT_PATHS["maze"]))
    parser.add_argument("--minibehaviour_student_prompt", default=str(DEFAULT_STUDENT_PROMPT_PATHS["minibehaviour"]))
    parser.add_argument("--frozenlake_teacher_prompt", default=str(DEFAULT_TEACHER_PROMPT_PATHS["frozenlake"]))
    parser.add_argument("--maze_teacher_prompt", default=str(DEFAULT_TEACHER_PROMPT_PATHS["maze"]))
    parser.add_argument("--minibehaviour_teacher_prompt", default=str(DEFAULT_TEACHER_PROMPT_PATHS["minibehaviour"]))
    parser.add_argument(
        "--teacher_include_ground_truth",
        type=str_to_bool,
        default=True,
        help="Whether teacher_text_context includes the reference action plan. Defaults to true for pipeline compatibility.",
    )
    args = parser.parse_args()

    student_prompts = load_student_prompts(
        {
            "frozenlake": Path(args.frozenlake_student_prompt).expanduser().resolve(),
            "maze": Path(args.maze_student_prompt).expanduser().resolve(),
            "minibehaviour": Path(args.minibehaviour_student_prompt).expanduser().resolve(),
        }
    )
    teacher_prompts = load_teacher_prompts(
        {
            "frozenlake": Path(args.frozenlake_teacher_prompt).expanduser().resolve(),
            "maze": Path(args.maze_teacher_prompt).expanduser().resolve(),
            "minibehaviour": Path(args.minibehaviour_teacher_prompt).expanduser().resolve(),
        }
    )

    train_levels = {
        "frozenlake": parse_level_names(args.frozenlake_train_levels),
        "maze": parse_level_names(args.maze_train_levels),
        "minibehaviour": parse_level_names(args.minibehaviour_train_levels),
    }
    test_levels = {
        "frozenlake": parse_level_names(args.frozenlake_test_levels),
        "maze": parse_level_names(args.maze_test_levels),
        "minibehaviour": parse_level_names(args.minibehaviour_test_levels),
    }

    train_rows = collect_rows(
        Path(args.data_root).expanduser().resolve(),
        train_levels,
        limit_per_level=args.limit_per_level,
        sample_per_level=args.sample_per_level,
        seed=args.seed,
        student_prompts=student_prompts,
        teacher_prompts=teacher_prompts,
        include_teacher_ground_truth=args.teacher_include_ground_truth,
        path_root=Path(args.path_root).expanduser().resolve(),
    )
    test_rows = collect_rows(
        Path(args.test_data_root).expanduser().resolve(),
        test_levels,
        limit_per_level=args.limit_per_level,
        sample_per_level=args.test_sample_per_level,
        seed=args.seed + 1,
        student_prompts=student_prompts,
        teacher_prompts=teacher_prompts,
        include_teacher_ground_truth=args.teacher_include_ground_truth,
        path_root=Path(args.path_root).expanduser().resolve(),
    )

    if args.shuffle_train:
        random.Random(args.seed).shuffle(train_rows)

    save_json(Path(args.train_save_path).expanduser().resolve(), train_rows)
    save_json(Path(args.test_save_path).expanduser().resolve(), test_rows)
    print(f"Saved train JSON: {args.train_save_path} ({len(train_rows)} samples)")
    print(f"Saved test JSON: {args.test_save_path} ({len(test_rows)} samples)")


if __name__ == "__main__":
    main()

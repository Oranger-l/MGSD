from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

import gymnasium.envs.toy_text.frozen_lake as frozen_lake


GRID_COLOR = (180, 200, 230)


@lru_cache(maxsize=1)
def _image_dir() -> Path:
    return Path(frozen_lake.__file__).resolve().parent / "img"


@lru_cache(maxsize=None)
def _load_tile_image(name: str, cell_size: int) -> Image.Image:
    image_path = _image_dir() / name
    image = Image.open(image_path).convert("RGBA")
    return image.resize((cell_size, cell_size), resample=Image.Resampling.LANCZOS)


def render_desc_to_image(
    desc: Sequence[str] | Sequence[Sequence[str]],
    cell_size: int = 64,
    agent_state: int | None = None,
    last_action: int = 1,
) -> Image.Image:
    rows = ["".join(row) if not isinstance(row, str) else row for row in desc]
    nrow = len(rows)
    ncol = len(rows[0])
    canvas = Image.new("RGBA", (ncol * cell_size, nrow * cell_size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    ice_img = _load_tile_image("ice.png", cell_size)
    hole_img = _load_tile_image("hole.png", cell_size)
    cracked_hole_img = _load_tile_image("cracked_hole.png", cell_size)
    goal_img = _load_tile_image("goal.png", cell_size)
    start_img = _load_tile_image("stool.png", cell_size)
    elf_names = ["elf_left.png", "elf_down.png", "elf_right.png", "elf_up.png"]
    elf_img = _load_tile_image(elf_names[last_action], cell_size)

    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            pos = (x * cell_size, y * cell_size)
            canvas.alpha_composite(ice_img, pos)
            if tile == "H":
                canvas.alpha_composite(hole_img, pos)
            elif tile == "G":
                canvas.alpha_composite(goal_img, pos)
            elif tile == "S":
                canvas.alpha_composite(start_img, pos)
            draw.rectangle(
                (pos[0], pos[1], pos[0] + cell_size - 1, pos[1] + cell_size - 1),
                outline=GRID_COLOR,
                width=1,
            )

    if agent_state is None:
        start_row = start_col = None
        for row_idx, row in enumerate(rows):
            col_idx = row.find("S")
            if col_idx != -1:
                start_row, start_col = row_idx, col_idx
                break
        if start_row is None or start_col is None:
            agent_state = 0
        else:
            agent_state = start_row * ncol + start_col

    bot_row, bot_col = divmod(agent_state, ncol)
    bot_pos = (bot_col * cell_size, bot_row * cell_size)
    if rows[bot_row][bot_col] == "H":
        canvas.alpha_composite(cracked_hole_img, bot_pos)
    else:
        canvas.alpha_composite(elf_img, bot_pos)

    return canvas.convert("RGB")

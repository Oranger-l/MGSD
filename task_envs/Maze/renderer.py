from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw


Grid = Sequence[Sequence[dict[str, bool]]]


def render_maze(
    grid: Grid,
    start: tuple[int, int],
    target: tuple[int, int],
    image_size: int = 256,
) -> Image.Image:
    n = len(grid)
    img = Image.new("RGB", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)
    cell_size = float(image_size) / n
    wall_width = max(2, int(cell_size / 4.0))
    half_wall = wall_width / 2.0
    grid_width = max(1, int(cell_size / 16.0))

    for row in range(n):
        for col in range(n):
            x1 = col * cell_size + half_wall
            y1 = row * cell_size + half_wall
            x2 = (col + 1) * cell_size - half_wall
            y2 = (row + 1) * cell_size - half_wall
            draw.rectangle([(x1, y1), (x2, y2)], fill="white")
            cell = grid[row][col]
            if not cell["S"] and row < n - 1:
                draw.rectangle([(x1, y2), (x2, y2 + wall_width)], fill="white")
            if not cell["E"] and col < n - 1:
                draw.rectangle([(x2, y1), (x2 + wall_width, y2)], fill="white")

    grid_color = (224, 224, 224)
    for row in range(n):
        for col in range(n):
            if row < n - 1 and not grid[row][col]["S"]:
                y = (row + 1) * cell_size
                x1 = col * cell_size + half_wall
                x2 = (col + 1) * cell_size - half_wall
                draw.line([(x1, y), (x2, y)], fill=grid_color, width=grid_width)
            if col < n - 1 and not grid[row][col]["E"]:
                x = (col + 1) * cell_size
                y1 = row * cell_size + half_wall
                y2 = (row + 1) * cell_size - half_wall
                draw.line([(x, y1), (x, y2)], fill=grid_color, width=grid_width)

    def draw_dot(coord: tuple[int, int], color: str) -> None:
        row, col = coord
        cx = col * cell_size + cell_size / 2.0
        cy = row * cell_size + cell_size / 2.0
        radius = max(3, int((cell_size - wall_width) * 0.25))
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

    draw_dot(start, "yellow")
    draw_dot(target, "blue")
    return img

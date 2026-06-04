from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw


Coord = tuple[int, int]

CANVAS_GRAY = (96, 96, 96)
FREE_CELL = (0, 0, 0)
GRID_LINE = (60, 60, 60)
TABLE_COLOR = (181, 143, 96)
AGENT_COLOR = (255, 24, 16)
PRINTER_COLOR = (255, 255, 255)
PRINTER_DETAIL = (0, 0, 0)


def _cell_box(level: int, coord: Coord, image_size: int) -> tuple[float, float, float, float]:
    grid_n = level
    margin = 0.0
    board_size = float(image_size)
    cell_size = board_size / grid_n
    row, col = coord
    x1 = margin + col * cell_size
    y1 = margin + row * cell_size
    x2 = margin + (col + 1) * cell_size
    y2 = margin + (row + 1) * cell_size
    return x1, y1, x2, y2


def _draw_agent_arrow(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    pad_x = width * 0.22
    pad_y = height * 0.20
    points = [
        (x2 - pad_x, (y1 + y2) / 2.0),
        (x1 + pad_x, y1 + pad_y),
        (x1 + pad_x, y2 - pad_y),
    ]
    draw.polygon(points, fill=AGENT_COLOR)


def _draw_printer_icon(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = box
    cell_size = min(x2 - x1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    icon_w = cell_size * 0.58
    icon_h = cell_size * 0.52
    left = cx - icon_w / 2.0
    top = cy - icon_h / 2.0
    right = cx + icon_w / 2.0
    bottom = cy + icon_h / 2.0
    radius = cell_size * 0.04

    paper_top = top
    paper_bottom = top + icon_h * 0.34
    body_top = top + icon_h * 0.26
    body_bottom = top + icon_h * 0.72
    tray_top = top + icon_h * 0.56
    tray_bottom = bottom

    draw.rectangle([left + icon_w * 0.20, paper_top, right - icon_w * 0.20, paper_bottom], fill=PRINTER_COLOR)
    draw.rounded_rectangle([left, body_top, right, body_bottom], radius=radius, fill=PRINTER_COLOR)
    draw.rectangle([left + icon_w * 0.18, tray_top, right - icon_w * 0.18, tray_bottom], fill=PRINTER_COLOR)

    slot_y = body_top + (body_bottom - body_top) * 0.55
    draw.rounded_rectangle(
        [
            left + icon_w * 0.22,
            slot_y - cell_size * 0.025,
            right - icon_w * 0.12,
            slot_y + cell_size * 0.025,
        ],
        radius=radius,
        fill=PRINTER_DETAIL,
    )
    light_r = cell_size * 0.035
    light_cx = left + icon_w * 0.18
    light_cy = body_top + (body_bottom - body_top) * 0.36
    draw.ellipse(
        [light_cx - light_r, light_cy - light_r, light_cx + light_r, light_cy + light_r],
        fill=PRINTER_DETAIL,
    )


def render_minibehaviour(
    level: int,
    agent_pos: Coord,
    printer_pos: Coord,
    table_pos: Sequence[Coord],
    image_size: int = 256,
) -> Image.Image:
    grid_n = level
    margin = 0.0
    board_size = float(image_size)
    cell_size = board_size / grid_n

    image = Image.new("RGB", (image_size, image_size), CANVAS_GRAY)
    draw = ImageDraw.Draw(image)
    draw.rectangle([margin, margin, margin + board_size, margin + board_size], fill=FREE_CELL)

    for row, col in table_pos:
        x1, y1, x2, y2 = _cell_box(level, (row, col), image_size)
        draw.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], fill=TABLE_COLOR)

    _draw_printer_icon(draw, _cell_box(level, printer_pos, image_size))
    _draw_agent_arrow(draw, _cell_box(level, agent_pos, image_size))

    for idx in range(grid_n + 1):
        pos = margin + idx * cell_size
        draw.line([(margin, pos), (margin + board_size, pos)], fill=GRID_LINE, width=1)
        draw.line([(pos, margin), (pos, margin + board_size)], fill=GRID_LINE, width=1)

    return image

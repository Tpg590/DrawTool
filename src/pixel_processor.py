"""Deterministic pixel-art processing (background removal, NEAREST resize, spritesheet slicing, tile seams, quantization)."""

from __future__ import annotations

import math
import re
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse WxH pixel dimensions (e.g. '32x32')."""
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", size_str)
    if not match:
        raise ValueError(f"Invalid --size '{size_str}'. Expected WxH (e.g. 32x32).")
    w, h = int(match.group(1)), int(match.group(2))
    if w < 4 or h < 4 or w > 4096 or h > 4096:
        raise ValueError(f"Size {w}x{h} out of valid range (4..4096).")
    return w, h


def _parse_hex_token(token: str) -> tuple[int, int, int]:
    match = HEX_COLOR_RE.fullmatch(token.strip())
    if not match:
        raise ValueError(f"Invalid hex color '{token}'.")
    hex_digits = match.group(1)
    if len(hex_digits) == 3:
        hex_digits = "".join(ch * 2 for ch in hex_digits)
    return (
        int(hex_digits[0:2], 16),
        int(hex_digits[2:4], 16),
        int(hex_digits[4:6], 16),
    )


def parse_palette(palette_arg: str) -> list[tuple[int, int, int]]:
    """Parse palette from a .gpl file, a text file of hex codes, or a comma-separated hex string."""
    candidate = Path(palette_arg)
    colors: list[tuple[int, int, int]] = []

    if candidate.is_file():
        lines = candidate.read_text(encoding="utf-8").splitlines()
        if candidate.suffix.lower() == ".gpl":
            for line in lines:
                stripped = line.strip()
                if (
                    not stripped
                    or stripped.startswith("#")
                    or stripped == "GIMP Palette"
                    or stripped.startswith("Name:")
                    or stripped.startswith("Columns:")
                ):
                    continue
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                    except ValueError:
                        continue
                    if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                        colors.append((r, g, b))
        else:
            for line in lines:
                stripped = line.split("#", 1)[0].strip() if not line.strip().startswith("#") else line.strip()
                # Handle lines like "#ff0000" vs comments starting with "#"
                raw_line = line.strip()
                if not raw_line:
                    continue
                if raw_line.startswith("#") and not HEX_COLOR_RE.fullmatch(raw_line.split()[0]):
                    continue
                for token in re.split(r"[\s,;]+", raw_line):
                    if token and HEX_COLOR_RE.fullmatch(token):
                        colors.append(_parse_hex_token(token))
    else:
        for token in re.split(r"[\s,;]+", palette_arg.strip()):
            if token:
                colors.append(_parse_hex_token(token))

    # Deduplicate while preserving order
    unique_colors = list(dict.fromkeys(colors))
    if not unique_colors:
        raise ValueError(f"No valid colors found in palette '{palette_arg}'.")
    return unique_colors


def _detect_border_color(arr: np.ndarray) -> np.ndarray:
    h, w, _ = arr.shape
    border_pixels = np.concatenate(
        [
            arr[0, :, :3],
            arr[h - 1, :, :3],
            arr[:, 0, :3],
            arr[:, w - 1, :3],
        ],
        axis=0,
    )
    # Quantize to 8-step buckets to find the dominant solid backdrop robustly
    buckets = (border_pixels // 8) * 8
    packed = (
        buckets[:, 0].astype(np.int32) << 16
    ) | (buckets[:, 1].astype(np.int32) << 8) | buckets[:, 2].astype(np.int32)
    unique_vals, counts = np.unique(packed, return_counts=True)
    dominant = unique_vals[int(np.argmax(counts))]
    mask = packed == dominant
    return np.median(border_pixels[mask], axis=0).astype(np.float32)


def remove_background(img: Image.Image, tolerance: int = 28) -> Image.Image:
    """Remove exterior background using border-connected flood fill while preserving interior colors."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    h, w, _ = arr.shape
    if h == 0 or w == 0:
        return rgba

    border_alpha = np.concatenate(
        [arr[0, :, 3], arr[h - 1, :, 3], arr[:, 0, 3], arr[:, w - 1, 3]]
    )
    if float(np.mean(border_alpha < 250)) > 0.05:
        out = arr.copy()
        out[out[:, :, 3] < 16, 3] = 0
        return Image.fromarray(out, mode="RGBA")

    bg_rgb = _detect_border_color(arr)
    rgb = arr[:, :, :3].astype(np.float32)
    dist = np.linalg.norm(rgb - bg_rgb, axis=2)
    is_bg_candidate = dist <= float(tolerance)

    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def try_seed(y: int, x: int) -> None:
        if not visited[y, x] and is_bg_candidate[y, x]:
            visited[y, x] = True
            queue.append((y, x))

    for x in range(w):
        try_seed(0, x)
        try_seed(h - 1, x)
    for y in range(1, h - 1):
        try_seed(y, 0)
        try_seed(y, w - 1)

    while queue:
        cy, cx = queue.popleft()
        if cy > 0 and not visited[cy - 1, cx] and is_bg_candidate[cy - 1, cx]:
            visited[cy - 1, cx] = True
            queue.append((cy - 1, cx))
        if cy + 1 < h and not visited[cy + 1, cx] and is_bg_candidate[cy + 1, cx]:
            visited[cy + 1, cx] = True
            queue.append((cy + 1, cx))
        if cx > 0 and not visited[cy, cx - 1] and is_bg_candidate[cy, cx - 1]:
            visited[cy, cx - 1] = True
            queue.append((cy, cx - 1))
        if cx + 1 < w and not visited[cy, cx + 1] and is_bg_candidate[cy, cx + 1]:
            visited[cy, cx + 1] = True
            queue.append((cy, cx + 1))

    out = arr.copy()
    out[visited, 3] = 0
    out[visited, :3] = 0
    return Image.fromarray(out, mode="RGBA")


def quantize_image(
    img: Image.Image,
    palette: list[tuple[int, int, int]] | None = None,
    max_colors: int | None = None,
) -> Image.Image:
    """Snap alpha to crisp binary edges and quantize opaque pixels without dithering."""
    rgba = img.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    opaque_mask = arr[:, :, 3] >= 128

    out = np.zeros_like(arr, dtype=np.uint8)
    if not np.any(opaque_mask):
        return Image.fromarray(out, mode="RGBA")

    out[opaque_mask, 3] = 255
    opaque_rgb = arr[opaque_mask, :3]

    if palette:
        pal_arr = np.array(palette, dtype=np.float32)  # (K, 3)
        diffs = opaque_rgb.astype(np.float32)[:, None, :] - pal_arr[None, :, :]
        dists = np.sum(diffs * diffs, axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        out[opaque_mask, :3] = pal_arr[nearest_idx].astype(np.uint8)
        return Image.fromarray(out, mode="RGBA")

    if max_colors is not None and max_colors > 0:
        unique_rgb = np.unique(opaque_rgb, axis=0)
        if len(unique_rgb) > max_colors:
            # Quantize only opaque pixels in a packed 1xN strip so transparency never steals palette slots
            strip = Image.fromarray(opaque_rgb.reshape(1, -1, 3), mode="RGB")
            quantized_strip = strip.quantize(
                colors=max_colors,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            ).convert("RGB")
            mapped_rgb = np.array(quantized_strip, dtype=np.uint8).reshape(-1, 3)
            out[opaque_mask, :3] = mapped_rgb
            return Image.fromarray(out, mode="RGBA")

    out[opaque_mask, :3] = opaque_rgb
    return Image.fromarray(out, mode="RGBA")


def _compute_inner_box(target_size: tuple[int, int], pad: int) -> tuple[int, int]:
    tw, th = target_size
    effective_pad = pad if (tw > pad * 2 + 2 and th > pad * 2 + 2) else 0
    return max(1, tw - 2 * effective_pad), max(1, th - 2 * effective_pad)


def process_item(
    img: Image.Image,
    target_size: tuple[int, int],
    palette: list[tuple[int, int, int]] | None = None,
    max_colors: int | None = 16,
    pad: int = 1,
    bg_tolerance: int = 28,
) -> Image.Image:
    """Remove background, crop tightly, NEAREST-resize into centered canvas, and quantize."""
    cleaned = remove_background(img, tolerance=bg_tolerance)
    bbox = cleaned.getbbox()
    if bbox is None:
        return Image.new("RGBA", target_size, (0, 0, 0, 0))

    cropped = cleaned.crop(bbox)
    cw, ch = cropped.size
    inner_w, inner_h = _compute_inner_box(target_size, pad)
    scale = min(inner_w / cw, inner_h / ch)
    new_w = max(1, min(inner_w, int(round(cw * scale))))
    new_h = max(1, min(inner_h, int(round(ch * scale))))

    resized = cropped.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)
    return quantize_image(canvas, palette=palette, max_colors=max_colors)


def process_anim(
    img: Image.Image,
    target_size: tuple[int, int],
    frames: int,
    cols: int,
    palette: list[tuple[int, int, int]] | None = None,
    max_colors: int | None = 16,
    pad: int = 1,
    bg_tolerance: int = 28,
) -> tuple[Image.Image, list[Image.Image]]:
    """Slice spritesheet grid, scale frames consistently via NEAREST, stitch, and quantize uniformly."""
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}")

    rows = math.ceil(frames / cols)
    tw, th = target_size
    rgba = img.convert("RGBA")
    src_w, src_h = rgba.size

    cropped_frames: list[Image.Image | None] = []
    max_cw = 0
    max_ch = 0

    for idx in range(frames):
        col = idx % cols
        row = idx // cols
        x0 = round(col * src_w / cols)
        x1 = round((col + 1) * src_w / cols)
        y0 = round(row * src_h / rows)
        y1 = round((row + 1) * src_h / rows)
        cell = rgba.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)))
        cleaned = remove_background(cell, tolerance=bg_tolerance)
        bbox = cleaned.getbbox()
        if bbox is None:
            cropped_frames.append(None)
            continue
        cropped = cleaned.crop(bbox)
        cropped_frames.append(cropped)
        max_cw = max(max_cw, cropped.size[0])
        max_ch = max(max_ch, cropped.size[1])

    sheet = Image.new("RGBA", (cols * tw, rows * th), (0, 0, 0, 0))
    if max_cw == 0 or max_ch == 0:
        blank_frames = [Image.new("RGBA", target_size, (0, 0, 0, 0)) for _ in range(frames)]
        return sheet, blank_frames

    inner_w, inner_h = _compute_inner_box(target_size, pad)
    shared_scale = min(inner_w / max_cw, inner_h / max_ch)
    scaled_max_h = max(1, min(inner_h, int(round(max_ch * shared_scale))))
    anchor_bottom_y = (th - scaled_max_h) // 2 + scaled_max_h

    for idx, cropped in enumerate(cropped_frames):
        if cropped is None:
            continue
        cw, ch = cropped.size
        new_w = max(1, min(inner_w, int(round(cw * shared_scale))))
        new_h = max(1, min(inner_h, int(round(ch * shared_scale))))
        resized = cropped.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

        col = idx % cols
        row = idx // cols
        cell_x = col * tw + (tw - new_w) // 2
        cell_y = row * th + max(0, anchor_bottom_y - new_h)
        sheet.paste(resized, (cell_x, cell_y), resized)

    # Quantize full sheet at once so every animation frame shares the exact same palette
    quantized_sheet = quantize_image(sheet, palette=palette, max_colors=max_colors)
    frame_images: list[Image.Image] = []
    for idx in range(frames):
        col = idx % cols
        row = idx // cols
        frame_images.append(
            quantized_sheet.crop((col * tw, row * th, (col + 1) * tw, (row + 1) * th))
        )
    return quantized_sheet, frame_images


def process_tile(
    img: Image.Image,
    target_size: tuple[int, int],
    palette: list[tuple[int, int, int]] | None = None,
    max_colors: int | None = 16,
    seam_blend: int = 2,
) -> Image.Image:
    """NEAREST-resize tile, blend opposing borders toroidally to reduce tile seams, and quantize."""
    tw, th = target_size
    resized = img.convert("RGBA").resize((tw, th), resample=Image.Resampling.NEAREST)
    arr = np.array(resized, dtype=np.float32)

    blend_x = min(seam_blend, tw // 4) if seam_blend > 0 else 0
    blend_y = min(seam_blend, th // 4) if seam_blend > 0 else 0

    if blend_x > 0:
        orig = arr.copy()
        for d in range(blend_x):
            w_other = 0.5 * (blend_x - d) / float(blend_x)
            w_self = 1.0 - w_other
            arr[:, d] = w_self * orig[:, d] + w_other * orig[:, tw - 1 - d]
            arr[:, tw - 1 - d] = w_self * orig[:, tw - 1 - d] + w_other * orig[:, d]

    if blend_y > 0:
        orig = arr.copy()
        for d in range(blend_y):
            w_other = 0.5 * (blend_y - d) / float(blend_y)
            w_self = 1.0 - w_other
            arr[d, :] = w_self * orig[d, :] + w_other * orig[th - 1 - d, :]
            arr[th - 1 - d, :] = w_self * orig[th - 1 - d, :] + w_other * orig[d, :]

    blended = Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8), mode="RGBA")
    return quantize_image(blended, palette=palette, max_colors=max_colors)


def process_asset(
    img: Image.Image,
    asset_type: str,
    target_size: tuple[int, int],
    frames: int = 4,
    cols: int | None = None,
    palette: list[tuple[int, int, int]] | None = None,
    max_colors: int | None = 16,
    bg_tolerance: int = 28,
) -> tuple[Image.Image, dict[str, Any]]:
    """Process raw generated image according to asset_type and return (output_image, metadata)."""
    if asset_type == "item":
        out = process_item(
            img=img,
            target_size=target_size,
            palette=palette,
            max_colors=max_colors,
            bg_tolerance=bg_tolerance,
        )
        return out, {"cols": 1, "rows": 1, "frames": 1, "sheet_size": out.size}

    if asset_type == "tile":
        out = process_tile(
            img=img,
            target_size=target_size,
            palette=palette,
            max_colors=max_colors,
        )
        return out, {"cols": 1, "rows": 1, "frames": 1, "sheet_size": out.size}

    if asset_type == "anim":
        resolved_cols = frames if cols is None else cols
        rows = math.ceil(frames / resolved_cols)
        sheet, _ = process_anim(
            img=img,
            target_size=target_size,
            frames=frames,
            cols=resolved_cols,
            palette=palette,
            max_colors=max_colors,
            bg_tolerance=bg_tolerance,
        )
        return sheet, {
            "cols": resolved_cols,
            "rows": rows,
            "frames": frames,
            "sheet_size": sheet.size,
        }

    raise ValueError(f"Unsupported asset_type: {asset_type}")

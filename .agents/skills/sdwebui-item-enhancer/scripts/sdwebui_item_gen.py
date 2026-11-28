"""2-stage Paint-First -> Pixel-Craft SD WebUI Forge item generator for lush 16-bit RPG icons."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.image_gen import SDWebUIImageProvider

BG_COLOR = (0xD0, 0xD0, 0xD0)

ANTI_TILING_ITEM_NEGATIVES = (
    "sprite sheet, tileset, collage, grid, pattern, multiple items, multiple objects, "
    "icon set, border, frame, pedestal, ground, drop shadow, chibi, character, face, "
    "eyes, arms, legs, mascot, lowres, messy, square frame, UI box, card, badge, "
    "icon border, vignette, background scenery, landscape, sky, mountains, "
    "white border, sticker border, white outline"
)

ANTI_TILING_CHARACTER_NEGATIVES = (
    "sprite sheet, turnaround, character sheet, tileset, collage, grid, pattern, "
    "multiple characters, multiple views, border, frame, pedestal, ground, drop shadow, "
    "cast shadow, shadow under feet, painterly, oil painting, watercolor, semi-realistic, "
    "grimdark, dark fantasy cape, lowres, messy, square frame, UI box, card, badge, vignette, "
    "background scenery, landscape, sky, mountains, white border, sticker border, white outline, "
    "realistic proportions, tall realistic anatomy, portrait, bust, cropped legs, cut off feet"
)

WEAPON_KEYWORDS_RE = re.compile(
    r"\b(?:sword|blade|dagger|knife|staff|wand|spear|axe|bow|gun|rifle|pistol|shield|hammer|mace|scythe|katana|weapon|lance|club|holding)\b",
    flags=re.IGNORECASE,
)

NEGATIVE_PRESET_SECTIONS: dict[str, tuple[str, ...]] = {
    "item_icon": ("base", "cutout_clean", "item_general"),
    "beast_meat": ("base", "cutout_clean", "monster_material", "liquid_raw"),
    "potion": ("base", "cutout_clean", "potion_jar"),
    "character": ("base", "cutout_clean", "character_sprite"),
}

CHARACTER_FACING_PHRASES: dict[str, str] = {
    "front": "facing forward toward camera",
    "back": "facing directly away from camera, back view",
    "east": "facing right, 3/4 side view",
    "west": "facing left, 3/4 side view",
}


def parse_hex_color(value: str) -> tuple[int, int, int]:
    cleaned = value.strip().lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Expected 6-digit hex color, got '{value}'")
    return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))


def parse_negative_prompt_library(negative_file: Path) -> dict[str, str]:
    """Parse named sections from negative_prompt.md into {section_key: prompt_text}."""
    if not negative_file.is_file():
        return {}

    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    for raw_line in negative_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(">") or line.startswith("|"):
            continue
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            heading = line[3:].strip().lower()
            if heading.startswith("1.") or "base" in heading:
                current_key = "base"
            elif heading.startswith("2.") or "background" in heading or "cutout" in heading:
                current_key = "cutout_clean"
            else:
                # Section 3 is just a container for ### subsections; Section 4+ is the presets table
                current_key = None
            continue
        if line.startswith("### "):
            m = re.search(r"`([^`]+)`", line)
            current_key = m.group(1).strip() if m else None
            continue
        if current_key is not None:
            sections.setdefault(current_key, []).append(line)

    return {k: ", ".join(v) for k, v in sections.items() if v}


def _dedupe_tokens(text: str) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in text.split(","):
        clean = " ".join(token.strip().split())
        if not clean:
            continue
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(clean)
    return ", ".join(ordered)


def load_negative_prompt(negative_file: Path, preset: str = "item_icon") -> str:
    """Load negative prompt preset from negative_prompt.md and append anti-tiling negatives."""
    extra_anti = (
        ANTI_TILING_CHARACTER_NEGATIVES if preset == "character" else ANTI_TILING_ITEM_NEGATIVES
    )
    if not negative_file.is_file():
        return extra_anti

    sections = parse_negative_prompt_library(negative_file)
    if sections:
        keys = NEGATIVE_PRESET_SECTIONS.get(preset, NEGATIVE_PRESET_SECTIONS["item_icon"])
        parts = [sections[k] for k in keys if k in sections]
        parts.append(extra_anti)
        return _dedupe_tokens(", ".join(parts))

    # Fallback for plain-text negative prompt files
    base = " ".join(negative_file.read_text(encoding="utf-8").splitlines()).strip()
    return _dedupe_tokens(f"{base}, {extra_anti}") if base else extra_anti


def build_painted_sdxl_prompt(
    raw_prompt: str,
    base_negative: str,
    mode: str = "item",
    facing: str | None = None,
) -> tuple[str, str]:
    """Convert a pixel-art prompt into a lush 2D cel-shaded SDXL painting prompt."""
    extra_neg: list[str] = []

    def _collect_no(match: re.Match[str]) -> str:
        phrase = match.group(1).strip(" ,.")
        if phrase:
            extra_neg.append(phrase)
        return ""

    cleaned = re.sub(r"\bno\s+([^,]+)", _collect_no, raw_prompt, flags=re.IGNORECASE)

    if mode == "character":
        wants_weapon = bool(WEAPON_KEYWORDS_RE.search(raw_prompt))
        for pattern in (
            r"\bpixel\s*art\b",
            r"\b16-bit\b",
            r"\b8-bit\b",
            r"\b\d+\s*[xX]\s*\d+\b",
            r"\bclean\s+cel-shaded\s+pixel\s+blocks\b",
            r"\bclean\s+pixel\s+clusters\b",
            r"\btall\s+slender\s+build(?:\s+with)?\b",
            r"\bin\s+donghua\s+male\s+lead\s+style\b",
            r"\bdonghua\s+male\s+lead\s+style\b",
            r"\bsolid\s+(?:uniform\s+)?flat\s+light\s+gray\s+background\s*#?D0D0D0\b",
            r"#D0D0D0",
        ):
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        parts = [p.strip(" ,.") for p in cleaned.split(",") if p.strip(" ,.")]
        subject = ", ".join(parts)
        facing_key = (facing or "front").strip().lower()
        facing_phrase = CHARACTER_FACING_PHRASES.get(facing_key, "facing forward toward camera")
        unarmed_clause = "" if wants_weapon else "empty hands, unarmed, "
        eye_clause = "" if facing_key == "back" else "big expressive anime eyes, "
        if not wants_weapon:
            extra_neg.extend([
                "weapon", "sword", "blade", "dagger", "staff", "wand", "spear",
                "bow", "gun", "shield", "holding weapon", "holding sword",
                "holding item", "holding object",
            ])
        pos_final = (
            f"2d top-down 2.5D RPG chibi character illustration, single isolated full-body head-to-toe {subject}, "
            f"cute 2.5-head chibi proportion, {facing_phrase}, {unarmed_clause}standing cleanly with both shoes visible, "
            f"hand-painted anime cel shading, glossy hair highlights, {eye_clause}bold clean dark plum outline, "
            "clear readable silhouette, warm vibrant colors, "
            "floating centered on completely empty plain solid light gray background #D0D0D0"
        )
    else:
        # Strip pixel-art / chibi / icon / background boilerplate so Stage 1 paints clean die-cut art
        for pattern in (
            r"\bpixel\s*art\b",
            r"\b16-bit\b",
            r"\b8-bit\b",
            r"\b(?:retro\s+)?(?:RPG\s+)?(?:inventory\s+|game\s+|item\s+)*icon\b",
            r"\bclean\s+cel-shaded\s+pixel\s+blocks\b",
            r"\bclean\s+pixel\s+clusters\b",
            r"\bmatching\s+chibi\s+RPG\s+style\b",
            r"\bchibi\b",
            r"\bsolid\s+(?:uniform\s+)?flat\s+light\s+gray\s+background\s*#?D0D0D0\b",
            r"#D0D0D0",
        ):
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        parts = [p.strip(" ,.") for p in cleaned.split(",") if p.strip(" ,.")]
        subject = ", ".join(parts)

        pos_final = (
            f"2d fantasy game prop illustration, single isolated {subject}, "
            "hand-painted cel shading, bold clean dark plum outline, "
            "glossy specular highlights, warm vibrant colors, "
            "floating centered on completely empty plain solid light gray background #D0D0D0"
        )

    neg_final = base_negative
    if extra_neg:
        neg_final = _dedupe_tokens(f"{neg_final}, " + ", ".join(extra_neg))
    return pos_final, neg_final


def remove_bg_floodfill(
    img: Image.Image,
    tolerance: int = 34,
    remove_interior_holes: bool = True,
    interior_tolerance: int = 16,
) -> Image.Image:
    """Flood-fill exterior background and enclosed interior background holes to produce a clean RGBA foreground."""
    rgba = img.convert("RGBA")
    w, h = rgba.size
    px = rgba.load()

    corners = [px[0, 0][:3], px[w - 1, 0][:3], px[0, h - 1][:3], px[w - 1, h - 1][:3]]
    bg_r = sum(c[0] for c in corners) / 4.0
    bg_g = sum(c[1] for c in corners) / 4.0
    bg_b = sum(c[2] for c in corners) / 4.0

    def is_bg(rgb: tuple[int, int, int]) -> bool:
        return (
            abs(rgb[0] - bg_r) <= tolerance
            and abs(rgb[1] - bg_g) <= tolerance
            and abs(rgb[2] - bg_b) <= tolerance
        )

    visited = [[False] * w for _ in range(h)]
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        for y in (0, h - 1):
            if is_bg(px[x, y][:3]) and not visited[y][x]:
                visited[y][x] = True
                q.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if is_bg(px[x, y][:3]) and not visited[y][x]:
                visited[y][x] = True
                q.append((x, y))

    while q:
        cx, cy = q.popleft()
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx] and is_bg(px[nx, ny][:3]):
                visited[ny][nx] = True
                q.append((nx, ny))

    # Refine background RGB estimate from exterior visited pixels
    ext_r = ext_g = ext_b = ext_count = 0
    for y in range(h):
        for x in range(w):
            if visited[y][x]:
                r, g, b = px[x, y][:3]
                ext_r += r
                ext_g += g
                ext_b += b
                ext_count += 1
    if ext_count > 0:
        bg_r = ext_r / ext_count
        bg_g = ext_g / ext_count
        bg_b = ext_b / ext_count

    if remove_interior_holes:
        min_hole_area = max(16, int(w * h * 0.0001))

        def is_tight_bg(rgb: tuple[int, int, int]) -> bool:
            return (
                abs(rgb[0] - bg_r) <= interior_tolerance
                and abs(rgb[1] - bg_g) <= interior_tolerance
                and abs(rgb[2] - bg_b) <= interior_tolerance
            )

        checked = [[False] * w for _ in range(h)]
        fringe_q: deque[tuple[int, int]] = deque()
        for y in range(h):
            for x in range(w):
                if visited[y][x] or checked[y][x] or not is_tight_bg(px[x, y][:3]):
                    continue
                checked[y][x] = True
                comp: list[tuple[int, int]] = [(x, y)]
                comp_q: deque[tuple[int, int]] = deque([(x, y)])
                diff_sum = 0.0
                while comp_q:
                    ux, uy = comp_q.popleft()
                    ur, ug, ub = px[ux, uy][:3]
                    diff_sum += abs(ur - bg_r) + abs(ug - bg_g) + abs(ub - bg_b)
                    for nx, ny in ((ux - 1, uy), (ux + 1, uy), (ux, uy - 1), (ux, uy + 1)):
                        if (
                            0 <= nx < w
                            and 0 <= ny < h
                            and not visited[ny][nx]
                            and not checked[ny][nx]
                            and is_tight_bg(px[nx, ny][:3])
                        ):
                            checked[ny][nx] = True
                            comp.append((nx, ny))
                            comp_q.append((nx, ny))
                mean_diff = diff_sum / (3.0 * len(comp))
                if len(comp) >= min_hole_area and mean_diff <= interior_tolerance * 0.75:
                    for hx, hy in comp:
                        visited[hy][hx] = True
                        fringe_q.append((hx, hy))

        # Expand confirmed interior holes into anti-aliased gray fringes
        while fringe_q:
            cx, cy = fringe_q.popleft()
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx] and is_bg(px[nx, ny][:3]):
                    visited[ny][nx] = True
                    fringe_q.append((nx, ny))

    for y in range(h):
        for x in range(w):
            if visited[y][x]:
                px[x, y] = (0, 0, 0, 0)
    return rgba


def score_candidate(img: Image.Image) -> dict[str, Any]:
    """Score painted candidate for single-object framing, clean background, and vibrant color."""
    small = img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    px = small.load()
    w, h = small.size

    corners = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
    bg_r = sum(c[0] for c in corners) / 4.0
    bg_g = sum(c[1] for c in corners) / 4.0
    bg_b = sum(c[2] for c in corners) / 4.0

    def is_bg(rgb: tuple[int, int, int]) -> bool:
        return (abs(rgb[0] - bg_r) + abs(rgb[1] - bg_g) + abs(rgb[2] - bg_b)) <= 65

    visited_bg = [[False] * w for _ in range(h)]
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        for y in (0, h - 1):
            if is_bg(px[x, y]) and not visited_bg[y][x]:
                visited_bg[y][x] = True
                q.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if is_bg(px[x, y]) and not visited_bg[y][x]:
                visited_bg[y][x] = True
                q.append((x, y))

    while q:
        cx, cy = q.popleft()
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if 0 <= nx < w and 0 <= ny < h and not visited_bg[ny][nx] and is_bg(px[nx, ny]):
                visited_bg[ny][nx] = True
                q.append((nx, ny))

    fg_pixels = 0
    edge_touches = 0
    sat_sum = 0.0
    min_x, min_y, max_x, max_y = w, h, -1, -1
    for y in range(h):
        for x in range(w):
            if not visited_bg[y][x]:
                fg_pixels += 1
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                r, g, b = px[x, y]
                sat_sum += (max(r, g, b) - min(r, g, b)) / 255.0
                if x <= 1 or x >= w - 2 or y <= 1 or y >= h - 2:
                    edge_touches += 1

    fg_ratio = fg_pixels / float(w * h)
    avg_sat = sat_sum / max(1, fg_pixels)
    bbox_area = max(1, (max_x - min_x + 1) * (max_y - min_y + 1)) if max_x >= min_x else 1
    bbox_fill = fg_pixels / float(bbox_area)

    seen_fg = [[False] * w for _ in range(h)]
    major_components = 0
    for y in range(h):
        for x in range(w):
            if not visited_bg[y][x] and not seen_fg[y][x]:
                seen_fg[y][x] = True
                comp_q: deque[tuple[int, int]] = deque([(x, y)])
                size = 0
                while comp_q:
                    ux, uy = comp_q.popleft()
                    size += 1
                    for nx, ny in ((ux - 1, uy), (ux + 1, uy), (ux, uy - 1), (ux, uy + 1)):
                        if 0 <= nx < w and 0 <= ny < h and not visited_bg[ny][nx] and not seen_fg[ny][nx]:
                            seen_fg[ny][nx] = True
                            comp_q.append((nx, ny))
                if size >= 30:
                    major_components += 1

    score = 100.0 + min(10.0, avg_sat * 15.0)
    if major_components == 0:
        score -= 90.0
    elif major_components > 1:
        score -= 25.0 * (major_components - 1)
    if edge_touches > 4:
        score -= min(40.0, edge_touches * 1.5)
    if fg_ratio < 0.08 or fg_ratio > 0.65:
        score -= 25.0
    if bbox_fill > 0.82 and fg_ratio > 0.38:
        score -= 35.0

    return {
        "score": round(score, 2),
        "components": major_components,
        "fg_ratio": round(fg_ratio, 3),
        "bbox_fill": round(bbox_fill, 3),
        "avg_sat": round(avg_sat, 3),
        "edge_touches": edge_touches,
    }


def _build_palette_image(palette_ref: Image.Image) -> Image.Image:
    """Extract opaque RGB colors from a reference sprite into a 256-color PIL palette image."""
    rgba = palette_ref.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4)
    opaque = arr[arr[:, 3] >= 128][:, :3]
    if len(opaque) == 0:
        return Image.new("P", (1, 1))
    uniq = np.unique(opaque, axis=0)[:256]
    pal_img = Image.new("P", (1, 1))
    flat: list[int] = []
    for r, g, b in uniq:
        flat.extend([int(r), int(g), int(b)])
    while len(flat) < 256 * 3:
        flat.extend(flat[:3])
    pal_img.putpalette(flat[: 256 * 3])
    return pal_img


def pixel_craft(
    painted_img: Image.Image,
    pixel_size: int = 48,
    max_colors: int = 20,
    outline_rgb: tuple[int, int, int] = (0x34, 0x1C, 0x38),
    anchor_bottom: bool = False,
    transparent_bg: bool = True,
    palette_ref: Image.Image | None = None,
) -> Image.Image:
    """Convert painted SDXL asset into a crisp, vibrant pixel-art sprite."""
    rgba = remove_bg_floodfill(painted_img, tolerance=34)

    # Slight vibrancy, contrast & sharpness boost before pixelization
    rgb_boost = ImageEnhance.Color(rgba.convert("RGB")).enhance(1.15)
    rgb_boost = ImageEnhance.Contrast(rgb_boost).enhance(1.10)
    rgb_boost = ImageEnhance.Sharpness(rgb_boost).enhance(1.25)
    r_ch, g_ch, b_ch = rgb_boost.split()
    rgba = Image.merge("RGBA", (r_ch, g_ch, b_ch, rgba.split()[3]))

    bbox = rgba.getbbox()
    if bbox is None:
        return Image.new("RGB", (512, 512), BG_COLOR)

    cropped = rgba.crop(bbox)
    cw, ch = cropped.size
    side = max(cw, ch)
    # Leave margin for the 1px outline + breathing room (slightly tighter for 32x32 characters)
    inner_target = max(8, pixel_size - (3 if anchor_bottom else 4))
    scale = inner_target / float(side)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    # Downscale RGB with LANCZOS for rich interior shading, and Alpha cleanly
    small = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (pixel_size, pixel_size), (0, 0, 0, 0))
    ox = (pixel_size - new_w) // 2
    oy = max(1, pixel_size - 2 - new_h) if anchor_bottom else (pixel_size - new_h) // 2
    canvas.paste(small, (ox, oy))

    # Threshold alpha to crisp binary mask
    c_px = canvas.load()
    fg_mask = [[False] * pixel_size for _ in range(pixel_size)]
    for y in range(pixel_size):
        for x in range(pixel_size):
            r, g, b, a = c_px[x, y]
            if a >= 140:
                fg_mask[y][x] = True
                c_px[x, y] = (r, g, b, 255)
            else:
                c_px[x, y] = (0, 0, 0, 0)

    # Quantize foreground colors cleanly using MEDIANCUT (or optional palette_ref)
    rgb_only = Image.new("RGB", (pixel_size, pixel_size), BG_COLOR)
    r_px = rgb_only.load()
    for y in range(pixel_size):
        for x in range(pixel_size):
            if fg_mask[y][x]:
                r_px[x, y] = c_px[x, y][:3]

    if palette_ref is not None:
        pal_img = _build_palette_image(palette_ref)
        quantized = rgb_only.quantize(palette=pal_img, dither=Image.Dither.NONE).convert("RGB")
    else:
        quantized = rgb_only.quantize(colors=max(4, max_colors), method=Image.Quantize.MEDIANCUT).convert("RGB")
    q_px = quantized.load()

    # Composite onto transparent RGBA or flat #D0D0D0 with a crisp 1px warm plum-brown exterior outline
    if transparent_bg:
        out = Image.new("RGBA", (pixel_size, pixel_size), (0, 0, 0, 0))
        o_px = out.load()
        for y in range(pixel_size):
            for x in range(pixel_size):
                if fg_mask[y][x]:
                    o_px[x, y] = (*q_px[x, y], 255)
                else:
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= nx < pixel_size and 0 <= ny < pixel_size and fg_mask[ny][nx]:
                            o_px[x, y] = (*outline_rgb, 255)
                            break
    else:
        out = Image.new("RGB", (pixel_size, pixel_size), BG_COLOR)
        o_px = out.load()
        for y in range(pixel_size):
            for x in range(pixel_size):
                if fg_mask[y][x]:
                    o_px[x, y] = q_px[x, y]
                else:
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= nx < pixel_size and 0 <= ny < pixel_size and fg_mask[ny][nx]:
                            o_px[x, y] = outline_rgb
                            break

    return out.resize((512, 512), Image.Resampling.NEAREST)


def save_contact_sheet(
    painted_list: list[Image.Image],
    metrics: list[dict[str, Any]],
    pixel_preview: Image.Image,
    out_path: Path,
) -> None:
    """Save contact sheet showing Stage 1 painted candidates + final Stage 2 pixel-art result."""
    thumb = 256
    pad = 12
    total = len(painted_list) + 1
    w = total * (thumb + pad) + pad
    h = thumb + 48
    sheet = Image.new("RGB", (w, h), (34, 36, 44))
    draw = ImageDraw.Draw(sheet)

    for idx, (img, m) in enumerate(zip(painted_list, metrics)):
        x0 = pad + idx * (thumb + pad)
        sheet.paste(img.resize((thumb, thumb), Image.Resampling.BILINEAR), (x0, pad))
        draw.text((x0 + 4, thumb + pad + 6), f"Paint #{idx + 1} s={m['score']}", fill=(235, 240, 248))

    x_final = pad + len(painted_list) * (thumb + pad)
    preview_resized = pixel_preview.resize((thumb, thumb), Image.Resampling.NEAREST)
    if preview_resized.mode == "RGBA":
        tile = Image.new("RGB", (thumb, thumb), BG_COLOR)
        tile.paste(preview_resized, (0, 0), preview_resized)
        preview_resized = tile
    sheet.paste(preview_resized, (x_final, pad))
    draw.text((x_final + 4, thumb + pad + 6), "Final Pixel-Craft", fill=(140, 245, 190))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2-stage Paint-First -> Pixel-Craft SD WebUI item & character generator"
    )
    parser.add_argument("--prompt", required=True, help="Raw positive prompt")
    parser.add_argument("--name", required=True, help="Output base name")
    parser.add_argument("--mode", choices=("item", "character"), default="item", help="Asset mode: item | character")
    parser.add_argument(
        "--facing",
        choices=("front", "back", "east", "west"),
        default=None,
        help="Optional facing direction for character mode",
    )
    parser.add_argument(
        "--negative-preset",
        choices=("auto", "item_icon", "beast_meat", "potion", "character"),
        default="auto",
        help="Negative prompt preset from negative_prompt.md (default: auto by --mode)",
    )
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--negative-file", default="negative_prompt.md", help="Path to negative prompt file")
    parser.add_argument("--candidates", type=int, default=3, help="Number of Stage 1 candidates")
    parser.add_argument(
        "--pixel-size",
        type=int,
        default=None,
        help="Logical pixel-art grid size (default: 64 for character, 48 for item)",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=None,
        help="Max quantized colors (default: 32 for character, 20 for item)",
    )
    parser.add_argument("--outline-color", default="#341C38", help="Warm dark outline hex color")
    parser.add_argument("--palette-from", default="", help="Optional reference PNG to lock quantized colors")
    parser.add_argument("--guide", default="", help="Optional guide PNG for img2img")
    parser.add_argument("--denoise", type=float, default=0.62, help="Denoising strength if --guide is set")
    parser.add_argument("--lora-weight", type=float, default=0.0, help="Optional LoRA weight (0.0 disables LoRA)")
    parser.add_argument("--steps", type=int, default=28, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--sampler", default="DPM++ 2M", help="Sampler name")
    args = parser.parse_args()

    pixel_size = args.pixel_size if args.pixel_size is not None else (64 if args.mode == "character" else 48)
    max_colors = args.max_colors if args.max_colors is not None else (32 if args.mode == "character" else 20)
    neg_preset = (
        ("character" if args.mode == "character" else "item_icon")
        if args.negative_preset == "auto"
        else args.negative_preset
    )

    cfg = load_config()
    base_neg = load_negative_prompt(Path(args.negative_file), preset=neg_preset)
    pos_prompt, neg_prompt = build_painted_sdxl_prompt(
        args.prompt,
        base_neg,
        mode=args.mode,
        facing=args.facing,
    )

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_path = Path(args.guide) if args.guide else None

    lora_name = cfg.sdwebui_lora if args.lora_weight > 0.0 else "none"
    provider = SDWebUIImageProvider(
        api_url=cfg.image_api_url,
        model=cfg.image_model,
        lora=lora_name,
        lora_weight=args.lora_weight,
        steps=args.steps,
        cfg_scale=args.cfg,
        sampler=args.sampler,
        denoising=args.denoise,
        negative_prompt=neg_prompt,
        timeout=cfg.request_timeout,
    )

    cand_images: list[Image.Image] = []
    cand_bytes: list[bytes] = []
    cand_metrics: list[dict[str, Any]] = []

    for _ in range(max(1, args.candidates)):
        raw_bytes = provider.generate(
            pos_prompt,
            gen_size=(1024, 1024),
            target_size=(pixel_size, pixel_size),
            ref_path=ref_path,
        )
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        cand_bytes.append(raw_bytes)
        cand_images.append(img)
        cand_metrics.append(score_candidate(img))

    best_idx = max(range(len(cand_metrics)), key=lambda idx: cand_metrics[idx]["score"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    painted_path = out_dir / f"{args.name}_painted.png"
    painted_path.write_bytes(cand_bytes[best_idx])

    outline_rgb = parse_hex_color(args.outline_color)
    palette_ref_img = Image.open(args.palette_from) if args.palette_from else None
    pixel_img = pixel_craft(
        cand_images[best_idx],
        pixel_size=pixel_size,
        max_colors=max_colors,
        outline_rgb=outline_rgb,
        anchor_bottom=(args.mode == "character"),
        palette_ref=palette_ref_img,
    )
    raw_path = out_dir / f"{args.name}_raw.png"
    pixel_img.save(raw_path, format="PNG")

    grid_path = cache_dir / f"{args.name}_grid.png"
    save_contact_sheet(cand_images, cand_metrics, pixel_img, grid_path)

    result = {
        "best_index": best_idx + 1,
        "mode": args.mode,
        "negative_preset": neg_preset,
        "painted_path": str(painted_path.resolve()),
        "raw_pixel_path": str(raw_path.resolve()),
        "grid_path": str(grid_path.resolve()),
        "pixel_size": pixel_size,
        "max_colors": max_colors,
        "positive_prompt": pos_prompt,
        "scores": cand_metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

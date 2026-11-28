"""Prompt optimization module with locked 8-bit top-down defaults and minimal LLM token usage."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Any

from src.config import AppConfig

DEFAULT_STYLE = "8bit-topdown"

LOCKED_8BIT_TOPDOWN_KEYWORDS = (
    "8-bit pixel art, top-down game view, retro 8-bit aesthetic, crisp pixel edges, "
    "limited color palette, simple clean shading, no anti-aliasing, no 3D render, "
    "no text, no watermark"
)

STYLE_PRESETS: dict[str, str] = {
    "8bit-topdown": LOCKED_8BIT_TOPDOWN_KEYWORDS,
    "8bit-chibi-topdown": (
        "8-bit pixel art, top-down 2.5D 3/4 RPG game perspective, 2.5-head chibi character proportion, "
        "cute compact silhouette with readable head and eyes, crisp 1px dark outline, "
        "retro 8-bit limited 16-color palette, simple clean flat cel shading, "
        "crisp pixel edges, no anti-aliasing, no 3D render, no text, no watermark"
    ),
    "16bit-topdown": (
        "16-bit pixel art, top-down game view, retro SNES aesthetic, crisp pixel edges, "
        "limited color palette, clean cel shading, no anti-aliasing, no 3D render, "
        "no text, no watermark"
    ),
    "anime-rpg-topdown": (
        "16-bit anime RPG pixel art, top-down 3/4 RPG view, classic SNES/GBA JRPG character aesthetic, "
        "crisp 1px dark outline, expressive clear anime pixel features, rich limited color palette, "
        "clean cel shading, strong silhouette readability, no anti-aliasing, no 3D render, "
        "no text, no watermark"
    ),
    "hibit-anime-side": (
        "high-resolution 32-bit anime pixel art, 2D side-scroller action RPG sprite aesthetic "
        "(Castlevania / Metroidvania / modern hi-bit anime style), tall 4.5-head anime character proportion, "
        "crisp selective colored outlines (sel-out) with darker hue-shifted contour pixels, "
        "rich multi-tone cel shading with smooth hue-shifted pixel ramps, intricate hair and fabric flow, "
        "rich 48 to 64 color palette, true 1:1 crisp pixel edges, no anti-aliasing blur, "
        "no 3D render, no text, no watermark"
    ),
    "isometric-pixel": (
        "isometric pixel art, 2D retro game asset, crisp pixel edges, "
        "limited color palette, simple clean shading, no anti-aliasing, no 3D render, "
        "no text, no watermark"
    ),
}

VALID_ASSET_TYPES = ("item", "tile", "anim", "character")
VALID_FACINGS = ("east", "west", "front", "back")
FACING_DESCRIPTIONS: dict[str, str] = {
    "east": "strict 2D side-profile view facing right (east)",
    "west": "strict 2D side-profile view facing left (west)",
    "front": "facing straight forward toward camera",
    "back": "facing directly away from camera (back view)",
}


@dataclass(frozen=True)
class PromptResult:
    """Result of prompt optimization."""

    subject: str
    final_prompt: str
    style: str
    asset_type: str
    used_llm: bool
    cols: int
    rows: int
    facing: str | None = None


def normalize_facing(facing: str | None) -> str | None:
    """Validate and normalize optional facing direction."""
    if facing is None:
        return None
    cleaned = facing.strip().lower()
    if not cleaned:
        return None
    if cleaned not in VALID_FACINGS:
        raise ValueError(f"Unsupported facing '{facing}'. Expected one of {VALID_FACINGS}.")
    return cleaned


def compute_grid(frames: int, cols: int | None = None) -> tuple[int, int]:
    """Compute (cols, rows) spritesheet dimensions for the given frame count."""
    if frames < 1:
        raise ValueError(f"--frames must be >= 1, got {frames}")
    resolved_cols = frames if cols is None else cols
    if resolved_cols < 1:
        raise ValueError(f"--cols must be >= 1, got {resolved_cols}")
    rows = math.ceil(frames / resolved_cols)
    return resolved_cols, rows


def get_style_keywords(style: str) -> str:
    """Return locked keywords for a known style preset or construct safe fallback rules."""
    normalized = (style or DEFAULT_STYLE).strip()
    if normalized in STYLE_PRESETS:
        return STYLE_PRESETS[normalized]
    return f"{normalized}, crisp pixel edges, no anti-aliasing, no text, no watermark"


def build_type_layout_prompt(
    asset_type: str,
    size: tuple[int, int],
    frames: int = 4,
    cols: int | None = None,
    has_ref: bool = False,
    bg_color: str | None = None,
    action_notes: str | None = None,
    style: str = DEFAULT_STYLE,
    facing: str | None = None,
) -> str:
    """Build deterministic composition instructions tailored to item, tile, or anim."""
    if asset_type not in VALID_ASSET_TYPES:
        raise ValueError(f"Unsupported asset type '{asset_type}'. Expected one of {VALID_ASSET_TYPES}.")

    norm_facing = normalize_facing(facing)
    norm_style = (style or DEFAULT_STYLE).strip().lower()
    is_side = "side" in norm_style
    is_iso = "isometric" in norm_style

    w, h = size
    clean_bg = " ".join(bg_color.strip().split()) if bg_color else ""
    if asset_type == "item":
        bg_phrase = (
            f"solid uniform flat {clean_bg} background for clean background removal"
            if clean_bg
            else "solid uniform flat magenta or plain single-color background for clean background removal"
        )
        if is_side:
            layout = (
                f"single isolated full-body game sprite centered cleanly in frame, sized clearly for a {w}x{h} sprite, "
                f"2D side-view game perspective, {bg_phrase}, "
                "strong silhouette, crisp selective colored outlines, and detailed multi-tone pixel shading"
            )
        elif is_iso:
            layout = (
                f"single isolated game item centered cleanly in frame, sized clearly for a {w}x{h} sprite, "
                f"isometric 2D game perspective, {bg_phrase}, "
                "strong silhouette and crisp pixel shading"
            )
        elif norm_style == DEFAULT_STYLE:
            layout = (
                f"single isolated game item centered cleanly in frame, sized clearly for a {w}x{h} sprite, "
                f"top-down 2D game perspective, {bg_phrase}, "
                "strong silhouette and distinct 8-bit contrast shading"
            )
        else:
            layout = (
                f"single isolated game item centered cleanly in frame, sized clearly for a {w}x{h} sprite, "
                f"top-down 2D game perspective, {bg_phrase}, "
                "strong silhouette and crisp pixel shading"
            )
    elif asset_type == "tile":
        if is_side:
            perspective_tile = "2D side-scroller"
        elif is_iso:
            perspective_tile = "isometric"
        else:
            perspective_tile = "top-down"
        layout = (
            f"seamless tileable {perspective_tile} terrain texture filling the entire square edge-to-edge "
            f"for a {w}x{h} tile, uniform repeating micro-pattern, avoid isolated objects on borders "
            "so edges connect seamlessly without visible tile seams"
        )
    elif asset_type == "character":
        bg_phrase = (
            f"solid uniform flat {clean_bg} background for clean background removal"
            if clean_bg
            else "solid uniform flat magenta or plain single-color background for clean background removal"
        )
        if "chibi" in norm_style:
            layout = (
                f"single isolated full-body 2D chibi RPG character sprite centered cleanly in frame, "
                f"sized clearly for a {w}x{h} sprite, top-down 2.5D 3/4 RPG game perspective, "
                f"2.5-head chibi proportion, standing pose with feet planted cleanly shoulder-width apart, "
                f"{bg_phrase}, strong silhouette and distinct 8-bit contrast shading"
            )
        elif is_side:
            layout = (
                f"single isolated full-body 2D game character sprite centered cleanly in frame, "
                f"sized clearly for a {w}x{h} sprite, 2D side-view game perspective, "
                f"standing pose with feet planted cleanly, {bg_phrase}, "
                "strong silhouette, crisp selective colored outlines, and detailed multi-tone pixel shading"
            )
        elif is_iso:
            layout = (
                f"single isolated full-body 2D RPG character sprite centered cleanly in frame, "
                f"sized clearly for a {w}x{h} sprite, isometric 2D game perspective, "
                f"standing pose with feet planted cleanly, {bg_phrase}, "
                "strong silhouette and crisp pixel shading"
            )
        elif norm_style == DEFAULT_STYLE:
            layout = (
                f"single isolated full-body 2D RPG character sprite centered cleanly in frame, "
                f"sized clearly for a {w}x{h} sprite, top-down 2.5D 3/4 RPG game perspective, "
                f"standing pose with feet planted cleanly, {bg_phrase}, "
                "strong silhouette and distinct 8-bit contrast shading"
            )
        else:
            layout = (
                f"single isolated full-body 2D RPG character sprite centered cleanly in frame, "
                f"sized clearly for a {w}x{h} sprite, top-down 2.5D 3/4 RPG game perspective, "
                f"standing pose with feet planted cleanly, {bg_phrase}, "
                "strong silhouette and crisp pixel shading"
            )
    else:
        resolved_cols, rows = compute_grid(frames, cols)
        bg_phrase = (
            f"solid uniform flat {clean_bg} background for clean background removal"
            if clean_bg
            else "solid uniform flat single-color background"
        )
        if is_side:
            seq_perspective = "2D side-scroller readable motion sequence"
        elif is_iso:
            seq_perspective = "isometric RPG readable motion sequence"
        else:
            seq_perspective = "top-down RPG readable motion sequence"
        layout = (
            f"2D sprite sheet arranged in an exact {resolved_cols} columns by {rows} rows uniform grid "
            f"with {frames} animation frames ordered left-to-right top-to-bottom, each cell {w}x{h} proportion, "
            f"{seq_perspective}, consistent character proportions, outfit, and colors "
            f"across every frame, {bg_phrase}, no visible grid lines or borders"
        )
        if action_notes and action_notes.strip():
            clean_notes = " ".join(action_notes.strip().split())
            layout += f", frame sequence: {clean_notes}"

    if norm_facing:
        layout += f", {FACING_DESCRIPTIONS[norm_facing]}"
    if has_ref:
        layout += ", match the visual design, proportions, and color palette of the reference image"
    return layout


def build_deterministic_prompt(
    subject: str,
    asset_type: str,
    size: tuple[int, int],
    frames: int = 4,
    cols: int | None = None,
    style: str = DEFAULT_STYLE,
    has_ref: bool = False,
    bg_color: str | None = None,
    action_notes: str | None = None,
    facing: str | None = None,
) -> str:
    """Assemble the complete deterministic image prompt without spending LLM tokens."""
    clean_subject = " ".join(subject.strip().split())
    if not clean_subject:
        raise ValueError("Prompt cannot be empty.")

    norm_facing = normalize_facing(facing)
    layout = build_type_layout_prompt(
        asset_type=asset_type,
        size=size,
        frames=frames,
        cols=cols,
        has_ref=has_ref,
        bg_color=bg_color,
        action_notes=action_notes,
        style=style,
        facing=norm_facing,
    )
    style_keywords = get_style_keywords(style)
    return f"{clean_subject}, {layout}, {style_keywords}"


def _refine_subject_with_openai(
    user_prompt: str,
    asset_type: str,
    config: AppConfig,
) -> str:
    url = config.llm_api_url or "https://api.openai.com/v1/chat/completions"
    model = config.llm_model or "gpt-4o-mini"
    system_msg = (
        "You translate and condense a 2D game asset description into concise English visual phrasing "
        "(under 35 words). Describe only shape, colors, pose/surface details for a "
        f"{asset_type} asset. Never output pixel grids, hex codes, coordinates, or style buzzwords."
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 120,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.llm_api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.request_timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return " ".join(str(content).strip().split())


def _refine_subject_with_gemini(
    user_prompt: str,
    asset_type: str,
    config: AppConfig,
) -> str:
    model = config.llm_model or "gemini-2.5-flash"
    base_url = config.llm_api_url or (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )
    sep = "&" if "?" in base_url else "?"
    url = f"{base_url}{sep}key={config.llm_api_key}"
    prompt_text = (
        "Translate and condense this 2D game asset request into a concise English visual description "
        f"(under 35 words) for a {asset_type} asset. Describe only the object/terrain/action and colors. "
        "Never output pixel grids, coordinates, or style tags.\n\n"
        f"Request: {user_prompt}"
    )
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": 120, "temperature": 0.2},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.request_timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return " ".join(str(text).strip().split())


def optimize_prompt(
    user_prompt: str,
    asset_type: str,
    size: tuple[int, int],
    frames: int = 4,
    cols: int | None = None,
    style: str = DEFAULT_STYLE,
    has_ref: bool = False,
    config: AppConfig | None = None,
    use_llm: bool = True,
    bg_color: str | None = None,
    action_notes: str | None = None,
    facing: str | None = None,
) -> PromptResult:
    """Optimize prompt using optional tiny-token LLM translation + locked deterministic rules."""
    resolved_cols, rows = compute_grid(frames, cols) if asset_type == "anim" else (1, 1)
    norm_facing = normalize_facing(facing)
    subject = " ".join(user_prompt.strip().split())
    used_llm = False

    if use_llm and config is not None and config.llm_provider in ("openai", "gemini"):
        if config.llm_api_key:
            try:
                if config.llm_provider == "openai":
                    refined = _refine_subject_with_openai(subject, asset_type, config)
                else:
                    refined = _refine_subject_with_gemini(subject, asset_type, config)
                if refined:
                    subject = refined
                    used_llm = True
            except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
                warnings.warn(
                    f"LLM prompt optimizer failed ({exc}); falling back to deterministic prompt.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    final_prompt = build_deterministic_prompt(
        subject=subject,
        asset_type=asset_type,
        size=size,
        frames=frames,
        cols=resolved_cols if asset_type == "anim" else None,
        style=style,
        has_ref=has_ref,
        bg_color=bg_color,
        action_notes=action_notes,
        facing=norm_facing,
    )
    return PromptResult(
        subject=subject,
        final_prompt=final_prompt,
        style=style,
        asset_type=asset_type,
        used_llm=used_llm,
        cols=resolved_cols,
        rows=rows,
        facing=norm_facing,
    )

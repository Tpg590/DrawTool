"""CLI entry point for DrawTool."""

from __future__ import annotations

import argparse
import io
import re
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

from PIL import Image

from src.aseprite_runner import AsepriteError, run_aseprite_build
from src.config import AppConfig, load_config
from src.image_gen import (
    ImageGenError,
    ImageGeneratorClient,
    choose_generation_size,
    create_provider,
)
from src.pixel_processor import parse_palette, parse_size, process_asset
from src.prompt_optimizer import DEFAULT_STYLE, VALID_ASSET_TYPES, compute_grid, optimize_prompt


def slugify_name(text: str, max_len: int = 40) -> str:
    """Create a safe ASCII filename stem from an arbitrary prompt string."""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.lower()).strip("_")
    if not slug:
        return "asset"
    return slug[:max_len].strip("_") or "asset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drawtool",
        description="Generate 2D 8-bit pixel art game assets (.aseprite + .png) from text prompts.",
    )
    parser.add_argument("--prompt", required=True, help="Asset description (e.g. 'bình máu đỏ').")
    parser.add_argument(
        "--type",
        dest="asset_type",
        choices=VALID_ASSET_TYPES,
        default="item",
        help="Asset mode: item | tile | anim (default: item).",
    )
    parser.add_argument(
        "--size",
        default="32x32",
        help="Target pixel size per sprite/frame WxH (default: 32x32).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=4,
        help="Number of animation frames for --type anim (default: 4).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of spritesheet columns for --type anim (default: equals --frames).",
    )
    parser.add_argument(
        "--style",
        default=None,
        help=f"Style preset name (default: {DEFAULT_STYLE}).",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Optional reference image path to preserve character/palette consistency.",
    )
    parser.add_argument(
        "--palette",
        default=None,
        help="Optional .gpl file, hex file, or comma-separated hex palette (e.g. '#000000,#ffffff').",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=None,
        help="Max quantized colors (default: 16 for 8bit-topdown when --palette is omitted; 0 disables).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for .aseprite and .png (default: ./output).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Animation frame rate for Aseprite timeline (default: 8).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional output base filename (without extension).",
    )
    parser.add_argument(
        "--raw-image",
        default=None,
        help="Use an existing local raw image instead of calling the image generation API.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override raw image cache directory (default: .cache).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass raw image cache and force a fresh API request.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM prompt refinement and build prompt deterministically (0 LLM tokens).",
    )
    parser.add_argument(
        "--skip-aseprite",
        action="store_true",
        help="Save processed PNG without invoking Aseprite CLI.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to JSON configuration file.",
    )
    return parser


def resolve_max_colors(
    style: str,
    has_palette: bool,
    cli_max_colors: int | None,
) -> int | None:
    """Resolve effective max-colors setting respecting --palette and 8bit-topdown default."""
    if has_palette:
        return None
    if cli_max_colors is not None:
        return None if cli_max_colors <= 0 else cli_max_colors
    if style == DEFAULT_STYLE:
        return 16
    return None


def run_pipeline(args: argparse.Namespace, config: AppConfig | None = None) -> int:
    """Execute prompt optimization, image acquisition, pixel processing, and Aseprite export."""
    cfg = config if config is not None else load_config(Path(args.config) if args.config else None)
    style = args.style or cfg.default_style or DEFAULT_STYLE
    target_size = parse_size(args.size)

    if args.asset_type == "anim":
        cols, rows = compute_grid(args.frames, args.cols)
    else:
        cols, rows = 1, 1

    ref_path: Path | None = None
    if args.ref:
        ref_path = Path(args.ref)
        if not ref_path.is_file():
            print(f"[error] Reference image not found: {ref_path}", file=sys.stderr)
            return 2

    palette_colors = parse_palette(args.palette) if args.palette else None
    effective_max_colors = resolve_max_colors(
        style=style,
        has_palette=palette_colors is not None,
        cli_max_colors=args.max_colors,
    )

    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir
    cache_dir = Path(args.cache_dir) if args.cache_dir else cfg.cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem = args.name.strip() if args.name else f"{args.asset_type}_{slugify_name(args.prompt)}"

    # 1. Prompt Optimization (minimal/zero LLM tokens)
    prompt_res = optimize_prompt(
        user_prompt=args.prompt,
        asset_type=args.asset_type,
        size=target_size,
        frames=args.frames,
        cols=cols if args.asset_type == "anim" else None,
        style=style,
        has_ref=ref_path is not None,
        config=cfg,
        use_llm=not args.no_llm,
    )
    print(f"[1/4] Prompt ({'LLM+rules' if prompt_res.used_llm else 'deterministic'}): {prompt_res.final_prompt}")

    # 2. Image Generation or Local Raw Input
    if args.raw_image:
        raw_path = Path(args.raw_image)
        if not raw_path.is_file():
            print(f"[error] --raw-image file not found: {raw_path}", file=sys.stderr)
            return 2
        raw_bytes = raw_path.read_bytes()
        print(f"[2/4] Loaded local raw image: {raw_path}")
    else:
        try:
            provider = create_provider(cfg)
            client = ImageGeneratorClient(
                provider=provider,
                cache_dir=cache_dir,
                model_tag=cfg.image_model,
            )
            gen_size = choose_generation_size(
                target_size=target_size,
                asset_type=args.asset_type,
                cols=cols,
                rows=rows,
            )
            gen_res = client.generate_or_load(
                prompt=prompt_res.final_prompt,
                gen_size=gen_size,
                target_size=target_size,
                ref_path=ref_path,
                use_cache=not args.no_cache,
                extra_meta={"asset_type": args.asset_type, "style": style},
            )
            raw_bytes = gen_res.image_bytes
            source_label = "cache hit" if gen_res.from_cache else f"generated via {gen_res.provider}"
            print(f"[2/4] Raw image ({source_label}): {gen_res.cache_path}")
        except ImageGenError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2

    # 3. Deterministic Pixel & Spritesheet Processing
    with Image.open(io.BytesIO(raw_bytes)) as raw_img:
        processed_img, proc_meta = process_asset(
            img=raw_img,
            asset_type=args.asset_type,
            target_size=target_size,
            frames=args.frames,
            cols=cols if args.asset_type == "anim" else None,
            palette=palette_colors,
            max_colors=effective_max_colors,
        )

    staged_png = cache_dir / f"{stem}_processed.png"
    out_png = output_dir / f"{stem}.png"
    out_aseprite = output_dir / f"{stem}.aseprite"

    if args.skip_aseprite:
        processed_img.save(out_png, format="PNG")
        print(
            f"[3/4] Processed PNG saved ({proc_meta['sheet_size'][0]}x{proc_meta['sheet_size'][1]}): {out_png}"
        )
        print("[4/4] Skipped Aseprite packaging (--skip-aseprite).")
        return 0

    processed_img.save(staged_png, format="PNG")
    print(
        f"[3/4] Processed pixel asset ({proc_meta['sheet_size'][0]}x{proc_meta['sheet_size'][1]}): {staged_png}"
    )

    # 4. Aseprite CLI + Lua Script Packaging
    try:
        ase_res = run_aseprite_build(
            input_png=staged_png,
            out_aseprite=out_aseprite,
            out_png=out_png,
            asset_type=args.asset_type,
            frame_size=target_size,
            frames=args.frames if args.asset_type == "anim" else 1,
            cols=cols,
            fps=args.fps,
            configured_aseprite_path=cfg.aseprite_path or None,
            work_dir=cache_dir,
        )
    except AsepriteError as exc:
        # Preserve processed PNG in output_dir even if Aseprite binary isn't configured yet
        processed_img.save(out_png, format="PNG")
        print(f"[warn] Saved processed PNG to {out_png}, but Aseprite step failed: {exc}", file=sys.stderr)
        return 3

    print(f"[4/4] Exported Aseprite: {ase_res.aseprite_path}")
    print(f"[done] Exported PNG:      {ase_res.png_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_pipeline(args)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

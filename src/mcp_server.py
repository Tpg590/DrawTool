"""Zero-dependency MCP (Model Context Protocol) stdio server for DrawTool + Aseprite."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from PIL import Image

from src.aseprite_runner import (
    AsepriteError,
    AsepriteNotFoundError,
    find_aseprite,
    generate_lua_script,
    run_aseprite_build,
)
from src.cli import resolve_max_colors, slugify_name
from src.config import AppConfig, load_config
from src.image_gen import (
    ImageGenError,
    ImageGeneratorClient,
    choose_generation_size,
    create_provider,
)
from src.pixel_processor import parse_palette, parse_size, process_asset
from src.prompt_optimizer import DEFAULT_STYLE, VALID_ASSET_TYPES, compute_grid, optimize_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26")
SERVER_INFO = {"name": "drawtool-aseprite", "version": "0.1.0"}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "build_pixel_prompt",
        "description": (
            "Build an optimized 2D 8-bit top-down pixel art prompt for item, tile, or anim spritesheet "
            "generation, returning the final prompt and recommended generation dimensions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Asset description (e.g. 'red health potion')."},
                "type": {
                    "type": "string",
                    "enum": list(VALID_ASSET_TYPES),
                    "default": "item",
                    "description": "Asset mode: item | tile | anim.",
                },
                "size": {"type": "string", "default": "32x32", "description": "Target sprite/frame WxH."},
                "frames": {"type": "integer", "minimum": 1, "default": 4},
                "cols": {"type": "integer", "minimum": 1, "description": "Grid columns for anim (defaults to frames)."},
                "style": {"type": "string", "default": DEFAULT_STYLE},
                "has_ref": {"type": "boolean", "default": False},
                "use_llm": {"type": "boolean", "default": False},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "process_image_to_aseprite",
        "description": (
            "Process a raw generated image (from Gemini Web, Antigravity image-generator, or local file) "
            "deterministically: remove background, NEAREST-neighbor resize, slice/align anim spritesheets, "
            "blend tile seams, quantize palette, and package into .aseprite (Base + Detail & Shading layers) + .png."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "raw_image_path": {"type": "string", "description": "Path to raw input image."},
                "name": {"type": "string", "description": "Output base filename (without extension)."},
                "type": {"type": "string", "enum": list(VALID_ASSET_TYPES), "default": "item"},
                "size": {"type": "string", "default": "32x32"},
                "frames": {"type": "integer", "minimum": 1, "default": 4},
                "cols": {"type": "integer", "minimum": 1},
                "fps": {"type": "number", "exclusiveMinimum": 0, "default": 8.0},
                "style": {"type": "string", "default": DEFAULT_STYLE},
                "palette": {"type": "string", "description": "Optional .gpl file or comma-separated hex list."},
                "max_colors": {"type": "integer", "minimum": 0},
                "output_dir": {"type": "string", "description": "Output directory (default: ./output)."},
                "skip_aseprite": {"type": "boolean", "default": False},
            },
            "required": ["raw_image_path"],
        },
    },
    {
        "name": "generate_game_asset",
        "description": (
            "End-to-end pipeline: optimize prompt, generate raw image via configured IMAGE_PROVIDER "
            "(gemini, openai, bridge, sdwebui), process pixels, and export .aseprite + .png."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "name": {"type": "string"},
                "type": {"type": "string", "enum": list(VALID_ASSET_TYPES), "default": "item"},
                "size": {"type": "string", "default": "32x32"},
                "frames": {"type": "integer", "minimum": 1, "default": 4},
                "cols": {"type": "integer", "minimum": 1},
                "fps": {"type": "number", "exclusiveMinimum": 0, "default": 8.0},
                "style": {"type": "string", "default": DEFAULT_STYLE},
                "ref_path": {"type": "string"},
                "palette": {"type": "string"},
                "max_colors": {"type": "integer", "minimum": 0},
                "output_dir": {"type": "string"},
                "use_cache": {"type": "boolean", "default": True},
                "use_llm": {"type": "boolean", "default": True},
                "skip_aseprite": {"type": "boolean", "default": False},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "check_aseprite_status",
        "description": (
            "Check Aseprite CLI availability, active DrawTool configuration, and whether provider keys "
            "are configured (returns boolean flags only, never secret values)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _load_repo_config() -> AppConfig:
    env_file = REPO_ROOT / ".env"
    json_file = REPO_ROOT / "config.json"
    return load_config(
        config_path=json_file if json_file.is_file() else None,
        env_path=env_file if env_file.is_file() else None,
    )


def _resolve_path(raw_path: str | None, fallback: Path) -> Path:
    if raw_path and raw_path.strip():
        candidate = Path(raw_path.strip()).expanduser()
        return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
    return fallback if fallback.is_absolute() else (REPO_ROOT / fallback)


def _package_processed_image(
    processed_img: Image.Image,
    stem: str,
    asset_type: str,
    target_size: tuple[int, int],
    frames: int,
    cols: int,
    fps: float,
    output_dir: Path,
    cache_dir: Path,
    cfg: AppConfig,
    skip_aseprite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    staged_png = cache_dir / f"{stem}_processed.png"
    out_png = output_dir / f"{stem}.png"
    out_aseprite = output_dir / f"{stem}.aseprite"

    if skip_aseprite:
        processed_img.save(out_png, format="PNG")
        return {
            "status": "png_only",
            "png_path": str(out_png),
            "aseprite_path": None,
        }

    processed_img.save(staged_png, format="PNG")
    try:
        ase_res = run_aseprite_build(
            input_png=staged_png,
            out_aseprite=out_aseprite,
            out_png=out_png,
            asset_type=asset_type,
            frame_size=target_size,
            frames=frames if asset_type == "anim" else 1,
            cols=cols,
            fps=fps,
            configured_aseprite_path=cfg.aseprite_path or None,
            work_dir=cache_dir,
        )
        return {
            "status": "ok",
            "aseprite_path": str(ase_res.aseprite_path),
            "png_path": str(ase_res.png_path),
            "lua_script_path": str(ase_res.lua_script_path),
            "aseprite_executable": str(ase_res.executable),
        }
    except AsepriteError as exc:
        processed_img.save(out_png, format="PNG")
        lua_path = cache_dir / f"{stem}_build.lua"
        lua_path.write_text(
            generate_lua_script(
                input_png=staged_png,
                out_aseprite=out_aseprite,
                out_png=out_png,
                asset_type=asset_type,
                frame_size=target_size,
                frames=frames if asset_type == "anim" else 1,
                cols=cols,
                fps=fps,
            ),
            encoding="utf-8",
        )
        return {
            "status": "partial_png_only",
            "png_path": str(out_png),
            "aseprite_path": None,
            "lua_script_path": str(lua_path),
            "aseprite_error": str(exc),
        }


def _tool_build_pixel_prompt(args: dict[str, Any], cfg: AppConfig) -> dict[str, Any]:
    prompt_text = str(args.get("prompt", "")).strip()
    if not prompt_text:
        raise ValueError("'prompt' is required.")
    asset_type = str(args.get("type", "item"))
    target_size = parse_size(str(args.get("size", "32x32")))
    frames = int(args.get("frames", 4))
    cols_arg = args.get("cols")
    cols = int(cols_arg) if cols_arg is not None else None
    style = str(args.get("style") or cfg.default_style or DEFAULT_STYLE)
    has_ref = bool(args.get("has_ref", False))
    use_llm = bool(args.get("use_llm", False))

    res = optimize_prompt(
        user_prompt=prompt_text,
        asset_type=asset_type,
        size=target_size,
        frames=frames,
        cols=cols,
        style=style,
        has_ref=has_ref,
        config=cfg,
        use_llm=use_llm,
    )
    gen_size = choose_generation_size(
        target_size=target_size,
        asset_type=asset_type,
        cols=res.cols,
        rows=res.rows,
    )
    return {
        "final_prompt": res.final_prompt,
        "subject": res.subject,
        "asset_type": res.asset_type,
        "style": res.style,
        "target_size": f"{target_size[0]}x{target_size[1]}",
        "gen_size": f"{gen_size[0]}x{gen_size[1]}",
        "cols": res.cols,
        "rows": res.rows,
        "used_llm": res.used_llm,
    }


def _tool_process_image_to_aseprite(args: dict[str, Any], cfg: AppConfig) -> dict[str, Any]:
    raw_arg = str(args.get("raw_image_path", "")).strip()
    if not raw_arg:
        raise ValueError("'raw_image_path' is required.")
    raw_path = _resolve_path(raw_arg, REPO_ROOT)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw image not found: {raw_path}")

    asset_type = str(args.get("type", "item"))
    target_size = parse_size(str(args.get("size", "32x32")))
    frames = int(args.get("frames", 4))
    cols_arg = args.get("cols")
    cols, _ = compute_grid(frames, int(cols_arg) if cols_arg is not None else None) if asset_type == "anim" else (1, 1)
    fps = float(args.get("fps", 8.0))
    style = str(args.get("style") or cfg.default_style or DEFAULT_STYLE)

    palette_arg = args.get("palette")
    palette_colors = parse_palette(str(palette_arg)) if palette_arg else None
    max_colors_arg = args.get("max_colors")
    effective_max_colors = resolve_max_colors(
        style=style,
        has_palette=palette_colors is not None,
        cli_max_colors=int(max_colors_arg) if max_colors_arg is not None else None,
    )

    name_arg = str(args.get("name", "")).strip()
    stem = name_arg or f"{asset_type}_{slugify_name(raw_path.stem)}"
    output_dir = _resolve_path(args.get("output_dir"), cfg.output_dir)
    cache_dir = _resolve_path(None, cfg.cache_dir)
    skip_aseprite = bool(args.get("skip_aseprite", False))

    with Image.open(raw_path) as raw_img:
        processed_img, proc_meta = process_asset(
            img=raw_img,
            asset_type=asset_type,
            target_size=target_size,
            frames=frames,
            cols=cols if asset_type == "anim" else None,
            palette=palette_colors,
            max_colors=effective_max_colors,
        )

    pkg = _package_processed_image(
        processed_img=processed_img,
        stem=stem,
        asset_type=asset_type,
        target_size=target_size,
        frames=frames,
        cols=cols,
        fps=fps,
        output_dir=output_dir,
        cache_dir=cache_dir,
        cfg=cfg,
        skip_aseprite=skip_aseprite,
    )
    pkg.update(
        {
            "asset_type": asset_type,
            "target_size": f"{target_size[0]}x{target_size[1]}",
            "sheet_size": f"{proc_meta['sheet_size'][0]}x{proc_meta['sheet_size'][1]}",
            "cols": proc_meta["cols"],
            "rows": proc_meta["rows"],
            "frames": proc_meta["frames"],
        }
    )
    return pkg


def _tool_generate_game_asset(args: dict[str, Any], cfg: AppConfig) -> dict[str, Any]:
    prompt_text = str(args.get("prompt", "")).strip()
    if not prompt_text:
        raise ValueError("'prompt' is required.")
    asset_type = str(args.get("type", "item"))
    target_size = parse_size(str(args.get("size", "32x32")))
    frames = int(args.get("frames", 4))
    cols_arg = args.get("cols")
    cols, rows = compute_grid(frames, int(cols_arg) if cols_arg is not None else None) if asset_type == "anim" else (1, 1)
    fps = float(args.get("fps", 8.0))
    style = str(args.get("style") or cfg.default_style or DEFAULT_STYLE)

    ref_arg = args.get("ref_path")
    ref_path: Path | None = None
    if ref_arg:
        ref_path = _resolve_path(str(ref_arg), REPO_ROOT)
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")

    palette_arg = args.get("palette")
    palette_colors = parse_palette(str(palette_arg)) if palette_arg else None
    max_colors_arg = args.get("max_colors")
    effective_max_colors = resolve_max_colors(
        style=style,
        has_palette=palette_colors is not None,
        cli_max_colors=int(max_colors_arg) if max_colors_arg is not None else None,
    )

    output_dir = _resolve_path(args.get("output_dir"), cfg.output_dir)
    cache_dir = _resolve_path(None, cfg.cache_dir)
    name_arg = str(args.get("name", "")).strip()
    stem = name_arg or f"{asset_type}_{slugify_name(prompt_text)}"

    prompt_res = optimize_prompt(
        user_prompt=prompt_text,
        asset_type=asset_type,
        size=target_size,
        frames=frames,
        cols=cols if asset_type == "anim" else None,
        style=style,
        has_ref=ref_path is not None,
        config=cfg,
        use_llm=bool(args.get("use_llm", True)),
    )

    provider = create_provider(cfg)
    client = ImageGeneratorClient(provider=provider, cache_dir=cache_dir, model_tag=cfg.image_model)
    gen_size = choose_generation_size(target_size=target_size, asset_type=asset_type, cols=cols, rows=rows)
    gen_res = client.generate_or_load(
        prompt=prompt_res.final_prompt,
        gen_size=gen_size,
        target_size=target_size,
        ref_path=ref_path,
        use_cache=bool(args.get("use_cache", True)),
        extra_meta={"asset_type": asset_type, "style": style},
    )

    with Image.open(io.BytesIO(gen_res.image_bytes)) as raw_img:
        processed_img, proc_meta = process_asset(
            img=raw_img,
            asset_type=asset_type,
            target_size=target_size,
            frames=frames,
            cols=cols if asset_type == "anim" else None,
            palette=palette_colors,
            max_colors=effective_max_colors,
        )

    pkg = _package_processed_image(
        processed_img=processed_img,
        stem=stem,
        asset_type=asset_type,
        target_size=target_size,
        frames=frames,
        cols=cols,
        fps=fps,
        output_dir=output_dir,
        cache_dir=cache_dir,
        cfg=cfg,
        skip_aseprite=bool(args.get("skip_aseprite", False)),
    )
    pkg.update(
        {
            "final_prompt": prompt_res.final_prompt,
            "raw_cache_path": str(gen_res.cache_path),
            "from_cache": gen_res.from_cache,
            "provider": gen_res.provider,
            "sheet_size": f"{proc_meta['sheet_size'][0]}x{proc_meta['sheet_size'][1]}",
        }
    )
    return pkg


def _tool_check_aseprite_status(_: dict[str, Any], cfg: AppConfig) -> dict[str, Any]:
    aseprite_found = False
    aseprite_exe: str | None = None
    aseprite_msg = ""
    try:
        aseprite_exe = str(find_aseprite(cfg.aseprite_path or None))
        aseprite_found = True
    except AsepriteNotFoundError as exc:
        aseprite_msg = str(exc)

    return {
        "aseprite_found": aseprite_found,
        "aseprite_executable": aseprite_exe,
        "aseprite_message": aseprite_msg or "ready",
        "image_provider": cfg.image_provider,
        "has_image_api_key": bool(cfg.image_api_key),
        "llm_provider": cfg.llm_provider,
        "has_llm_api_key": bool(cfg.llm_api_key),
        "default_style": cfg.default_style,
        "output_dir": str(_resolve_path(None, cfg.output_dir)),
        "cache_dir": str(_resolve_path(None, cfg.cache_dir)),
    }


def call_tool(
    name: str,
    arguments: dict[str, Any],
    config: AppConfig | None = None,
) -> tuple[dict[str, Any], bool]:
    """Dispatch an MCP tool call and return (result_dict, is_error)."""
    cfg = config if config is not None else _load_repo_config()
    handlers = {
        "build_pixel_prompt": _tool_build_pixel_prompt,
        "process_image_to_aseprite": _tool_process_image_to_aseprite,
        "generate_game_asset": _tool_generate_game_asset,
        "check_aseprite_status": _tool_check_aseprite_status,
    }
    handler = handlers.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}, True

    try:
        return handler(arguments or {}, cfg), False
    except (ValueError, FileNotFoundError, ImageGenError, AsepriteError, OSError) as exc:
        return {"error": str(exc)}, True


def handle_jsonrpc_request(
    req: dict[str, Any],
    config: AppConfig | None = None,
) -> dict[str, Any] | None:
    """Handle one JSON-RPC 2.0 request object; returns None for notifications."""
    if not isinstance(req, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}

    method = req.get("method")
    if "id" not in req:
        # JSON-RPC notification (e.g. notifications/initialized) must not receive a response
        return None

    req_id = req.get("id")
    params = req.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params"}}

    if method == "initialize":
        client_proto = str(params.get("protocolVersion", ""))
        proto = client_proto if client_proto in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": proto,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_DEFINITIONS}}

    if method == "tools/call":
        tool_name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Tool arguments must be an object"}}
        payload, is_error = call_tool(tool_name, arguments, config=config)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}],
                "isError": is_error,
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def serve_stdio(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Run the MCP server loop over newline-delimited JSON-RPC stdio."""
    in_stream = stdin if stdin is not None else sys.stdin
    out_stream = stdout if stdout is not None else sys.stdout

    for raw_line in in_stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {exc}"}}
            out_stream.write(json.dumps(err, ensure_ascii=False) + "\n")
            out_stream.flush()
            continue

        resp = handle_jsonrpc_request(req)
        if resp is not None:
            out_stream.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out_stream.flush()

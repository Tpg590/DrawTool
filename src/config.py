"""Configuration and .env loader for DrawTool."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _strip_inline_comment(value: str) -> str:
    """Strip unquoted inline # comments from a .env value."""
    in_single = False
    in_double = False
    for idx, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or value[idx - 1].isspace():
                return value[:idx].rstrip()
    return value.strip()


def load_dotenv(env_path: Path | None = None) -> dict[str, str]:
    """Load key=value pairs from a .env file without overriding existing os.environ keys."""
    target = env_path if env_path is not None else Path.cwd() / ".env"
    loaded: dict[str, str] = {}
    if not target.is_file():
        return loaded

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_val = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        val = _strip_inline_comment(raw_val.strip())
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        loaded[key] = val
        if key not in os.environ:
            os.environ[key] = val
    return loaded


@dataclass
class AppConfig:
    """Runtime configuration resolved from environment, .env, and optional JSON config."""

    image_provider: str = "bridge"
    image_api_url: str = ""
    image_api_key: str = ""
    image_model: str = ""
    llm_provider: str = "none"
    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    aseprite_path: str = ""
    default_style: str = "8bit-topdown"
    cache_dir: Path = Path(".cache")
    output_dir: Path = Path("output")
    request_timeout: int = 120
    sdwebui_lora: str = "pixel-art-xl"
    sdwebui_lora_weight: float = 0.9
    sdwebui_steps: int = 20
    sdwebui_cfg_scale: float = 7.0
    sdwebui_sampler: str = "DPM++ 2M"
    sdwebui_denoising: float = 0.65
    sdwebui_negative_prompt: str = (
        "3d render, realistic, photo, blurry, anti-aliasing, smooth gradient, "
        "watermark, text, signature, jpeg artifacts"
    )


def _pick(env_key: str, json_data: dict[str, Any], json_key: str, default: Any) -> Any:
    env_val = os.environ.get(env_key)
    if env_val is not None and env_val != "":
        return env_val
    if json_key in json_data and json_data[json_key] not in (None, ""):
        return json_data[json_key]
    return default


def load_config(
    config_path: Path | None = None,
    env_path: Path | None = None,
) -> AppConfig:
    """Load AppConfig from .env, optional JSON file, and environment variables."""
    load_dotenv(env_path)

    json_data: dict[str, Any] = {}
    candidate_json = config_path if config_path is not None else Path.cwd() / "config.json"
    if candidate_json.is_file():
        json_data = json.loads(candidate_json.read_text(encoding="utf-8"))
    elif config_path is not None:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    timeout_raw = _pick("REQUEST_TIMEOUT", json_data, "request_timeout", 120)
    try:
        request_timeout = max(5, int(timeout_raw))
    except (TypeError, ValueError):
        request_timeout = 120

    image_provider = str(_pick("IMAGE_PROVIDER", json_data, "image_provider", "bridge")).lower()
    image_api_key = str(_pick("IMAGE_API_KEY", json_data, "image_api_key", ""))
    if image_provider == "gemini" and not image_api_key:
        image_api_key = str(_pick("GEMINI_API_KEY", json_data, "gemini_api_key", ""))

    llm_provider = str(_pick("LLM_PROVIDER", json_data, "llm_provider", "none")).lower()
    llm_api_key = str(_pick("LLM_API_KEY", json_data, "llm_api_key", ""))
    if llm_provider == "gemini" and not llm_api_key:
        llm_api_key = str(_pick("GEMINI_API_KEY", json_data, "gemini_api_key", ""))

    sdwebui_lora = str(_pick("SDWEBUI_LORA", json_data, "sdwebui_lora", "pixel-art-xl"))
    if os.environ.get("SDWEBUI_LORA") == "":
        sdwebui_lora = ""

    return AppConfig(
        image_provider=image_provider,
        image_api_url=str(_pick("IMAGE_API_URL", json_data, "image_api_url", "")),
        image_api_key=image_api_key,
        image_model=str(_pick("IMAGE_MODEL", json_data, "image_model", "")),
        llm_provider=llm_provider,
        llm_api_url=str(_pick("LLM_API_URL", json_data, "llm_api_url", "")),
        llm_api_key=llm_api_key,
        llm_model=str(_pick("LLM_MODEL", json_data, "llm_model", "")),
        aseprite_path=str(_pick("ASEPRITE_PATH", json_data, "aseprite_path", "")),
        default_style=str(_pick("DEFAULT_STYLE", json_data, "default_style", "8bit-topdown")),
        cache_dir=Path(str(_pick("CACHE_DIR", json_data, "cache_dir", ".cache"))),
        output_dir=Path(str(_pick("OUTPUT_DIR", json_data, "output_dir", "output"))),
        request_timeout=request_timeout,
        sdwebui_lora=sdwebui_lora,
        sdwebui_lora_weight=float(_pick("SDWEBUI_LORA_WEIGHT", json_data, "sdwebui_lora_weight", 0.9)),
        sdwebui_steps=int(_pick("SDWEBUI_STEPS", json_data, "sdwebui_steps", 20)),
        sdwebui_cfg_scale=float(_pick("SDWEBUI_CFG_SCALE", json_data, "sdwebui_cfg_scale", 7.0)),
        sdwebui_sampler=str(_pick("SDWEBUI_SAMPLER", json_data, "sdwebui_sampler", "DPM++ 2M")),
        sdwebui_denoising=float(_pick("SDWEBUI_DENOISING", json_data, "sdwebui_denoising", 0.65)),
        sdwebui_negative_prompt=str(
            _pick(
                "SDWEBUI_NEGATIVE_PROMPT",
                json_data,
                "sdwebui_negative_prompt",
                "3d render, realistic, photo, blurry, anti-aliasing, smooth gradient, "
                "watermark, text, signature, jpeg artifacts",
            )
        ),
    )

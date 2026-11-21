"""Image generation providers and SHA-256 disk caching client."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import AppConfig


class ImageGenError(RuntimeError):
    """Raised when image generation fails or configuration is incomplete."""


@dataclass(frozen=True)
class GeneratedImage:
    """Raw generated image bytes and cache metadata."""

    image_bytes: bytes
    cache_path: Path
    from_cache: bool
    provider: str
    prompt: str


def choose_generation_size(
    target_size: tuple[int, int],
    asset_type: str,
    cols: int = 1,
    rows: int = 1,
) -> tuple[int, int]:
    """Pick an API-friendly generation resolution matching the target or sheet aspect ratio."""
    w, h = target_size
    sheet_w = w * cols if asset_type == "anim" else w
    sheet_h = h * rows if asset_type == "anim" else h
    aspect = sheet_w / max(1, sheet_h)
    if aspect >= 1.35:
        return (1536, 1024)
    if aspect <= 0.74:
        return (1024, 1536)
    return (1024, 1024)


def _decode_base64_image(raw_value: str) -> bytes:
    payload = raw_value.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def _download_url(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _extract_image_bytes(data: dict[str, Any], timeout: int) -> bytes:
    for key in ("b64_json", "image_base64", "image"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return _decode_base64_image(val)

    data_list = data.get("data")
    if isinstance(data_list, list) and data_list:
        first = data_list[0]
        if isinstance(first, dict):
            if isinstance(first.get("b64_json"), str):
                return _decode_base64_image(first["b64_json"])
            if isinstance(first.get("url"), str):
                return _download_url(first["url"], timeout)

    images_list = data.get("images")
    if isinstance(images_list, list) and images_list:
        first = images_list[0]
        if isinstance(first, str):
            return _decode_base64_image(first)
        if isinstance(first, dict):
            if isinstance(first.get("b64_json"), str):
                return _decode_base64_image(first["b64_json"])
            if isinstance(first.get("url"), str):
                return _download_url(first["url"], timeout)

    predictions = data.get("predictions")
    if isinstance(predictions, list) and predictions:
        first = predictions[0]
        if isinstance(first, dict) and isinstance(first.get("bytesBase64Encoded"), str):
            return _decode_base64_image(first["bytesBase64Encoded"])

    raise ImageGenError(f"Unrecognized image API response structure: keys={list(data.keys())}")


def _http_post_raw(
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: int,
) -> tuple[bytes, str]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            return resp.read(), content_type
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ImageGenError(f"Image API HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ImageGenError(f"Failed to reach image API {url}: {exc.reason}") from exc


def _build_multipart_form(
    fields: list[tuple[str, str]],
    files: list[tuple[str, str, str, bytes]],
) -> tuple[bytes, str]:
    boundary = f"----DrawToolBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.append(f"--{boundary}\r\n".encode("ascii"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(value.encode("utf-8"))
        chunks.append(b"\r\n")
    for name, filename, content_type, data in files:
        chunks.append(f"--{boundary}\r\n".encode("ascii"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("ascii"))
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class ImageProvider(ABC):
    """Abstract base class for image generation API providers."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        gen_size: tuple[int, int],
        target_size: tuple[int, int],
        ref_path: Path | None = None,
    ) -> bytes:
        """Generate an image and return raw PNG/image bytes."""


class BridgeImageProvider(ImageProvider):
    """Generic HTTP JSON bridge provider (compatible with local gpt-image-bridge and custom proxies)."""

    name = "bridge"

    def __init__(self, api_url: str, api_key: str = "", model: str = "", timeout: int = 120) -> None:
        if not api_url:
            raise ImageGenError(
                "IMAGE_API_URL is required for provider 'bridge'. "
                "Set IMAGE_API_URL in .env (for example: http://127.0.0.1:8080/v1/images/generations)."
            )
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        gen_size: tuple[int, int],
        target_size: tuple[int, int],
        ref_path: Path | None = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "size": f"{gen_size[0]}x{gen_size[1]}",
            "target_size": f"{target_size[0]}x{target_size[1]}",
            "n": 1,
            "response_format": "b64_json",
        }
        if self.model:
            payload["model"] = self.model
        if ref_path is not None:
            raw_ref = ref_path.read_bytes()
            mime = mimetypes.guess_type(str(ref_path))[0] or "image/png"
            b64_ref = base64.b64encode(raw_ref).decode("ascii")
            payload["ref_image_base64"] = b64_ref
            payload["reference_image"] = f"data:{mime};base64,{b64_ref}"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        raw_resp, content_type = _http_post_raw(
            self.api_url,
            json.dumps(payload).encode("utf-8"),
            headers,
            self.timeout,
        )
        if content_type.lower().startswith("image/"):
            return raw_resp
        return _extract_image_bytes(json.loads(raw_resp.decode("utf-8")), self.timeout)


class OpenAIImageProvider(ImageProvider):
    """OpenAI Images API provider (generations and edits when --ref is provided)."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        api_url: str = "",
        model: str = "",
        timeout: int = 120,
    ) -> None:
        if not api_key:
            raise ImageGenError(
                "IMAGE_API_KEY is required for provider 'openai'. Set IMAGE_API_KEY in .env."
            )
        self.api_key = api_key
        self.api_url = api_url or "https://api.openai.com/v1/images/generations"
        self.model = model or "gpt-image-1"
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        gen_size: tuple[int, int],
        target_size: tuple[int, int],
        ref_path: Path | None = None,
    ) -> bytes:
        size_str = f"{gen_size[0]}x{gen_size[1]}"
        if ref_path is not None:
            edits_url = (
                self.api_url.replace("/images/generations", "/images/edits")
                if "/images/generations" in self.api_url
                else self.api_url
            )
            mime = mimetypes.guess_type(str(ref_path))[0] or "image/png"
            body, content_type = _build_multipart_form(
                fields=[
                    ("model", self.model),
                    ("prompt", prompt),
                    ("size", size_str),
                ],
                files=[("image", ref_path.name, mime, ref_path.read_bytes())],
            )
            raw_resp, resp_ct = _http_post_raw(
                edits_url,
                body,
                {
                    "Content-Type": content_type,
                    "Authorization": f"Bearer {self.api_key}",
                },
                self.timeout,
            )
        else:
            payload: dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "size": size_str,
                "n": 1,
            }
            if self.model.startswith("dall-e"):
                payload["response_format"] = "b64_json"
            raw_resp, resp_ct = _http_post_raw(
                self.api_url,
                json.dumps(payload).encode("utf-8"),
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                self.timeout,
            )

        if resp_ct.lower().startswith("image/"):
            return raw_resp
        return _extract_image_bytes(json.loads(raw_resp.decode("utf-8")), self.timeout)


class GeminiImageProvider(ImageProvider):
    """Google Gemini / Imagen direct REST provider (no localhost bridge required)."""

    name = "gemini"
    DEFAULT_MODEL = "gemini-2.5-flash-image"

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "",
        model: str = "",
        timeout: int = 120,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("IMAGE_API_KEY", "")
        ).strip()
        if not resolved_key:
            raise ImageGenError(
                "GEMINI_API_KEY or IMAGE_API_KEY is required for provider 'gemini'. Set it in .env."
            )
        self.api_key = resolved_key

        clean_model = (model or "").strip()
        if not clean_model or clean_model.lower().startswith(("gpt-", "dall-e")):
            clean_model = self.DEFAULT_MODEL
        self.model = clean_model
        self.api_url = (api_url or "").strip()
        self.timeout = timeout

    def _aspect_ratio(self, gen_size: tuple[int, int]) -> str:
        w, h = gen_size
        if w > h:
            return "3:2"
        if h > w:
            return "2:3"
        return "1:1"

    def _resolve_endpoint(self, action: str) -> str:
        url = self.api_url
        if url:
            lower = url.lower()
            # Ignore leftover bridge/OpenAI URLs when IMAGE_PROVIDER=gemini
            if "localhost" in lower or "127.0.0.1" in lower or "/images/generations" in lower:
                url = ""
        if url:
            return url
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:{action}"

    def generate(
        self,
        prompt: str,
        gen_size: tuple[int, int],
        target_size: tuple[int, int],
        ref_path: Path | None = None,
    ) -> bytes:
        is_imagen = "imagen" in self.model.lower()
        if is_imagen and ref_path is not None:
            raise ImageGenError(
                "Imagen ':predict' models do not support --ref directly. "
                "Use a Gemini multimodal image model (e.g. IMAGE_MODEL=gemini-2.5-flash-image) "
                "or IMAGE_PROVIDER=bridge."
            )

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        if is_imagen:
            url = self._resolve_endpoint("predict")
            payload: dict[str, Any] = {
                "instances": [{"prompt": prompt}],
                "parameters": {
                    "sampleCount": 1,
                    "aspectRatio": self._aspect_ratio(gen_size),
                },
            }
            raw_resp, _ = _http_post_raw(
                url,
                json.dumps(payload).encode("utf-8"),
                headers,
                self.timeout,
            )
            return _extract_image_bytes(json.loads(raw_resp.decode("utf-8")), self.timeout)

        url = self._resolve_endpoint("generateContent")
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if ref_path is not None:
            mime = mimetypes.guess_type(str(ref_path))[0] or "image/png"
            b64_ref = base64.b64encode(ref_path.read_bytes()).decode("ascii")
            parts.append({"inlineData": {"mimeType": mime, "data": b64_ref}})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        raw_resp, _ = _http_post_raw(
            url,
            json.dumps(payload).encode("utf-8"),
            headers,
            self.timeout,
        )
        data = json.loads(raw_resp.decode("utf-8"))
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline, dict) and isinstance(inline.get("data"), str):
                    return _decode_base64_image(inline["data"])
        raise ImageGenError("Gemini response did not contain an inline image part.")


class SDWebUIImageProvider(ImageProvider):
    """Local Stable Diffusion WebUI Forge / Automatic1111 REST provider."""

    name = "sdwebui"
    DEFAULT_BASE_URL = "http://127.0.0.1:7860"
    DEFAULT_NEGATIVE_PROMPT = (
        "3d render, realistic, photo, blurry, anti-aliasing, smooth gradient, "
        "watermark, text, signature, jpeg artifacts"
    )

    def __init__(
        self,
        api_url: str = "",
        model: str = "",
        lora: str = "pixel-art-xl",
        lora_weight: float = 0.9,
        steps: int = 20,
        cfg_scale: float = 7.0,
        sampler: str = "DPM++ 2M",
        denoising: float = 0.65,
        negative_prompt: str = "",
        timeout: int = 120,
    ) -> None:
        url = (api_url or "").strip()
        if url and "/v1/images/generations" in url.lower():
            url = ""
        if not url:
            url = self.DEFAULT_BASE_URL
        for suffix in ("/sdapi/v1/txt2img", "/sdapi/v1/img2img"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
        self.base_url = url.rstrip("/")

        clean_model = (model or "").strip()
        if clean_model.lower().startswith(("gpt-", "dall-e", "gemini", "imagen")):
            clean_model = ""
        self.model = clean_model

        clean_lora = (lora or "").strip()
        self.lora = "" if clean_lora.lower() in ("none", "off", "false") else clean_lora
        self.lora_weight = float(lora_weight)
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.sampler = sampler or "DPM++ 2M"
        self.denoising = float(denoising)
        self.negative_prompt = negative_prompt or self.DEFAULT_NEGATIVE_PROMPT
        self.timeout = timeout

    def _sdxl_dimensions(self, gen_size: tuple[int, int]) -> tuple[int, int]:
        w, h = gen_size
        if w > h:
            return (1152, 896)
        if h > w:
            return (896, 1152)
        return (1024, 1024)

    def _build_prompt(self, prompt: str) -> str:
        if self.lora and "<lora:" not in prompt:
            return f"{prompt}, <lora:{self.lora}:{self.lora_weight:g}>"
        return prompt

    def generate(
        self,
        prompt: str,
        gen_size: tuple[int, int],
        target_size: tuple[int, int],
        ref_path: Path | None = None,
    ) -> bytes:
        width, height = self._sdxl_dimensions(gen_size)
        payload: dict[str, Any] = {
            "prompt": self._build_prompt(prompt),
            "negative_prompt": self.negative_prompt,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "sampler_name": self.sampler,
            "width": width,
            "height": height,
            "batch_size": 1,
            "n_iter": 1,
        }
        if self.model:
            payload["override_settings"] = {"sd_model_checkpoint": self.model}

        if ref_path is not None:
            endpoint = f"{self.base_url}/sdapi/v1/img2img"
            payload["init_images"] = [base64.b64encode(ref_path.read_bytes()).decode("ascii")]
            payload["denoising_strength"] = self.denoising
        else:
            endpoint = f"{self.base_url}/sdapi/v1/txt2img"

        try:
            raw_resp, resp_ct = _http_post_raw(
                endpoint,
                json.dumps(payload).encode("utf-8"),
                {"Content-Type": "application/json"},
                self.timeout,
            )
        except ImageGenError as exc:
            if "Failed to reach image API" in str(exc):
                raise ImageGenError(
                    f"{exc}. Make sure SD WebUI Forge is running with '--api' at {self.base_url}."
                ) from exc
            raise

        if resp_ct.lower().startswith("image/"):
            return raw_resp
        return _extract_image_bytes(json.loads(raw_resp.decode("utf-8")), self.timeout)


def create_provider(config: AppConfig) -> ImageProvider:
    """Instantiate the configured image generation provider."""
    provider = config.image_provider.strip().lower()
    if provider == "bridge":
        return BridgeImageProvider(
            api_url=config.image_api_url,
            api_key=config.image_api_key,
            model=config.image_model,
            timeout=config.request_timeout,
        )
    if provider == "openai":
        return OpenAIImageProvider(
            api_key=config.image_api_key,
            api_url=config.image_api_url,
            model=config.image_model,
            timeout=config.request_timeout,
        )
    if provider == "gemini":
        return GeminiImageProvider(
            api_key=config.image_api_key or os.environ.get("GEMINI_API_KEY", ""),
            api_url=config.image_api_url,
            model=config.image_model,
            timeout=config.request_timeout,
        )
    if provider in ("sdwebui", "forge", "a1111"):
        return SDWebUIImageProvider(
            api_url=config.image_api_url,
            model=config.image_model,
            lora=config.sdwebui_lora,
            lora_weight=config.sdwebui_lora_weight,
            steps=config.sdwebui_steps,
            cfg_scale=config.sdwebui_cfg_scale,
            sampler=config.sdwebui_sampler,
            denoising=config.sdwebui_denoising,
            negative_prompt=config.sdwebui_negative_prompt,
            timeout=config.request_timeout,
        )
    raise ImageGenError(
        f"Unsupported IMAGE_PROVIDER '{config.image_provider}'. "
        "Expected 'bridge', 'openai', 'gemini', or 'sdwebui'."
    )


class ImageGeneratorClient:
    """High-level image client with SHA-256 disk caching for raw outputs."""

    def __init__(self, provider: ImageProvider, cache_dir: Path, model_tag: str = "") -> None:
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.model_tag = model_tag

    def _build_cache_key(
        self,
        prompt: str,
        gen_size: tuple[int, int],
        ref_path: Path | None,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.provider.name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(self.model_tag.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(f"{gen_size[0]}x{gen_size[1]}".encode("ascii"))
        hasher.update(b"\x00")
        hasher.update(prompt.encode("utf-8"))
        if ref_path is not None:
            hasher.update(b"\x00")
            hasher.update(hashlib.sha256(ref_path.read_bytes()).digest())
        return hasher.hexdigest()[:24]

    def generate_or_load(
        self,
        prompt: str,
        gen_size: tuple[int, int],
        target_size: tuple[int, int],
        ref_path: Path | None = None,
        use_cache: bool = True,
        extra_meta: dict[str, Any] | None = None,
    ) -> GeneratedImage:
        """Return raw image bytes from cache or call provider and persist to cache_dir."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._build_cache_key(prompt, gen_size, ref_path)
        cache_png = self.cache_dir / f"{cache_key}.png"
        cache_meta = self.cache_dir / f"{cache_key}.meta.json"

        if use_cache and cache_png.is_file():
            return GeneratedImage(
                image_bytes=cache_png.read_bytes(),
                cache_path=cache_png,
                from_cache=True,
                provider=self.provider.name,
                prompt=prompt,
            )

        raw_bytes = self.provider.generate(
            prompt=prompt,
            gen_size=gen_size,
            target_size=target_size,
            ref_path=ref_path,
        )
        # Normalize raw image into lossless PNG for deterministic reprocessing
        with Image.open(io.BytesIO(raw_bytes)) as img:
            buf = io.BytesIO()
            img.convert("RGBA").save(buf, format="PNG")
            png_bytes = buf.getvalue()

        cache_png.write_bytes(png_bytes)
        meta: dict[str, Any] = {
            "provider": self.provider.name,
            "model": self.model_tag,
            "gen_size": f"{gen_size[0]}x{gen_size[1]}",
            "target_size": f"{target_size[0]}x{target_size[1]}",
            "prompt": prompt,
            "ref_path": str(ref_path) if ref_path else None,
        }
        if extra_meta:
            meta.update(extra_meta)
        cache_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        return GeneratedImage(
            image_bytes=png_bytes,
            cache_path=cache_png,
            from_cache=False,
            provider=self.provider.name,
            prompt=prompt,
        )

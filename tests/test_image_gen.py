"""Tests for direct Gemini image generation provider and key/endpoint resolution."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.config import load_config
from src.image_gen import GeminiImageProvider, ImageGenError, SDWebUIImageProvider, create_provider


def _tiny_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeHeaders(dict):
    def get(self, key: str, default: str = "") -> str:
        return super().get(key, default)


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")
        self.headers = _FakeHeaders({"Content-Type": "application/json"})

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class GeminiImageProviderTests(unittest.TestCase):
    def test_config_falls_back_to_gemini_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "IMAGE_PROVIDER=gemini\nGEMINI_API_KEY=gem-secret-key\n"
                "IMAGE_API_URL=http://127.0.0.1:8080/v1/images/generations\n"
                "IMAGE_MODEL=gpt-image-1\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                cfg = load_config(env_path=env_file)
                self.assertEqual(cfg.image_provider, "gemini")
                self.assertEqual(cfg.image_api_key, "gem-secret-key")
                provider = create_provider(cfg)
                self.assertIsInstance(provider, GeminiImageProvider)
                assert isinstance(provider, GeminiImageProvider)
                self.assertEqual(provider.api_key, "gem-secret-key")
                self.assertEqual(provider.model, "gemini-2.5-flash-image")

    def test_gemini_generate_content_ignores_localhost_url_and_uses_header_auth(self) -> None:
        png_bytes = _tiny_png_bytes()
        b64_png = base64.b64encode(png_bytes).decode("ascii")
        fake_resp = _FakeHTTPResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"inlineData": {"mimeType": "image/png", "data": b64_png}}
                            ]
                        }
                    }
                ]
            }
        )

        provider = GeminiImageProvider(
            api_key="gem-key-123",
            api_url="http://127.0.0.1:8080/v1/images/generations",
            model="gpt-image-1",
        )

        with patch("urllib.request.urlopen", return_value=fake_resp) as mocked:
            out = provider.generate("red potion", (1024, 1024), (32, 32))

        self.assertEqual(out, png_bytes)
        req = mocked.call_args[0][0]
        self.assertEqual(
            req.full_url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
        )
        self.assertEqual(req.get_header("X-goog-api-key"), "gem-key-123")
        sent = json.loads(req.data.decode("utf-8"))
        self.assertEqual(sent["generationConfig"]["responseModalities"], ["TEXT", "IMAGE"])

    def test_gemini_generate_content_sends_reference_image(self) -> None:
        png_bytes = _tiny_png_bytes()
        b64_png = base64.b64encode(png_bytes).decode("ascii")
        fake_resp = _FakeHTTPResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"inline_data": {"mime_type": "image/png", "data": b64_png}}]}}
                ]
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ref = Path(tmpdir) / "ref.png"
            ref.write_bytes(png_bytes)
            provider = GeminiImageProvider(api_key="gem-key-ref")
            with patch("urllib.request.urlopen", return_value=fake_resp) as mocked:
                out = provider.generate("knight walk", (1024, 1024), (32, 32), ref_path=ref)

        self.assertEqual(out, png_bytes)
        sent = json.loads(mocked.call_args[0][0].data.decode("utf-8"))
        parts = sent["contents"][0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[1]["inlineData"]["mimeType"], "image/png")
        self.assertEqual(parts[1]["inlineData"]["data"], b64_png)

    def test_gemini_imagen_predict_endpoint_and_header(self) -> None:
        png_bytes = _tiny_png_bytes()
        b64_png = base64.b64encode(png_bytes).decode("ascii")
        fake_resp = _FakeHTTPResponse({"predictions": [{"bytesBase64Encoded": b64_png}]})

        provider = GeminiImageProvider(
            api_key="gem-imagen-key",
            model="imagen-3.0-generate-002",
        )
        with patch("urllib.request.urlopen", return_value=fake_resp) as mocked:
            out = provider.generate("grass tile", (1024, 1024), (16, 16))

        self.assertEqual(out, png_bytes)
        req = mocked.call_args[0][0]
        self.assertEqual(
            req.full_url,
            "https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict",
        )
        self.assertEqual(req.get_header("X-goog-api-key"), "gem-imagen-key")

    def test_gemini_missing_key_raises_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ImageGenError) as ctx:
                GeminiImageProvider(api_key="")
            self.assertIn("GEMINI_API_KEY", str(ctx.exception))


class SDWebUIImageProviderTests(unittest.TestCase):
    def test_sdwebui_txt2img_ignores_leftover_bridge_url_and_injects_lora(self) -> None:
        png_bytes = _tiny_png_bytes()
        b64_png = base64.b64encode(png_bytes).decode("ascii")
        fake_resp = _FakeHTTPResponse({"images": [b64_png]})

        provider = SDWebUIImageProvider(
            api_url="http://127.0.0.1:8080/v1/images/generations",
            model="gpt-image-1",
            lora="pixel-art-xl",
            lora_weight=0.9,
        )

        with patch("urllib.request.urlopen", return_value=fake_resp) as mocked:
            out = provider.generate("red potion", (1024, 1024), (32, 32))

        self.assertEqual(out, png_bytes)
        req = mocked.call_args[0][0]
        self.assertEqual(req.full_url, "http://127.0.0.1:7860/sdapi/v1/txt2img")
        sent = json.loads(req.data.decode("utf-8"))
        self.assertIn("<lora:pixel-art-xl:0.9>", sent["prompt"])
        self.assertEqual(sent["width"], 1024)
        self.assertEqual(sent["height"], 1024)
        self.assertNotIn("override_settings", sent)
        self.assertIn("3d render", sent["negative_prompt"])

    def test_sdwebui_img2img_with_reference_image(self) -> None:
        png_bytes = _tiny_png_bytes()
        b64_png = base64.b64encode(png_bytes).decode("ascii")
        fake_resp = _FakeHTTPResponse({"images": [b64_png]})

        with tempfile.TemporaryDirectory() as tmpdir:
            ref = Path(tmpdir) / "ref.png"
            ref.write_bytes(png_bytes)
            provider = SDWebUIImageProvider(
                api_url="http://127.0.0.1:7860",
                model="sd_xl_base_1.0.safetensors",
                denoising=0.6,
            )
            with patch("urllib.request.urlopen", return_value=fake_resp) as mocked:
                out = provider.generate("knight walk", (1536, 1024), (32, 32), ref_path=ref)

        self.assertEqual(out, png_bytes)
        req = mocked.call_args[0][0]
        self.assertEqual(req.full_url, "http://127.0.0.1:7860/sdapi/v1/img2img")
        sent = json.loads(req.data.decode("utf-8"))
        self.assertEqual(sent["init_images"], [b64_png])
        self.assertEqual(sent["denoising_strength"], 0.6)
        self.assertEqual(sent["width"], 1152)
        self.assertEqual(sent["height"], 896)
        self.assertEqual(
            sent["override_settings"]["sd_model_checkpoint"],
            "sd_xl_base_1.0.safetensors",
        )

    def test_create_provider_supports_sdwebui_and_forge_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("IMAGE_PROVIDER=forge\nSDWEBUI_LORA=none\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                cfg = load_config(env_path=env_file)
                provider = create_provider(cfg)
                self.assertIsInstance(provider, SDWebUIImageProvider)
                assert isinstance(provider, SDWebUIImageProvider)
                self.assertEqual(provider.lora, "")


if __name__ == "__main__":
    unittest.main()

"""Tests for configuration, image client caching, and end-to-end CLI pipeline."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.cli import main, resolve_max_colors
from src.config import load_config
from src.image_gen import ImageGenError, ImageGeneratorClient, ImageProvider


class _DummyImageProvider(ImageProvider):
    name = "dummy"

    def __init__(self, png_bytes: bytes) -> None:
        self._png_bytes = png_bytes
        self.calls = 0

    def generate(
        self,
        prompt: str,
        gen_size: tuple[int, int],
        target_size: tuple[int, int],
        ref_path: Path | None = None,
    ) -> bytes:
        self.calls += 1
        return self._png_bytes


def _make_sample_png_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    arr = np.full((size[1], size[0], 4), [255, 0, 255, 255], dtype=np.uint8)
    arr[16:48, 16:48] = [220, 30, 30, 255]
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class CliAndConfigTests(unittest.TestCase):
    def test_load_config_reads_dotenv_without_overriding_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "IMAGE_PROVIDER=openai\nIMAGE_API_KEY=from_dotenv\nDEFAULT_STYLE=16bit-topdown\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"IMAGE_PROVIDER": "gemini"}, clear=True):
                cfg = load_config(env_path=env_file)
                self.assertEqual(cfg.image_provider, "gemini")
                self.assertEqual(cfg.image_api_key, "from_dotenv")
                self.assertEqual(cfg.default_style, "16bit-topdown")

    def test_image_generator_client_caches_by_prompt(self) -> None:
        sample = _make_sample_png_bytes()
        provider = _DummyImageProvider(sample)
        with tempfile.TemporaryDirectory() as tmpdir:
            client = ImageGeneratorClient(provider=provider, cache_dir=Path(tmpdir), model_tag="m1")
            res1 = client.generate_or_load("red potion", (1024, 1024), (32, 32))
            res2 = client.generate_or_load("red potion", (1024, 1024), (32, 32))
            self.assertFalse(res1.from_cache)
            self.assertTrue(res2.from_cache)
            self.assertEqual(provider.calls, 1)
            self.assertTrue(res1.cache_path.is_file())

    def test_cli_offline_raw_image_skip_aseprite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.png"
            raw_path.write_bytes(_make_sample_png_bytes((128, 64)))
            out_dir = root / "output"
            cache_dir = root / "cache"

            rc = main(
                [
                    "--prompt",
                    "nhân vật hiệp sĩ đi bộ",
                    "--type",
                    "anim",
                    "--size",
                    "16x16",
                    "--frames",
                    "2",
                    "--cols",
                    "2",
                    "--raw-image",
                    str(raw_path),
                    "--output-dir",
                    str(out_dir),
                    "--cache-dir",
                    str(cache_dir),
                    "--name",
                    "knight_walk",
                    "--skip-aseprite",
                ]
            )
            self.assertEqual(rc, 0)
            out_png = out_dir / "knight_walk.png"
            self.assertTrue(out_png.is_file())
            with Image.open(out_png) as img:
                self.assertEqual(img.size, (32, 16))

    def test_cli_with_mocked_aseprite_produces_aseprite_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.png"
            raw_path.write_bytes(_make_sample_png_bytes((64, 64)))
            fake_exe = root / "Aseprite.exe"
            fake_exe.write_bytes(b"stub")
            out_dir = root / "output"
            cache_dir = root / "cache"

            def fake_run(cmd, capture_output, text, check):
                (out_dir / "potion.aseprite").write_bytes(b"ase")
                (out_dir / "potion.png").write_bytes(_make_sample_png_bytes((32, 32)))
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch.dict(os.environ, {"ASEPRITE_PATH": str(fake_exe)}, clear=False), patch(
                "subprocess.run", side_effect=fake_run
            ):
                rc = main(
                    [
                        "--prompt",
                        "bình máu đỏ",
                        "--type",
                        "item",
                        "--size",
                        "32x32",
                        "--raw-image",
                        str(raw_path),
                        "--output-dir",
                        str(out_dir),
                        "--cache-dir",
                        str(cache_dir),
                        "--name",
                        "potion",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "potion.aseprite").is_file())
            self.assertTrue((out_dir / "potion.png").is_file())

    def test_cli_missing_provider_config_reports_error_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"IMAGE_PROVIDER": "openai", "IMAGE_API_KEY": "", "LLM_PROVIDER": "none"},
                clear=True,
            ):
                rc = main(
                    [
                        "--prompt",
                        "red potion",
                        "--output-dir",
                        tmpdir,
                        "--cache-dir",
                        tmpdir,
                    ]
                )
            self.assertEqual(rc, 2)

    def test_cli_idle_from_single_skip_aseprite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "single_char.png"
            raw_path.write_bytes(_make_sample_png_bytes((96, 96)))
            out_dir = root / "output"
            cache_dir = root / "cache"

            rc = main(
                [
                    "--prompt",
                    "female mage",
                    "--idle-from-single",
                    "--idle-style",
                    "bob",
                    "--bob-px",
                    "1",
                    "--size",
                    "32x32",
                    "--frames",
                    "6",
                    "--cols",
                    "3",
                    "--raw-image",
                    str(raw_path),
                    "--output-dir",
                    str(out_dir),
                    "--cache-dir",
                    str(cache_dir),
                    "--name",
                    "mage_idle",
                    "--skip-aseprite",
                ]
            )
            self.assertEqual(rc, 0)
            out_png = out_dir / "mage_idle.png"
            self.assertTrue(out_png.is_file())
            with Image.open(out_png) as img:
                self.assertEqual(img.size, (96, 64))

    def test_cli_hibit_side_with_facing_and_export_gif(self) -> None:
        self.assertEqual(resolve_max_colors("hibit-anime-side", False, None), 64)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "side_char.png"
            raw_path.write_bytes(_make_sample_png_bytes((128, 128)))
            out_dir = root / "output"
            cache_dir = root / "cache"

            rc = main(
                [
                    "--prompt",
                    "gothic demoness",
                    "--style",
                    "hibit-anime-side",
                    "--facing",
                    "east",
                    "--idle-from-single",
                    "--size",
                    "128x128",
                    "--frames",
                    "4",
                    "--cols",
                    "2",
                    "--fps",
                    "8",
                    "--raw-image",
                    str(raw_path),
                    "--output-dir",
                    str(out_dir),
                    "--cache-dir",
                    str(cache_dir),
                    "--name",
                    "demoness_idle",
                    "--export-gif",
                    "--skip-aseprite",
                ]
            )
            self.assertEqual(rc, 0)
            out_png = out_dir / "demoness_idle.png"
            out_gif = out_dir / "demoness_idle.gif"
            self.assertTrue(out_png.is_file())
            self.assertTrue(out_gif.is_file())
            with Image.open(out_gif) as gif_img:
                self.assertEqual(gif_img.size, (128, 128))
                self.assertEqual(getattr(gif_img, "n_frames", 1), 4)

    def test_cli_character_8bit_chibi_topdown_32x32(self) -> None:
        self.assertEqual(resolve_max_colors("8bit-chibi-topdown", False, None), 16)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "chibi.png"
            raw_path.write_bytes(_make_sample_png_bytes((64, 64)))
            out_dir = root / "output"
            cache_dir = root / "cache"

            rc = main(
                [
                    "--prompt",
                    "chibi knight in blue armor",
                    "--type",
                    "character",
                    "--style",
                    "8bit-chibi-topdown",
                    "--size",
                    "32x32",
                    "--facing",
                    "front",
                    "--bg-color",
                    "#D0D0D0",
                    "--raw-image",
                    str(raw_path),
                    "--output-dir",
                    str(out_dir),
                    "--cache-dir",
                    str(cache_dir),
                    "--name",
                    "chibi_knight",
                    "--skip-aseprite",
                ]
            )
            self.assertEqual(rc, 0)
            out_png = out_dir / "chibi_knight.png"
            self.assertTrue(out_png.is_file())
            with Image.open(out_png) as img:
                self.assertEqual(img.size, (32, 32))


if __name__ == "__main__":
    unittest.main()

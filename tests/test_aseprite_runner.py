"""Tests for src.aseprite_runner."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.aseprite_runner import (
    AsepriteNotFoundError,
    find_aseprite,
    generate_lua_script,
    run_aseprite_build,
)


class AsepriteRunnerTests(unittest.TestCase):
    def test_find_aseprite_via_configured_path_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_exe = Path(tmpdir) / "Aseprite.exe"
            fake_exe.write_bytes(b"stub")

            found_cfg = find_aseprite(str(fake_exe))
            self.assertEqual(found_cfg, fake_exe.resolve())

            with patch.dict(os.environ, {"ASEPRITE_PATH": str(fake_exe)}, clear=False):
                found_env = find_aseprite()
                self.assertEqual(found_env, fake_exe.resolve())

    def test_find_aseprite_raises_clear_error_when_missing(self) -> None:
        with patch.dict(os.environ, {"ASEPRITE_PATH": ""}, clear=False), patch(
            "shutil.which", return_value=None
        ), patch("src.aseprite_runner._candidate_aseprite_paths", return_value=[]):
            with self.assertRaises(AsepriteNotFoundError) as ctx:
                find_aseprite()
            self.assertIn("ASEPRITE_PATH", str(ctx.exception))

    def test_generate_lua_script_for_item(self) -> None:
        lua = generate_lua_script(
            input_png=Path("C:/tmp/potion_processed.png"),
            out_aseprite=Path("C:/tmp/potion.aseprite"),
            out_png=Path("C:/tmp/potion.png"),
            asset_type="item",
            frame_size=(32, 32),
            frames=1,
            cols=1,
            fps=8.0,
        )
        self.assertIn('baseLayer.name = "Base"', lua)
        self.assertIn('detailLayer.name = "Detail & Shading"', lua)
        self.assertIn("Sprite(FRAME_W, FRAME_H, ColorMode.RGB)", lua)
        self.assertIn("sprite:saveAs(OUT_ASEPRITE)", lua)
        self.assertIn("sprite:saveCopyAs(OUT_PNG)", lua)

    def test_generate_lua_script_for_anim(self) -> None:
        lua = generate_lua_script(
            input_png=Path("/tmp/knight_processed.png"),
            out_aseprite=Path("/tmp/knight.aseprite"),
            out_png=Path("/tmp/knight.png"),
            asset_type="anim",
            frame_size=(32, 32),
            frames=6,
            cols=4,
            fps=10.0,
        )
        self.assertIn("local FRAMES = 6", lua)
        self.assertIn("local COLS = 4", lua)
        self.assertIn("local ROWS = 2", lua)
        self.assertIn("local FRAME_DURATION = 0.100000", lua)
        self.assertIn("app.command.ExportSpriteSheet", lua)
        self.assertIn("SpriteSheetType.ROWS", lua)

    def test_run_aseprite_build_invokes_cli_and_verifies_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_exe = root / "aseprite.exe"
            fake_exe.write_bytes(b"stub")
            in_png = root / "in.png"
            in_png.write_bytes(b"png")
            out_ase = root / "out" / "hero.aseprite"
            out_png = root / "out" / "hero.png"

            def fake_subprocess_run(cmd, capture_output, text, check):
                out_ase.write_bytes(b"ase")
                out_png.write_bytes(b"png")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("subprocess.run", side_effect=fake_subprocess_run) as mocked_run:
                res = run_aseprite_build(
                    input_png=in_png,
                    out_aseprite=out_ase,
                    out_png=out_png,
                    asset_type="anim",
                    frame_size=(32, 32),
                    frames=4,
                    cols=4,
                    fps=8.0,
                    configured_aseprite_path=str(fake_exe),
                    work_dir=root / "work",
                )

            self.assertEqual(mocked_run.call_count, 1)
            called_cmd = mocked_run.call_args[0][0]
            self.assertEqual(called_cmd[0], str(fake_exe.resolve()))
            self.assertEqual(called_cmd[1:3], ["-b", "--script"])
            self.assertTrue(res.lua_script_path.is_file())
            self.assertTrue(res.aseprite_path.is_file())
            self.assertTrue(res.png_path.is_file())


if __name__ == "__main__":
    unittest.main()

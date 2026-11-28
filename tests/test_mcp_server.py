"""Tests for the DrawTool Aseprite MCP stdio server."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.config import AppConfig
from src.mcp_server import handle_jsonrpc_request

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_sample_raw(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    arr = np.full((size[1], size[0], 4), [255, 0, 255, 255], dtype=np.uint8)
    arr[16:48, 16:48] = [220, 40, 40, 255]
    Image.fromarray(arr, mode="RGBA").save(path, format="PNG")


class McpServerTests(unittest.TestCase):
    def test_initialize_ping_list_and_notification(self) -> None:
        init_resp = handle_jsonrpc_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
        )
        assert init_resp is not None
        self.assertEqual(init_resp["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(init_resp["result"]["serverInfo"]["name"], "drawtool-aseprite")

        notif = handle_jsonrpc_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertIsNone(notif)

        ping_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "ping"})
        assert ping_resp is not None
        self.assertEqual(ping_resp["result"], {})

        list_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        assert list_resp is not None
        tool_names = {t["name"] for t in list_resp["result"]["tools"]}
        self.assertEqual(
            tool_names,
            {
                "build_pixel_prompt",
                "process_image_to_aseprite",
                "create_idle_animation",
                "generate_game_asset",
                "check_aseprite_status",
            },
        )

        unknown = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 4, "method": "nope"})
        assert unknown is not None
        self.assertEqual(unknown["error"]["code"], -32601)

    def test_build_pixel_prompt_tool(self) -> None:
        cfg = AppConfig()
        resp = handle_jsonrpc_request(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "build_pixel_prompt",
                    "arguments": {
                        "prompt": "red healing potion",
                        "type": "anim",
                        "size": "32x32",
                        "frames": 6,
                        "cols": 4,
                    },
                },
            },
            config=cfg,
        )
        assert resp is not None
        self.assertFalse(resp["result"]["isError"])
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("8-bit pixel art", data["final_prompt"])
        self.assertEqual((data["cols"], data["rows"]), (4, 2))
        self.assertEqual(data["gen_size"], "1536x1024")

    def test_process_image_to_aseprite_skip_and_mocked_aseprite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.png"
            _write_sample_raw(raw_path)
            out_dir = root / "out"
            cfg = AppConfig(cache_dir=root / "cache", output_dir=out_dir)

            resp_skip = handle_jsonrpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "tools/call",
                    "params": {
                        "name": "process_image_to_aseprite",
                        "arguments": {
                            "raw_image_path": str(raw_path),
                            "name": "potion_skip",
                            "type": "item",
                            "size": "32x32",
                            "output_dir": str(out_dir),
                            "skip_aseprite": True,
                        },
                    },
                },
                config=cfg,
            )
            assert resp_skip is not None
            self.assertFalse(resp_skip["result"]["isError"])
            skip_data = json.loads(resp_skip["result"]["content"][0]["text"])
            self.assertEqual(skip_data["status"], "png_only")
            self.assertTrue(Path(skip_data["png_path"]).is_file())

            fake_exe = root / "Aseprite.exe"
            fake_exe.write_bytes(b"stub")
            cfg_ase = AppConfig(
                aseprite_path=str(fake_exe),
                cache_dir=root / "cache2",
                output_dir=out_dir,
            )

            def fake_run(cmd, capture_output, text, check):
                (out_dir / "potion_full.aseprite").write_bytes(b"ase")
                (out_dir / "potion_full.png").write_bytes(raw_path.read_bytes())
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                resp_full = handle_jsonrpc_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 12,
                        "method": "tools/call",
                        "params": {
                            "name": "process_image_to_aseprite",
                            "arguments": {
                                "raw_image_path": str(raw_path),
                                "name": "potion_full",
                                "type": "item",
                                "size": "32x32",
                                "output_dir": str(out_dir),
                            },
                        },
                    },
                    config=cfg_ase,
                )
            assert resp_full is not None
            full_data = json.loads(resp_full["result"]["content"][0]["text"])
            self.assertEqual(full_data["status"], "ok")
            self.assertTrue(Path(full_data["aseprite_path"]).is_file())
            self.assertTrue(Path(full_data["png_path"]).is_file())

    def test_create_idle_animation_tool_skip_and_mocked_aseprite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "mage_single.png"
            _write_sample_raw(raw_path, size=(96, 96))
            out_dir = root / "out"
            cfg = AppConfig(cache_dir=root / "cache", output_dir=out_dir)

            resp = handle_jsonrpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 21,
                    "method": "tools/call",
                    "params": {
                        "name": "create_idle_animation",
                        "arguments": {
                            "raw_image_path": str(raw_path),
                            "name": "mage_idle",
                            "size": "48x48",
                            "frames": 6,
                            "cols": 3,
                            "fps": 6,
                            "idle_style": "bob",
                            "bob_px": 1,
                            "style": "anime-rpg-topdown",
                            "output_dir": str(out_dir),
                            "skip_aseprite": True,
                        },
                    },
                },
                config=cfg,
            )
            assert resp is not None
            self.assertFalse(resp["result"]["isError"])
            data = json.loads(resp["result"]["content"][0]["text"])
            self.assertEqual(data["status"], "png_only")
            self.assertEqual(data["sheet_size"], "144x96")
            self.assertEqual((data["cols"], data["rows"], data["frames"]), (3, 2, 6))
            self.assertEqual(data["idle_style"], "bob")
            self.assertTrue(Path(data["png_path"]).is_file())

    def test_mcp_hibit_side_facing_auto_bob_and_export_gif(self) -> None:
        cfg = AppConfig()
        prompt_resp = handle_jsonrpc_request(
            {
                "jsonrpc": "2.0",
                "id": 31,
                "method": "tools/call",
                "params": {
                    "name": "build_pixel_prompt",
                    "arguments": {
                        "prompt": "gothic demoness",
                        "type": "item",
                        "size": "256x256",
                        "style": "hibit-anime-side",
                        "facing": "east",
                    },
                },
            },
            config=cfg,
        )
        assert prompt_resp is not None
        self.assertFalse(prompt_resp["result"]["isError"])
        p_data = json.loads(prompt_resp["result"]["content"][0]["text"])
        self.assertEqual(p_data["facing"], "east")
        self.assertIn("32-bit anime pixel art", p_data["final_prompt"])
        self.assertIn("facing right (east)", p_data["final_prompt"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "side_256.png"
            _write_sample_raw(raw_path, size=(256, 256))
            out_dir = root / "out"
            cfg_tmp = AppConfig(cache_dir=root / "cache", output_dir=out_dir)

            idle_resp = handle_jsonrpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 32,
                    "method": "tools/call",
                    "params": {
                        "name": "create_idle_animation",
                        "arguments": {
                            "raw_image_path": str(raw_path),
                            "name": "side_idle_256",
                            "size": "256x256",
                            "frames": 4,
                            "cols": 2,
                            "fps": 8,
                            "style": "hibit-anime-side",
                            "output_dir": str(out_dir),
                            "skip_aseprite": True,
                            "export_gif": True,
                        },
                    },
                },
                config=cfg_tmp,
            )
            assert idle_resp is not None
            self.assertFalse(idle_resp["result"]["isError"])
            i_data = json.loads(idle_resp["result"]["content"][0]["text"])
            self.assertEqual(i_data["bob_px"], 3)
            self.assertIsNotNone(i_data["gif_path"])
            self.assertTrue(Path(i_data["gif_path"]).is_file())
            with Image.open(i_data["gif_path"]) as gif_img:
                self.assertEqual(gif_img.size, (256, 256))
                self.assertEqual(getattr(gif_img, "n_frames", 1), 4)

    def test_check_aseprite_status_does_not_leak_secrets(self) -> None:
        cfg = AppConfig(image_provider="gemini", image_api_key="AIzaSUPERSECRET123")
        resp = handle_jsonrpc_request(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "tools/call",
                "params": {"name": "check_aseprite_status", "arguments": {}},
            },
            config=cfg,
        )
        assert resp is not None
        text = resp["result"]["content"][0]["text"]
        self.assertNotIn("AIzaSUPERSECRET123", text)
        data = json.loads(text)
        self.assertTrue(data["has_image_api_key"])

    def test_real_stdio_subprocess_roundtrip(self) -> None:
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "build_pixel_prompt",
                    "arguments": {"prompt": "gold coin", "type": "item", "size": "16x16"},
                },
            },
        ]
        stdin_payload = "\n".join(json.dumps(r) for r in requests) + "\n"
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "mcp_server.py")],
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        self.assertEqual(len(lines), 3)
        r1, r2, r3 = (json.loads(ln) for ln in lines)
        self.assertEqual(r1["id"], 1)
        self.assertEqual(r2["id"], 2)
        self.assertEqual(r3["id"], 3)
        prompt_data = json.loads(r3["result"]["content"][0]["text"])
        self.assertIn("gold coin", prompt_data["final_prompt"])

    def test_mcp_character_8bit_chibi_topdown_32x32(self) -> None:
        cfg = AppConfig()
        prompt_resp = handle_jsonrpc_request(
            {
                "jsonrpc": "2.0",
                "id": 41,
                "method": "tools/call",
                "params": {
                    "name": "build_pixel_prompt",
                    "arguments": {
                        "prompt": "chibi mage girl",
                        "type": "character",
                        "size": "32x32",
                        "style": "8bit-chibi-topdown",
                        "facing": "front",
                        "bg_color": "#D0D0D0",
                    },
                },
            },
            config=cfg,
        )
        assert prompt_resp is not None
        self.assertFalse(prompt_resp["result"]["isError"])
        p_data = json.loads(prompt_resp["result"]["content"][0]["text"])
        self.assertEqual(p_data["asset_type"], "character")
        self.assertEqual(p_data["facing"], "front")
        self.assertIn("2.5-head chibi", p_data["final_prompt"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "chibi_raw.png"
            _write_sample_raw(raw_path, size=(64, 64))
            out_dir = root / "out"
            cfg_tmp = AppConfig(cache_dir=root / "cache", output_dir=out_dir)

            proc_resp = handle_jsonrpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 42,
                    "method": "tools/call",
                    "params": {
                        "name": "process_image_to_aseprite",
                        "arguments": {
                            "raw_image_path": str(raw_path),
                            "name": "chibi_mage_32",
                            "type": "character",
                            "size": "32x32",
                            "style": "8bit-chibi-topdown",
                            "output_dir": str(out_dir),
                            "skip_aseprite": True,
                        },
                    },
                },
                config=cfg_tmp,
            )
            assert proc_resp is not None
            self.assertFalse(proc_resp["result"]["isError"])
            proc_data = json.loads(proc_resp["result"]["content"][0]["text"])
            self.assertEqual(proc_data["asset_type"], "character")
            self.assertEqual(proc_data["sheet_size"], "32x32")
            self.assertTrue(Path(proc_data["png_path"]).is_file())


if __name__ == "__main__":
    unittest.main()

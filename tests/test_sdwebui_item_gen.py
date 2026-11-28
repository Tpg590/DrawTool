"""Tests for .agents/skills/sdwebui-item-enhancer/scripts/sdwebui_item_gen.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / ".agents" / "skills" / "sdwebui-item-enhancer" / "scripts" / "sdwebui_item_gen.py"

_SPEC = importlib.util.spec_from_file_location("sdwebui_item_gen", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["sdwebui_item_gen"] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class SDWebUIItemGenTests(unittest.TestCase):
    def test_load_negative_prompt_presets_from_markdown(self) -> None:
        neg_file = REPO_ROOT / "negative_prompt.md"
        item_neg = _MODULE.load_negative_prompt(neg_file, preset="item_icon")
        char_neg = _MODULE.load_negative_prompt(neg_file, preset="character")

        self.assertIn("photorealistic", item_neg)
        self.assertIn("multiple items", item_neg)
        self.assertNotIn("Tên preset", item_neg)

        self.assertIn("photorealistic", char_neg)
        self.assertIn("multiple characters", char_neg)
        self.assertIn("turnaround", char_neg)
        self.assertNotIn("item_general", char_neg)
        self.assertNotIn("Tên preset", char_neg)

    def test_build_painted_sdxl_prompt_character_preserves_chibi(self) -> None:
        pos, neg = _MODULE.build_painted_sdxl_prompt(
            raw_prompt="cute 2.5-head chibi knight, blue armor, 32x32 8-bit pixel art, no ground shadow",
            base_negative="photorealistic, blurry",
            mode="character",
            facing="front",
        )
        self.assertIn("chibi", pos.lower())
        self.assertIn("facing forward toward camera", pos)
        self.assertIn("unarmed", pos)
        self.assertIn("#D0D0D0", pos)
        self.assertNotIn("32x32", pos)
        self.assertIn("ground shadow", neg)
        self.assertIn("sword", neg)

    def test_build_painted_sdxl_prompt_allows_weapon_when_requested(self) -> None:
        pos, neg = _MODULE.build_painted_sdxl_prompt(
            raw_prompt="chibi knight holding a silver sword",
            base_negative="photorealistic, blurry",
            mode="character",
            facing="front",
        )
        self.assertNotIn("unarmed", pos)
        self.assertNotIn("sword", neg)

    def test_pixel_craft_palette_ref_restricts_colors(self) -> None:
        # Reference sprite with 2 distinct colors
        ref_arr = np.full((32, 32, 4), [0, 0, 0, 0], dtype=np.uint8)
        ref_arr[4:28, 4:28] = [240, 210, 190, 255]
        ref_arr[10:20, 10:20] = [40, 25, 45, 255]
        ref_img = Image.fromarray(ref_arr, mode="RGBA")

        # High-res sprite with bright skin and a dark detail stripe
        arr = np.full((256, 256, 3), 0xD0, dtype=np.uint8)
        arr[20:236, 20:236] = [245, 220, 200]
        arr[96:112, 80:176] = [20, 15, 25]
        painted = Image.fromarray(arr, mode="RGB")

        crafted = _MODULE.pixel_craft(
            painted,
            pixel_size=64,
            max_colors=16,
            anchor_bottom=True,
            palette_ref=ref_img,
        )
        g64 = np.array(crafted.resize((64, 64), Image.Resampling.NEAREST))
        fg = g64[g64[:, :, 3] > 0]
        # All opaque colors must come from ref_img + outline (#341C38)
        allowed = {(240, 210, 190), (40, 25, 45), (0x34, 0x1C, 0x38)}
        unique_cols = {tuple(int(v) for v in c[:3]) for c in fg}
        self.assertTrue(unique_cols.issubset(allowed))

    def test_pixel_craft_character_anchors_bottom_and_limits_colors(self) -> None:
        arr = np.full((128, 128, 4), [0xD0, 0xD0, 0xD0, 255], dtype=np.uint8)
        for y in range(20, 100):
            for x in range(36, 92):
                arr[y, x] = [(x * 3) % 240, (y * 2) % 240, 140, 255]
        painted = Image.fromarray(arr, mode="RGBA").convert("RGB")

        crafted = _MODULE.pixel_craft(
            painted,
            pixel_size=32,
            max_colors=16,
            anchor_bottom=True,
        )
        self.assertEqual(crafted.size, (512, 512))
        self.assertEqual(crafted.mode, "RGBA")
        grid32 = crafted.resize((32, 32), Image.Resampling.NEAREST)
        g_arr = np.array(grid32)
        fg_mask = g_arr[:, :, 3] > 0
        self.assertTrue(np.any(fg_mask))
        ys, _ = np.where(fg_mask)
        # Bottom-most pixel (including 1px outline) sits near row 30 (32 - 2)
        self.assertGreaterEqual(int(ys.max()), 29)
        self.assertLessEqual(int(ys.max()), 31)
        unique_opaque_colors = np.unique(g_arr[fg_mask, :3], axis=0)
        # <= 16 interior colors + 1 outline
        self.assertLessEqual(len(unique_opaque_colors), 17)

    def test_remove_bg_floodfill_removes_enclosed_interior_holes(self) -> None:
        arr = np.full((128, 128, 3), 0xD0, dtype=np.uint8)
        # Draw a solid orange ring with a trapped #D0D0D0 hole in the center
        arr[20:108, 20:108] = [240, 140, 40]
        arr[44:84, 44:84] = [0xD0, 0xD0, 0xD0]
        img = Image.fromarray(arr, mode="RGB")

        out = _MODULE.remove_bg_floodfill(img)
        out_arr = np.array(out)
        # Exterior and interior hole should both be transparent
        self.assertEqual(int(out_arr[5, 5, 3]), 0)
        self.assertEqual(int(out_arr[64, 64, 3]), 0)
        # Ring itself remains opaque
        self.assertEqual(int(out_arr[30, 30, 3]), 255)

    def test_character_default_size_is_64x64(self) -> None:
        from src.mcp_server import call_tool

        res, is_err = call_tool(
            "build_pixel_prompt",
            {"prompt": "cute knight", "type": "character"},
        )
        self.assertFalse(is_err)
        self.assertEqual(res["target_size"], "64x64")

        res_item, is_err_item = call_tool(
            "build_pixel_prompt",
            {"prompt": "red potion", "type": "item"},
        )
        self.assertFalse(is_err_item)
        self.assertEqual(res_item["target_size"], "32x32")


if __name__ == "__main__":
    unittest.main()

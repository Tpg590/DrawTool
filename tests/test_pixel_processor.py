"""Tests for src.pixel_processor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pixel_processor import (
    parse_palette,
    parse_size,
    process_anim,
    process_item,
    process_tile,
    quantize_image,
    remove_background,
)


class PixelProcessorTests(unittest.TestCase):
    def test_parse_size(self) -> None:
        self.assertEqual(parse_size("32x32"), (32, 32))
        self.assertEqual(parse_size(" 64X16 "), (64, 16))
        with self.assertRaises(ValueError):
            parse_size("abc")
        with self.assertRaises(ValueError):
            parse_size("2x2")

    def test_parse_palette_gpl_and_hex_string(self) -> None:
        hex_colors = parse_palette("#000000, #fff, ff0000")
        self.assertEqual(hex_colors, [(0, 0, 0), (255, 255, 255), (255, 0, 0)])

        with tempfile.TemporaryDirectory() as tmpdir:
            gpl_file = Path(tmpdir) / "pico.gpl"
            gpl_file.write_text(
                "GIMP Palette\nName: Test\nColumns: 2\n#\n  20  30  40  Dark\n255 128   0  Orange\n",
                encoding="utf-8",
            )
            gpl_colors = parse_palette(str(gpl_file))
            self.assertEqual(gpl_colors, [(20, 30, 40), (255, 128, 0)])

    def test_remove_background_preserves_enclosed_interior_same_color(self) -> None:
        # Solid white background, red hollow square, white interior inside the square
        arr = np.full((32, 32, 4), [255, 255, 255, 255], dtype=np.uint8)
        arr[6:26, 6:26, :3] = [220, 20, 20]
        arr[10:22, 10:22, :3] = [255, 255, 255]
        img = Image.fromarray(arr, mode="RGBA")

        cleaned = remove_background(img, tolerance=15)
        out = np.array(cleaned)
        # Exterior corners must be transparent
        self.assertEqual(out[0, 0, 3], 0)
        self.assertEqual(out[31, 31, 3], 0)
        # Red outline must remain opaque
        self.assertEqual(out[7, 7, 3], 255)
        # Enclosed white interior must remain opaque (flood fill doesn't punch through)
        self.assertEqual(out[16, 16, 3], 255)

    def test_process_item_crops_centers_nearest_and_limits_colors(self) -> None:
        # Create 128x128 noisy image on solid magenta backdrop
        arr = np.full((128, 128, 4), [255, 0, 255, 255], dtype=np.uint8)
        # Draw a small off-center gradient block in bottom-right quadrant
        for y in range(60, 100):
            for x in range(70, 110):
                arr[y, x] = [(x * 3) % 255, (y * 2) % 255, 100, 255]
        raw = Image.fromarray(arr, mode="RGBA")

        out = process_item(raw, target_size=(32, 32), max_colors=8)
        self.assertEqual(out.size, (32, 32))
        out_arr = np.array(out)
        # Corners transparent, center opaque
        self.assertEqual(out_arr[0, 0, 3], 0)
        self.assertEqual(out_arr[16, 16, 3], 255)
        # Check color count on opaque pixels <= 8
        opaque = out_arr[out_arr[:, :, 3] == 255, :3]
        unique_colors = np.unique(opaque, axis=0)
        self.assertLessEqual(len(unique_colors), 8)

    def test_quantize_with_explicit_palette(self) -> None:
        arr = np.full((16, 16, 4), [250, 10, 10, 255], dtype=np.uint8)
        arr[0:8, 0:8] = [10, 10, 240, 255]
        arr[0:4, 0:4, 3] = 0
        img = Image.fromarray(arr, mode="RGBA")
        palette = [(255, 0, 0), (0, 0, 255)]
        q = quantize_image(img, palette=palette)
        q_arr = np.array(q)
        opaque = q_arr[q_arr[:, :, 3] == 255, :3]
        unique_tuples = {tuple(map(int, c)) for c in np.unique(opaque, axis=0)}
        self.assertTrue(unique_tuples.issubset(set(palette)))

    def test_process_anim_slices_grid_and_shares_palette(self) -> None:
        # 2x2 grid for 4 frames (each cell 64x64 -> 128x128)
        arr = np.full((128, 128, 4), [0, 255, 0, 255], dtype=np.uint8)
        # Draw sprite in each cell with slightly shifting X pose and varied shades
        for idx in range(4):
            col = idx % 2
            row = idx // 2
            cx = col * 64 + 16 + idx * 2
            cy = row * 64 + 16
            for dy in range(32):
                for dx in range(24):
                    arr[cy + dy, cx + dx] = [200 - idx * 10, 40 + dy * 4, 40 + dx * 4, 255]

        raw = Image.fromarray(arr, mode="RGBA")
        sheet, frames = process_anim(
            raw,
            target_size=(32, 32),
            frames=4,
            cols=2,
            max_colors=12,
        )
        self.assertEqual(sheet.size, (64, 64))
        self.assertEqual(len(frames), 4)
        for frame in frames:
            self.assertEqual(frame.size, (32, 32))
            self.assertIsNotNone(frame.getbbox())

        sheet_arr = np.array(sheet)
        opaque = sheet_arr[sheet_arr[:, :, 3] == 255, :3]
        self.assertLessEqual(len(np.unique(opaque, axis=0)), 12)

    def test_process_tile_wraps_opposite_borders_and_quantizes(self) -> None:
        # Left half bright green, right half dark green -> toroidal seam blend aligns edges
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[:, :32] = [80, 200, 80, 255]
        arr[:, 32:] = [20, 80, 20, 255]
        arr[:16, :] = [140, 220, 60, 255]
        raw = Image.fromarray(arr, mode="RGBA")

        tile = process_tile(raw, target_size=(16, 16), max_colors=8, seam_blend=2)
        self.assertEqual(tile.size, (16, 16))
        t_arr = np.array(tile)
        # Leftmost and rightmost columns must match at the seam
        np.testing.assert_array_equal(t_arr[:, 0, :3], t_arr[:, -1, :3])
        # Top and bottom rows must match at the seam
        np.testing.assert_array_equal(t_arr[0, :, :3], t_arr[-1, :, :3])
        unique_colors = np.unique(t_arr[:, :, :3].reshape(-1, 3), axis=0)
        self.assertLessEqual(len(unique_colors), 8)


if __name__ == "__main__":
    unittest.main()

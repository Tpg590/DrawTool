"""Tests for src.pixel_processor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pixel_processor import (
    export_animated_gif,
    filter_small_components,
    generate_idle_frames,
    parse_palette,
    parse_size,
    process_anim,
    process_asset,
    process_character,
    process_idle_from_single,
    process_item,
    process_tile,
    quantize_image,
    remove_background,
    resize_pixel_art,
    resolve_default_bob_px,
    slice_spritesheet_frames,
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

    def test_filter_small_components_removes_corner_speck_and_keeps_sprite(self) -> None:
        arr = np.zeros((200, 200, 4), dtype=np.uint8)
        arr[40:120, 70:130] = [100, 150, 220, 255]  # 60x80 main character
        arr[196:198, 195:198] = [10, 20, 10, 255]    # 3x2 corner speck
        cleaned = filter_small_components(Image.fromarray(arr, mode="RGBA"))
        bbox = cleaned.getbbox()
        self.assertEqual(bbox, (70, 40, 130, 120))

    def test_remove_background_despills_chroma_fringe(self) -> None:
        arr = np.full((64, 64, 4), [0, 255, 0, 255], dtype=np.uint8)
        # Draw sprite core + 1px dark outline + 1px green-blended JPEG halo
        arr[20:44, 20:44] = [20, 180, 20, 255]  # green halo ring
        arr[21:43, 21:43] = [15, 15, 25, 255]   # dark outline
        arr[23:41, 23:41] = [220, 220, 230, 255]  # white hair/body interior
        cleaned = remove_background(Image.fromarray(arr, mode="RGBA"), tolerance=28, despill=True)
        out = np.array(cleaned)
        # Green-blended halo ring should be removed
        self.assertEqual(out[20, 20, 3], 0)
        # Outline and interior remain opaque and unharmed
        self.assertEqual(out[21, 21, 3], 255)
        self.assertEqual(out[32, 32, 3], 255)
        self.assertEqual(tuple(out[32, 32, :3]), (220, 220, 230))

    def test_resize_pixel_art_mode_preserves_crisp_blocks_over_nearest(self) -> None:
        # 10x10 grid of 10x10px blocks (100x100), with 1px noisy border on each block edge
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        for by in range(10):
            for bx in range(10):
                color = [20 + bx * 20, 30 + by * 20, 180, 255]
                y0, y1 = by * 10, (by + 1) * 10
                x0, x1 = bx * 10, (bx + 1) * 10
                arr[y0:y1, x0:x1] = [0, 0, 0, 255]  # 1px noisy border around block
                arr[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = color
        down = resize_pixel_art(Image.fromarray(arr, mode="RGBA"), (10, 10), mode="mode")
        d_arr = np.array(down)
        for by in range(10):
            for bx in range(10):
                expected = (20 + bx * 20, 30 + by * 20, 180, 255)
                self.assertEqual(tuple(map(int, d_arr[by, bx])), expected)

    def test_process_anim_centroid_alignment_prevents_horizontal_jitter_and_corner_speck_shift(self) -> None:
        # 2x1 grid (each cell 100x100 -> 200x100) on #00FF00 background
        arr = np.full((100, 200, 4), [0, 255, 0, 255], dtype=np.uint8)
        # Frame 0: centered character body (x=35..65, y=20..85)
        arr[20:85, 35:65] = [60, 120, 220, 255]
        # Frame 1: same body shifted + 4px side cape tip on right + 2x2 corner speck at bottom-right
        arr[20:85, 135:165] = [60, 120, 220, 255]
        arr[55:70, 165:175] = [60, 120, 220, 255]
        arr[97:99, 197:199] = [10, 40, 10, 255]  # corner speck that previously broke frame 2

        sheet, frames = process_anim(
            Image.fromarray(arr, mode="RGBA"),
            target_size=(32, 32),
            frames=2,
            cols=2,
            max_colors=8,
            downsample_mode="mode",
            align_mode="centroid",
        )
        self.assertEqual(sheet.size, (64, 32))
        b0 = frames[0].getbbox()
        b1 = frames[1].getbbox()
        assert b0 is not None and b1 is not None
        # Bottom feet alignment must match within 1px (no vertical jump from corner speck)
        self.assertLessEqual(abs(b0[3] - b1[3]), 1)

    def test_generate_idle_frames_and_process_idle_from_single(self) -> None:
        # Create a 96x96 raw character sprite on neutral gray #D0D0D0
        arr = np.full((96, 96, 4), [208, 208, 208, 255], dtype=np.uint8)
        arr[16:46, 30:66] = [240, 240, 245, 255]  # head/hair
        arr[46:72, 26:70] = [30, 110, 200, 255]   # cloak torso
        arr[72:84, 34:44] = [70, 50, 40, 255]     # left boot
        arr[72:84, 52:62] = [70, 50, 40, 255]     # right boot
        raw = Image.fromarray(arr, mode="RGBA")

        for idle_style in ("bob", "breathe", "float"):
            sheet, frames = process_idle_from_single(
                raw,
                target_size=(32, 32),
                frames=6,
                cols=3,
                bob_px=1,
                idle_style=idle_style,
                max_colors=8,
            )
            self.assertEqual(sheet.size, (96, 64))
            self.assertEqual(len(frames), 6)
            base_colors = {
                tuple(map(int, c))
                for c in np.unique(
                    np.array(frames[0])[np.array(frames[0])[:, :, 3] == 255, :3],
                    axis=0,
                )
            }
            # Frames must not invent new colors and must actually animate (not all identical)
            diffs = 0
            for fr in frames:
                f_arr = np.array(fr)
                f_colors = {
                    tuple(map(int, c))
                    for c in np.unique(f_arr[f_arr[:, :, 3] == 255, :3], axis=0)
                }
                self.assertTrue(f_colors.issubset(base_colors))
                if not np.array_equal(f_arr, np.array(frames[0])):
                    diffs += 1
            self.assertGreater(diffs, 0, msg=f"idle_style={idle_style} produced static frames")
            if idle_style in ("bob", "breathe"):
                # Feet row (bottom-most opaque row of frame 0) must stay grounded in all frames
                b0 = frames[0].getbbox()
                assert b0 is not None
                for fr in frames:
                    b = fr.getbbox()
                    assert b is not None
                    self.assertEqual(b[3], b0[3])

    def test_resolve_default_bob_px_scales_with_sprite_size(self) -> None:
        self.assertEqual(resolve_default_bob_px((48, 48)), 1)
        self.assertEqual(resolve_default_bob_px((128, 128)), 2)
        self.assertEqual(resolve_default_bob_px((256, 256)), 3)

    def test_slice_and_export_animated_gif_preserves_transparency_and_palette(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.full((64, 64, 4), [230, 230, 230, 255], dtype=np.uint8)
            arr[12:56, 20:44] = [180, 30, 45, 255]
            raw = Image.fromarray(arr, mode="RGBA")
            sheet, _ = process_idle_from_single(
                raw,
                target_size=(64, 64),
                frames=4,
                cols=2,
                bob_px=2,
                idle_style="bob",
                max_colors=64,
            )
            frames = slice_spritesheet_frames(sheet, (64, 64), frames=4, cols=2)
            self.assertEqual(len(frames), 4)

            gif_path = Path(tmpdir) / "anim.gif"
            export_animated_gif(frames, gif_path, fps=10.0)
            self.assertTrue(gif_path.is_file())

            with Image.open(gif_path) as gif_img:
                self.assertEqual(gif_img.size, (64, 64))
                self.assertEqual(getattr(gif_img, "n_frames", 1), 4)
                self.assertEqual(gif_img.info.get("duration"), 100)
                gif_img.seek(0)
                f0 = gif_img.convert("RGBA")
                f0_arr = np.array(f0)
                self.assertEqual(f0_arr[0, 0, 3], 0)
                self.assertEqual(f0_arr[32, 32, 3], 255)

            # Also verify >255 unique colors fallback quantization path in export_animated_gif
            many_colors = np.zeros((20, 20, 4), dtype=np.uint8)
            for y in range(20):
                for x in range(20):
                    many_colors[y, x] = [(x * 12) % 255, (y * 12) % 255, (x + y * 7) % 255, 255]
            many_colors[0, 0, 3] = 0
            hi_frame1 = Image.fromarray(many_colors, mode="RGBA")
            many_colors2 = many_colors.copy()
            many_colors2[10, 10] = [255, 255, 255, 255]
            hi_frame2 = Image.fromarray(many_colors2, mode="RGBA")
            gif_hi = Path(tmpdir) / "anim_hi.gif"
            export_animated_gif([hi_frame1, hi_frame2], gif_hi, fps=8.0)
            with Image.open(gif_hi) as g_hi:
                self.assertEqual(getattr(g_hi, "n_frames", 1), 2)
                self.assertEqual(np.array(g_hi.convert("RGBA"))[0, 0, 3], 0)

    def test_process_character_anchors_feet_and_quantizes_16_colors(self) -> None:
        # 128x128 sprite in upper-left quadrant on #D0D0D0 background
        arr = np.full((128, 128, 4), [208, 208, 208, 255], dtype=np.uint8)
        for y in range(10, 70):
            for x in range(15, 55):
                arr[y, x] = [(x * 4) % 250, (y * 3) % 250, 120, 255]
        raw = Image.fromarray(arr, mode="RGBA")

        out = process_character(raw, target_size=(32, 32), max_colors=16, pad=1)
        self.assertEqual(out.size, (32, 32))
        bbox = out.getbbox()
        assert bbox is not None
        # Bottom-anchored to th - pad = 31
        self.assertEqual(bbox[3], 31)
        out_arr = np.array(out)
        opaque = out_arr[out_arr[:, :, 3] == 255, :3]
        self.assertLessEqual(len(np.unique(opaque, axis=0)), 16)

        asset_out, meta = process_asset(raw, asset_type="character", target_size=(32, 32), max_colors=16)
        self.assertEqual(asset_out.size, (32, 32))
        self.assertEqual(meta["frames"], 1)
        self.assertEqual(meta["sheet_size"], (32, 32))


if __name__ == "__main__":
    unittest.main()

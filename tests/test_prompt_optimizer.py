"""Tests for src.prompt_optimizer."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from src.config import AppConfig
from src.prompt_optimizer import (
    DEFAULT_STYLE,
    LOCKED_8BIT_TOPDOWN_KEYWORDS,
    build_deterministic_prompt,
    compute_grid,
    optimize_prompt,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class PromptOptimizerTests(unittest.TestCase):
    def test_locked_8bit_keywords_present_across_all_types(self) -> None:
        for asset_type in ("item", "tile", "anim"):
            prompt = build_deterministic_prompt(
                subject="red health potion",
                asset_type=asset_type,
                size=(32, 32),
                frames=4,
                cols=4,
                style=DEFAULT_STYLE,
            )
            self.assertIn(LOCKED_8BIT_TOPDOWN_KEYWORDS, prompt)
            self.assertIn("red health potion", prompt)

    def test_anim_grid_layout_mentions_cols_rows_and_frames(self) -> None:
        cols, rows = compute_grid(frames=6, cols=4)
        self.assertEqual((cols, rows), (4, 2))
        prompt = build_deterministic_prompt(
            subject="knight walking",
            asset_type="anim",
            size=(32, 32),
            frames=6,
            cols=4,
        )
        self.assertIn("4 columns by 2 rows", prompt)
        self.assertIn("6 animation frames", prompt)

    def test_tile_layout_mentions_seamless_edges(self) -> None:
        prompt = build_deterministic_prompt(
            subject="grass with small flowers",
            asset_type="tile",
            size=(16, 16),
        )
        self.assertIn("seamless tileable", prompt)
        self.assertIn("16x16", prompt)

    def test_style_override_and_reference_hint(self) -> None:
        prompt = build_deterministic_prompt(
            subject="wooden chest",
            asset_type="item",
            size=(32, 32),
            style="16bit-topdown",
            has_ref=True,
        )
        self.assertIn("16-bit pixel art", prompt)
        self.assertNotIn(LOCKED_8BIT_TOPDOWN_KEYWORDS, prompt)
        self.assertIn("reference image", prompt)

    def test_optimize_prompt_uses_llm_only_for_subject_and_keeps_locked_style(self) -> None:
        cfg = AppConfig(llm_provider="openai", llm_api_key="test-key", llm_model="gpt-4o-mini")
        fake_resp = _FakeHTTPResponse(
            {"choices": [{"message": {"content": "crimson glass healing potion with cork stopper"}}]}
        )
        with patch("urllib.request.urlopen", return_value=fake_resp) as mocked:
            res = optimize_prompt(
                user_prompt="bình máu đỏ",
                asset_type="item",
                size=(32, 32),
                config=cfg,
                use_llm=True,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertTrue(res.used_llm)
        self.assertEqual(res.subject, "crimson glass healing potion with cork stopper")
        self.assertIn(LOCKED_8BIT_TOPDOWN_KEYWORDS, res.final_prompt)

    def test_optimize_prompt_no_llm_spends_zero_tokens(self) -> None:
        cfg = AppConfig(llm_provider="openai", llm_api_key="test-key")
        with patch("urllib.request.urlopen") as mocked:
            res = optimize_prompt(
                user_prompt="iron sword",
                asset_type="item",
                size=(32, 32),
                config=cfg,
                use_llm=False,
            )
        self.assertEqual(mocked.call_count, 0)
        self.assertFalse(res.used_llm)

    def test_anime_rpg_preset_bg_color_and_action_notes(self) -> None:
        prompt = build_deterministic_prompt(
            subject="white-haired female mage",
            asset_type="anim",
            size=(48, 48),
            frames=4,
            cols=4,
            style="anime-rpg-topdown",
            bg_color="neutral light gray (#D0D0D0)",
            action_notes="1: idle, 2: step left, 3: idle, 4: step right",
        )
        self.assertIn("16-bit anime RPG pixel art", prompt)
        self.assertIn("neutral light gray (#D0D0D0)", prompt)
        self.assertIn("frame sequence: 1: idle, 2: step left, 3: idle, 4: step right", prompt)


if __name__ == "__main__":
    unittest.main()

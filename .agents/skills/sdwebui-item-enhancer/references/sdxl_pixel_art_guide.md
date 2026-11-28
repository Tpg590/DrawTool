# SDXL Paint-First → Pixel-Craft Guide

## Why Direct `pixel-art-xl` or Template `img2img` Looks Crude

1. **Direct `pixel-art-xl` on SDXL**: Struggles with small isolated RPG items — often generates multi-item sprite sheets, broken silhouettes, or muddy mixed-resolution pixels.
2. **Crude `img2img` Templates**: Tracing a flat geometric circle/rectangle at `denoise 0.6` locks in a stiff, amateur silhouette without volumetric depth or proper glass/metal rendering.

## How Paint-First → Pixel-Craft Fixes It

1. **Stage 1 — Paint-First (`sd_xl_base_1.0`, `lora_weight = 0.0`)**:
   - SDXL `sd_xl_base_1.0` excels at clean 2D hand-painted JRPG/Stardew-style cel-shaded icons with volumetric lighting, glass refraction, specular highlights, and organic silhouettes.
   - Strip `"pixel art"`, `"16-bit"`, `"no ..."`, and `"chibi"` from the Stage 1 positive prompt and replace with:
     `2d game item icon, hand-painted RPG inventory icon, bold clean dark outline, rich cel shading, glossy highlights, warm vibrant colors, Stardew Valley item art style`
   - Move all negations + `sprite sheet, tileset, collage, grid, multiple items, character, hands, pedestal, ground, drop shadow` into `negative_prompt`.

2. **Stage 2 — Pixel-Craft Post-Processor**:
   - Flood-fill remove `#D0D0D0` background + tight-crop with padding.
   - Downscale to a **punchy 16-bit pixel grid** (`32×32` or `24×24`, default `32×32`) using `LANCZOS` for interior shading + alpha threshold for crisp silhouette.
   - Apply a slight vibrancy/contrast boost (`1.15×` color, `1.10×` contrast), quantize to `20` colors (`MEDIANCUT`), and wrap with a **1px warm plum-brown outline** (`#341C38`) just like classic SNES/GBA/Stardew item icons.

## Recommended Parameters

| Parameter | Default | Notes |
| :--- | :--- | :--- |
| **`--lora-weight`** | `0.0` | Disables `pixel-art-xl` during Stage 1 painting |
| **`--steps`** | `28` | Clean SDXL convergence |
| **`--cfg`** | `7.5` | Crisp cel-shading and vibrant colors |
| **`--pixel-size`** | `48` | Logical pixel grid (`24`, `32`, `48`, or `64`) |
| **`--max-colors`** | `20` | Tight retro 16-bit palette |
| **`--outline-color`** | `#341C38` | Warm dark plum-brown outline |

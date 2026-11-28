---
name: sdwebui-item-enhancer
description: >-
  Generate lush, Stardew/GBA-quality 16-bit pixel art RPG item icons AND 8-bit/16-bit 2.5D top-down
  64x64 chibi character sprites using local SD WebUI Forge (sd_xl_base_1.0) via a 2-stage
  Paint-First -> Pixel-Craft pipeline. Use this skill whenever generating 2D game item icons or
  chibi character sprites via SD WebUI, or when raw pixel-art-xl output looks crude, muddy, or
  tiled.
---

# SD WebUI Item & Character Enhancer (Paint-First → Pixel-Craft)

Workflow to turn user item or character prompts into crisp 2D RPG sprites via local SD WebUI Forge (`http://127.0.0.1:7860`) and package them with `drawtool-aseprite`.

See [SDXL Pixel Art Tuning Guide](./references/sdxl_pixel_art_guide.md) for why `Paint-First → Pixel-Craft` outperforms raw `pixel-art-xl` or crude `img2img` templates.

## Steps

1. **Treat User Prompt as Positive & Load Negative Prompt**:
   - Use the user's prompt as the **Positive Prompt**.
   - Load `c:\Users\trilt\Documents\Study\DrawTool\negative_prompt.md` automatically (`--negative-preset auto` selects `item_icon` for items or `character` for characters).

2. **Run 2-Stage Paint-First → Pixel-Craft (`scripts/sdwebui_item_gen.py`)**:
   - **Stage 1 (Paint-First)**: Calls `sd_xl_base_1.0` *without* `pixel-art-xl` (`--lora-weight 0.0` by default) to paint a lush, cel-shaded 2D JRPG illustration with clean dark plum outlines on `#D0D0D0`.
   - **Stage 2 (Pixel-Craft)**: Removes exterior `#D0D0D0` background and enclosed interior background holes cleanly, crops tightly, downscales to the target pixel grid (`48` for 16-bit items, `64` for chibi characters with bottom-anchored feet), quantizes (`20` colors for items, `32` for characters), and applies a crisp 1px warm plum-brown exterior outline (`#341C38`) on a transparent RGBA canvas.

   Example (Item `48x48`):
   ```powershell
   python .agents/skills/sdwebui-item-enhancer/scripts/sdwebui_item_gen.py `
     --mode item `
     --prompt "<user positive prompt>" `
     --name my_item `
     --pixel-size 48 `
     --candidates 3
   ```

   Example (Character 2D Pixel Top-Down 2.5D Chibi `64x64`):
   ```powershell
   python .agents/skills/sdwebui-item-enhancer/scripts/sdwebui_item_gen.py `
     --mode character `
     --facing front `
     --prompt "<user character prompt>" `
     --name my_chibi `
     --pixel-size 64 `
     --max-colors 32 `
     --candidates 3
   ```

   Outputs:
   - `output/<name>_painted.png` — winning 1024×1024 painted SDXL illustration
   - `output/<name>_raw.png` — crisp pixel-crafted sprite upscaled (`NEAREST`) to 512×512 (transparent RGBA)
   - `.cache/<name>_grid.png` — contact sheet of all painted candidates + final pixel-art result

3. **Visually Inspect the Output**:
   - Read `.cache/<name>_grid.png` and `output/<name>_raw.png` with `read_file` to verify silhouette, shading, highlights, and outline.

4. **Package to `.png` + `.aseprite` via `drawtool-aseprite`**:
   - For **items**: call `process_image_to_aseprite` with `type="item"`, `size="48x48"`, `style="16bit-topdown"`, `max_colors=24`, `downsample_mode="nearest"`, `bg_tolerance=18`, `despill=false`, `speck_filter=true`.
   - For **characters (64x64 chibi)**: call `process_image_to_aseprite` with:
     - `raw_image_path`: `"c:/Users/trilt/Documents/Study/DrawTool/output/<name>_raw.png"`
     - `type`: `"character"`
     - `size`: `"64x64"`
     - `style`: `"16bit-topdown"`
     - `max_colors`: `32`
     - `downsample_mode`: `"nearest"` (preserves exact Stage 2 pixels 1:1)
     - `bg_tolerance`: `18`
     - `despill`: `false`
     - `speck_filter`: `true`
     - `name`: `"<name>_64"`
     - `output_dir`: `"c:/Users/trilt/Documents/Study/DrawTool/output"`

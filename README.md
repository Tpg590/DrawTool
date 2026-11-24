# DrawTool — CLI & MCP Server tạo Asset Game 2D Pixel Art + Aseprite

Công cụ CLI và MCP server tự động hóa quy trình tạo asset game 2D (**`item`**, **`tile`**, **`anim`**) từ mô tả văn bản hoặc ảnh thô, hỗ trợ các preset `8bit-topdown`, `16bit-topdown`, `anime-rpg-topdown`, tối ưu token LLM và đóng gói ra file `.aseprite` (có sẵn layer `Base`, `Detail & Shading`, timeline + FPS) kèm `.png` thành phẩm.

---

## 1. Kiến trúc & Luồng xử lý

- `drawtool.py` / `src/cli.py`: Giao diện dòng lệnh (`argparse`), điều phối toàn bộ pipeline.
- `src/config.py`: Đọc `.env`, biến môi trường và file `config.json` (tùy chọn).
- `src/prompt_optimizer.py`:
  - Khóa cứng từ khóa phong cách mặc định `8bit-topdown`, hỗ trợ thêm `16bit-topdown`, `anime-rpg-topdown`, `isometric-pixel`.
  - Hỗ trợ tùy chọn `bg_color` (màu nền phẳng chống ám viền) và `action_notes` (kịch bản tư thế từng frame cho `anim`).
  - Chế độ mặc định dùng 0 token LLM (deterministic), hoặc tùy chọn gọi LLM chỉ để dịch/tối ưu mô tả chủ thể ngắn gọn.
- `src/image_gen.py`:
  - Hỗ trợ `bridge` (`gpt-image-bridge` hoặc HTTP endpoint tùy chỉnh), `openai`, `gemini`, và `sdwebui`.
  - Lưu cache ảnh gốc theo SHA-256 trong `.cache/` để tái sử dụng không tốn phí API.
- `src/pixel_processor.py`:
  - **Tách nền + Chroma Despill**: Border flood-fill kết hợp khử viền xanh/hồng bám biên (`despill`) do nén `.jpg`.
  - **Lọc nhiễu góc (Speck Filter)**: Connected-component filtering loại bỏ các đốm pixel rác ở góc cell trước khi tính khung cắt `bbox` (tránh lỗi 1 frame bị đẩy lệch và co nhỏ cả sheet).
  - **Thu nhỏ chuẩn Pixel Art (`mode` majority-vote)**: Lấy mẫu màu chủ đạo bên trong từng khối pixel thay vì `NEAREST` thuần để giữ mắt, viền đen và chi tiết nhỏ không bị rách nét.
  - **Căn chỉnh Animation (`centroid` / `center`)**: Căn ngang theo trọng tâm khối (`centroid`) và neo chân theo đáy chung (`anchor_bottom_y`) để nhân vật không bị giật ngang/dọc.
  - **Procedural Idle Generator (`create_idle_animation`)**: Tạo animation đứng nhún/thở (`bob`, `breathe`, `float`) đồng bộ 100% từ **1 ảnh nhân vật đơn**, không bao giờ tự mọc thêm chi tiết thừa.
- `src/aseprite_runner.py` & `lua/build_asset.lua`:
  - Tự dò `aseprite` (hoặc đọc `ASEPRITE_PATH`), sinh Lua script chạy `aseprite -b --script ...` để tạo `.aseprite` và xuất `.png`.

---

## 2. Cài đặt

Yêu cầu Python 3.10+ và [Aseprite](https://www.aseprite.org/) (để đóng gói `.aseprite`).

```bash
pip install -r requirements.txt
```

---

## 3. Cấu hình API & Aseprite

Sao chép file mẫu `.env.example` thành `.env` (file `.env` đã nằm trong `.gitignore`, không bị commit lên GitHub):

```powershell
Copy-Item .env.example .env
```

Chỉnh `.env` theo provider bạn dùng:

```dotenv
# Ví dụ gọi trực tiếp Gemini API (không cần localhost bridge)
IMAGE_PROVIDER=gemini
GEMINI_API_KEY=AIza...
IMAGE_MODEL=gemini-2.5-flash-image

# Hoặc chạy Local FREE bằng SD WebUI Forge (bật --api) + SDXL + LoRA Pixel Art:
# IMAGE_PROVIDER=sdwebui
# IMAGE_API_URL=http://127.0.0.1:7860
# IMAGE_MODEL=sd_xl_base_1.0.safetensors
# SDWEBUI_LORA=pixel-art-xl
# SDWEBUI_LORA_WEIGHT=0.9

# Hoặc dùng local bridge / OpenAI:
# IMAGE_PROVIDER=openai
# IMAGE_API_KEY=sk-...
# IMAGE_MODEL=gpt-image-1

# Tùy chọn LLM dịch/rút gọn prompt (đặt none để tốn 0 token LLM)
LLM_PROVIDER=none

# Đường dẫn Aseprite nếu chưa thêm vào PATH
ASEPRITE_PATH=C:\Program Files\Aseprite\Aseprite.exe
```

---

## 4. Ví dụ sử dụng CLI

### Tạo Item (`item`)
```bash
python drawtool.py --prompt "bình máu đỏ nắp gỗ" --type item --size 32x32 --output-dir ./output
```

### Tạo Tile địa hình liền mạch (`tile`)
```bash
python drawtool.py --prompt "cỏ xanh có hoa nhỏ" --type tile --size 16x16 --max-colors 16 --output-dir ./output
```

### Tạo Idle Animation mượt từ 1 ảnh nhân vật đơn (`--idle-from-single`)
```bash
python drawtool.py --prompt "female mage" --idle-from-single --idle-style bob --bob-px 1 --size 48x48 --frames 6 --cols 3 --fps 6 --style anime-rpg-topdown --raw-image ./mage_single.png --name mage_idle
```

### Tạo Animation Spritesheet (`anim`) kèm ảnh tham chiếu & palette
```bash
python drawtool.py --prompt "nhân vật hiệp sĩ đi bộ" --type anim --size 48x48 --frames 4 --cols 4 --fps 8 --style anime-rpg-topdown --downsample-mode mode --align-mode centroid --ref ./knight_ref.png --palette "#140c1c,#442434,#30346d,#d04648,#d2aa99,#6daa2c,#deeed6" --output-dir ./output
```

### Các cờ xử lý pixel nâng cao
- `--downsample-mode mode|nearest|box` (mặc định `mode`): dùng `mode` cho ảnh AI high-res, dùng `nearest` nếu ảnh gốc đã là pixel grid chuẩn.
- `--align-mode centroid|center` (mặc định `centroid`): căn ngang theo trọng tâm nhân vật.
- `--bg-tolerance 28`: tăng lên `35–42` nếu nền bị nén `.jpg` loang mạnh, giảm xuống `18–22` nếu đồ vật có màu gần giống nền.
- `--no-despill` / `--no-speck-filter`: tắt khử viền màu hoặc tắt lọc đốm nhỏ nếu cần giữ nguyên pixel phụ rời (ví dụ hạt phép bay quanh nhân vật).

---

## 5. MCP Server cho Aseprite (Antigravity / Cursor / Claude Desktop)

`DrawTool` đi kèm MCP stdio server thuần Python (`mcp_server.py`, không cần cài thêm thư viện ngoài) với **5 tools**:

1. `build_pixel_prompt`: Sinh prompt chuẩn 2D pixel art (`8bit-topdown`, `16bit-topdown`, `anime-rpg-topdown`) + kích thước ảnh gợi ý; hỗ trợ `bg_color` và `action_notes`.
2. `process_image_to_aseprite`: Nhận ảnh thô bất kỳ (từ Gemini Web, Antigravity `image-generator`, hoặc file local), tách nền + khử viền `despill`, lọc đốm rác `speck_filter`, thu nhỏ `mode`, căn chỉnh `centroid`, ghép spritesheet / nối cạnh tile, ép bảng màu và đóng gói thành `.aseprite` + `.png`.
3. `create_idle_animation`: Nhận **1 ảnh nhân vật đơn**, xử lý sạch rồi tạo luôn spritesheet Idle `4` hoặc `6` frame (`bob` / `breathe` / `float`) đồng bộ 100% không mọc thêm chi tiết lạ, xuất thẳng `.aseprite` + `.png`.
4. `generate_game_asset`: Chạy trọn quy trình từ prompt → gọi `IMAGE_PROVIDER` → xử lý pixel → xuất `.aseprite` + `.png`.
5. `check_aseprite_status`: Kiểm tra đường dẫn `aseprite` và trạng thái cấu hình provider.

Cấu hình trong `mcp_config.json` / `settings.json`:

```json
{
  "mcpServers": {
    "drawtool-aseprite": {
      "command": "python",
      "args": ["C:/Users/trilt/Documents/Study/DrawTool/mcp_server.py"]
    }
  }
}
```

> **Lưu ý:** Sau khi cập nhật code `mcp_server.py`, hãy **Reload / Restart MCP Server** trong IDE để IDE nạp lại danh sách tool mới (bao gồm `create_idle_animation` và các tham số mới).

---

## 6. Kiểm thử

```bash
python -m unittest discover -s tests -v
```

---

## 7. Thư viện Prompt mẫu chuẩn (Item, Tilemap, Anim)

### Bảng chọn nhanh thông số trước khi Prompt

| Loại Asset | Size khuyến nghị | Style / `max_colors` | Màu nền tối ưu (`bg_color`) | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| **Item đơn giản** (thuốc, xu, chìa) | `16x16` hoặc `32x32` | `8bit-topdown` (`16`) | `#D0D0D0` (xám nhạt) hoặc `#FF00FF` | Đồ thủy tinh/trong suốt **bắt buộc ưu tiên `#D0D0D0`** để không bị ám hồng/xanh |
| **Item chi tiết** (kiếm, sách phép, giáp) | `32x32` hoặc `48x48` | `16bit-topdown` (`24–32`) | `#D0D0D0` (xám nhạt) | Viền tối 1px tách bạch nền |
| **Tilemap** (cỏ, đá, nước, sàn) | `16x16` hoặc `32x32` | `8bit-topdown` (`16`) hoặc `16bit-topdown` (`24`) | **Không có nền** (phủ kín 100% khung) | Dùng chung 1 `--palette` cho các tile cùng khu vực |
| **Character Idle (cơ bản)** | `48x48` hoặc `64x64` | `anime-rpg-topdown` (`24–32`) | `#D0D0D0` hoặc `#00FF00` | Vẽ **1 sprite đơn** rồi chạy `create_idle_animation` |
| **Character Action (`walk`/`run`/`attack`)** | `48x48` hoặc `64x64` | `anime-rpg-topdown` (`24–32`) | `#D0D0D0` hoặc `#00FF00` | Dùng **4 frame** + ảnh tham chiếu + `action_notes` từng frame |

> **Mẹo chọn size cho nhân vật RPG anime:** `32x32` chỉ phù hợp nhân vật chibi cực tối giản (~15px bề ngang). Muốn nhìn rõ tóc mái, mắt anime, cổ áo choàng và vũ khí hãy dùng **`48x48`** (chuẩn retro hiện đại) hoặc **`64x64`**.

---

### 7.1. Prompt mẫu cho `Item`

#### A. Vật phẩm tiêu hao / Thủy tinh (Bình máu, Lọ Mana)
```text
red health potion in a round glass bottle with a brown wooden cork stopper, bright crimson liquid inside with a tiny white specular highlight, crisp 1px dark outline, clean flat pixel shading, single isolated game item centered in frame, sized clearly for a 32x32 sprite, top-down 2D RPG icon view, solid uniform flat neutral light gray (#D0D0D0) background, no anti-aliasing, no text
```
*(Đổi màu nhanh: thay `red health potion ... crimson liquid` thành `blue mana potion ... glowing cyan liquid` hoặc `antidote potion ... emerald green liquid`.)*

#### B. Vũ khí & Trang bị (Kiếm, Gậy phép, Khiên)
```text
medieval silver longsword with a gold crossguard and ocean-blue leather grip, slight 45-degree diagonal icon angle, crisp 1px dark outline, strong metallic contrast highlights, single isolated game item centered cleanly in frame, sized for a 32x32 sprite, 16-bit pixel art RPG inventory icon, solid uniform flat neutral light gray (#D0D0D0) background, no text
```
```text
ornate wooden mage staff topped with a glowing cyan crystal and gold prongs, crisp 1px dark outline, clean cel shading, single isolated game item centered cleanly in frame, sized for a 32x32 sprite, 16-bit pixel art RPG inventory icon, solid uniform flat neutral light gray (#D0D0D0) background, no text
```

#### C. Rương báu & Prop tương tác
```text
closed medieval wooden treasure chest reinforced with iron bands and a gold lock, top-down 3/4 RPG game perspective, crisp 1px dark outline, distinct wood grain pixel clusters, single isolated game prop centered in frame, sized for a 32x32 sprite, 8-bit pixel art, solid uniform flat neutral light gray (#D0D0D0) background, no text
```

---

### 7.2. Prompt mẫu cho `Tilemap` (`tile`)

Nguyên tắc vàng của `tile`: **phủ kín 100% cạnh-đến-cạnh**, chi tiết phân bố đều, **cấm vẽ vật thể lớn nằm cắt ngang mép ảnh**.

#### A. Thảm cỏ tự nhiên có hoa nhỏ
```text
seamless tileable top-down lush green grass terrain texture with tiny scattered clover clusters and a few tiny white-yellow wildflowers, uniform micro-pixel texture, no large objects near borders so edges wrap seamlessly, filling the entire square edge-to-edge for a 16x16 or 32x32 RPG tile, 8-bit pixel art, top-down game view, limited green palette, crisp pixel edges, no grid lines
```

#### B. Đường đất / Đá cuội làng quê
```text
seamless tileable top-down warm dirt path terrain with subtle small pebbles and fine pixel grain, low-contrast balanced pattern with no big rocks on edges, filling the entire square edge-to-edge for a 32x32 RPG tile, 16-bit pixel art, top-down game view, crisp pixel edges
```

#### C. Đá lát Hầm ngục (Dungeon Stone Floor)
```text
seamless tileable top-down dark slate dungeon stone floor tiles with subtle moss in thin mortar cracks, uniform grid alignment that repeats cleanly across all 4 edges, filling the entire square edge-to-edge for a 32x32 RPG tile, 16-bit pixel art, top-down game view, limited cool gray-blue palette, crisp pixel edges
```

#### D. Mặt nước (Water Tile — kết hợp 2–3 frame hoặc 1 tile tĩnh)
```text
seamless tileable top-down clear ocean-blue water surface texture with subtle repeating pale-cyan pixel ripple highlights, uniform pattern across all borders, filling the entire square edge-to-edge for a 32x32 RPG tile, 16-bit pixel art, top-down game view, crisp pixel edges
```

---

### 7.3. Prompt mẫu cho `Anim` (Đầy đủ biến thể từng loại hành động)

> **Quy tắc bất biến khi tạo Anim:**
> - Với **Idle đứng nhún/thở cơ bản**: **KHÔNG** kêu AI tự vẽ 6 frame cùng lúc. Hãy dùng prompt **Single Sprite** bên dưới để vẽ 1 nhân vật sắc nét nhất, sau đó gọi MCP `create_idle_animation` (hoặc `--idle-from-single`).
> - Với **Idle có hành động / Walk / Run / Attack / Cast**: dùng **4 frame** (hoặc **6 frame** cho Cast/Death), viết rõ **`Frame 1..4`** trong `action_notes` để AI không tự bịa thêm chi tiết.

---

#### A. Nhóm `IDLE` (Đứng chờ)

##### 1) Idle cơ bản (Đứng nhún / Thở / Lơ lửng) — *Dùng với `create_idle_animation`*
Prompt tạo 1 sprite gốc cực nét (`type="item"` hoặc gọi `image-generator` 1 ảnh vuông `1024x1024`, sau đó đưa vào `create_idle_animation` với `size="48x48"`, `frames=6`, `cols=3`, `idle_style="bob"` hoặc `"breathe"` hoặc `"float"`):
```text
single isolated medieval RPG anime female mage character facing forward in top-down 3/4 view, 2.5-head chibi JRPG proportion, silver-white bob hair with clear bangs, distinct expressive blue anime eyes, open dark-navy hood resting on shoulders (face fully visible), ocean-blue cloak and tunic with gold trim and brown belt, holding a wooden staff with glowing cyan gem in right hand, feet planted firmly shoulder-width apart, crisp 1px dark outline, clean cel-shaded pixel blocks, sized clearly for a 48x48 sprite, solid uniform flat neutral light gray (#D0D0D0) background, 16-bit anime RPG pixel art, no text
```

##### 2) Idle có hành động (Action Idle — Nâng gậy phát sáng / Lật sách phép / Kiểm tra kiếm)
Dùng 4 frame (`cols=4`, `frames=4`, `fps=5`), đính kèm ảnh nhân vật gốc làm tham chiếu:
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, same female mage character from reference image on solid flat #D0D0D0 background, no grid lines.
Frame 1: neutral standing pose holding staff calmly.
Frame 2: slightly raises right hand with staff, cyan crystal begins glowing softly, cloak sways gently.
Frame 3: crystal emits bright sparkle pixel aura, hair bangs lift slightly, gentle smile.
Frame 4: lowers staff smoothly back toward neutral pose.
Feet remain planted at the exact same baseline across all 4 frames, consistent hair, outfit, and 1px dark outline, 16-bit anime RPG pixel art for 48x48 frames.
```

##### 3) Idle tư thế chiến đấu (Combat Stance Idle)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, same character from reference image in a low-center combat ready stance facing forward-right, knees slightly bent, weapon raised defensively, solid flat #D0D0D0 background.
Frame 1: alert combat guard stance.
Frame 2: upper body dips 1px down with weapon arm tensing.
Frame 3: deepest guard crouch with cloak flaring slightly.
Frame 4: rebounds smoothly to alert combat stance.
Keep exact same weapon, outfit, and grounded feet positions across all 4 frames, 16-bit anime RPG pixel art for 48x48 frames.
```

##### 4) Idle mệt mỏi / thương tích (Exhausted / Low-HP Idle)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, same character from reference image hunched slightly forward catching breath, one hand resting on knee or leaning on staff, solid flat #D0D0D0 background.
Frame 1: hunched tired posture, shoulders raised from heavy breath.
Frame 2: chest drops down heavily as exhaling, head dips slightly.
Frame 3: lowest tired slump, arms hanging heavy.
Frame 4: slowly lifts shoulders back up to inhale.
Consistent outfit and colors, 16-bit anime RPG pixel art for 48x48 frames.
```

---

#### B. Nhóm `WALK` (Đi bộ — 4 frame chuẩn chu kỳ bước)

##### 1) Walk RPG bình thường (Normal Top-Down Walk)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, same character from reference image walking downward (front view) in classic top-down JRPG style, solid flat #D0D0D0 background, no grid lines.
Frame 1: left foot stepping forward, right arm swinging slightly forward, 1px upward stride bounce.
Frame 2: both feet passing center (contact/passing pose), neutral height.
Frame 3: right foot stepping forward, left arm swinging slightly forward, 1px upward stride bounce.
Frame 4: both feet passing center, neutral height.
Keep exact same head, hair, outfit, staff, and color palette in every frame, 16-bit anime RPG pixel art for 48x48 frames.
```

##### 2) Walk kiểu kinh dị / xác sống / ma ám (Horror / Creepy / Shambling Limp Walk)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, creepy horror-style shambling walk of the character from reference image, head tilted awkwardly to one side, arms hanging limp or twitching forward, asymmetrical dragging limp gait, solid flat #D0D0D0 background.
Frame 1: drags right leg forward with torso leaning sharply left, head tilted.
Frame 2: heavy uneven slump downward as weight drops onto front foot.
Frame 3: drags stiff back leg forward while arms twitch slightly upward.
Frame 4: awkward jerky recoil back to hunched shambling stance.
Consistent silhouette and colors, eerie 16-bit RPG pixel art for 48x48 frames.
```

##### 3) Walk lén lút / rón rén (Stealth / Tiptoe Sneak)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, same character crouching low in a sneaky tiptoe walk, cloak pulled close, cautious lowered center of gravity, solid flat #D0D0D0 background.
Frame 1: crouched stealth pose, left tiptoe reaching quietly forward.
Frame 2: weight shifts low onto left foot, right knee tucked behind.
Frame 3: crouched stealth pose, right tiptoe reaching quietly forward.
Frame 4: weight shifts low onto right foot, left knee tucked behind.
Consistent outfit and colors, 16-bit anime RPG pixel art for 48x48 frames.
```

##### 4) Walk oai vệ / quý tộc / hành quân (Heroic / Confident March)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, same character marching forward with proud upright posture, chest out, rhythmic strong arm swing, crisp billowing cloak hem, solid flat #D0D0D0 background.
Frame 1: left leg crisp stride forward, chest proud, cloak flaring slightly right.
Frame 2: upright passing pose with crisp posture.
Frame 3: right leg crisp stride forward, cloak flaring slightly left.
Frame 4: upright passing pose returning cleanly to loop.
16-bit anime RPG pixel art for 48x48 frames.
```

##### 5) Walk nặng nề giáp sắt (Heavy Armored Stomp)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, heavy slow armored march of character from reference image, wide sturdy stance, heavy shoulder sway, solid flat #D0D0D0 background.
Frame 1: lifts left boot high with torso leaning slightly right.
Frame 2: heavy stomp down on left boot, entire body drops 2px with impact weight.
Frame 3: lifts right boot high with torso leaning slightly left.
Frame 4: heavy stomp down on right boot, entire body drops 2px with impact weight.
16-bit RPG pixel art for 48x48 frames.
```

---

#### C. Nhóm `RUN` (Chạy — 4 đến 6 frame tốc độ cao)

##### 1) Run tiêu chuẩn RPG (Standard Sprint — 4 frame)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, same character sprinting energetically downward/forward, strong arm pump, hair and cloak trailing upward/backward from wind, solid flat #D0D0D0 background.
Frame 1: left knee driven forward high, right leg extended back, cloak streaming behind.
Frame 2: airborne mid-stride stretch, body 2px higher, hair fluttering back.
Frame 3: right knee driven forward high, left leg extended back, cloak streaming behind.
Frame 4: airborne mid-stride stretch, body 2px higher, hair fluttering back.
Consistent outfit and weapon, 16-bit anime RPG pixel art for 48x48 frames.
```

##### 2) Run kiểu Anime / Ninja nghiêng người (Anime Lean-Forward Dash — 4 frame)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, dynamic anime-style fast dash, torso leaning steeply forward, one or both arms swept backward behind torso, long cloak and hair flying horizontally behind, solid flat #D0D0D0 background.
Frame 1: deep forward lean, left leg kicking forward, arms swept back.
Frame 2: fast low-glide extension, cloak billowing sharply backward.
Frame 3: deep forward lean, right leg kicking forward, arms swept back.
Frame 4: fast low-glide extension returning cleanly to loop.
16-bit anime RPG pixel art for 48x48 frames.
```

##### 3) Run hoảng loạn / chạy trốn (Frantic / Panic Run — 4 frame)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, frantic panicked run of character from reference image, arms flailing slightly up near chest, wide-eyed worried expression, rapid short high-knee steps, solid flat #D0D0D0 background.
Frame 1: wide panicked stride left, arms raised near shoulders, hair bouncing wildly.
Frame 2: quick bounce mid-step, cloak flapping unevenly.
Frame 3: wide panicked stride right, arms raised near shoulders.
Frame 4: quick bounce mid-step looping smoothly.
16-bit anime RPG pixel art for 48x48 frames.
```

---

#### D. Nhóm `ATTACK` / `CAST` / `HURT` / `DEATH`

##### 1) Attack — Vung kiếm / Đánh cận chiến (4 frame)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, melee slash attack of character from reference image, solid flat #D0D0D0 background.
Frame 1: wind-up anticipation, pulls weapon arm back, torso twisted slightly.
Frame 2: fast forward lunge with sweeping bright pixel slash arc in front.
Frame 3: follow-through stance, weapon extended on opposite side, fading slash trail.
Frame 4: recovers back toward ready stance.
Consistent character design, 16-bit anime RPG pixel art for 48x48 frames.
```

##### 2) Spell Cast — Niệm phép (4 frame)
```text
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, spellcasting animation of the female mage from reference image, solid flat #D0D0D0 background.
Frame 1: raises wooden staff upward with both hands, cyan crystal starts glowing.
Frame 2: gathers swirling cyan magic orb above staff tip, cloak and white hair lift upward from magic wind.
Frame 3: thrusts staff forward releasing bright cyan energy burst.
Frame 4: lowers staff calmly back to neutral stance as glow fades.
Consistent outfit and face, 16-bit anime RPG pixel art for 48x48 frames.
```

##### 3) Hurt (2–4 frame) & Death (4 frame)
```text
# HURT (2 frames, cols=2):
2D sprite sheet arranged in an exact 2 columns by 1 row uniform grid, hit-reaction stagger of character from reference image, solid flat #D0D0D0 background.
Frame 1: sharp recoil backward, head snapped back, arms flung slightly open.
Frame 2: recovering crouch holding torso before returning to idle.

# DEATH (4 frames, cols=4):
2D sprite sheet arranged in an exact 4 columns by 1 row uniform grid, defeat animation of character from reference image, solid flat #D0D0D0 background.
Frame 1: staggers backward dropping staff/weapon slightly.
Frame 2: collapses down onto both knees, head bowed, cloak pooling around.
Frame 3: falls forward/sideways onto the ground.
Frame 4: lying flat unconscious on ground, cloak spread neatly around.
```

---

## 8. Quy trình làm trọn bộ Animation hoàn chỉnh & đồng bộ từ 1 Base/Idle Sprite

Để làm một bộ đầy đủ (`Idle` + `Walk` + `Run` + `Attack/Cast` + `Hurt` + `Death`) cho 1 nhân vật mà **không bị biến dạng mặt, đổi màu áo hay lệch kích thước** giữa các state:

1. **Bước 1 — Tạo "Master Base Sprite" (1 ảnh đơn cực nét)**:
   - Chọn cố định kích thước mục tiêu (khuyên dùng **`48x48`** hoặc **`64x64`**) và phong cách (`anime-rpg-topdown`).
   - Sinh 1 ảnh nhân vật đơn đứng thẳng ở tư thế chuẩn trên nền phẳng `#D0D0D0` (hoặc `#00FF00`), xử lý thành `master_base.png` và **giữ lại cả ảnh gốc độ phân giải cao** (`master_base_raw.png`).
2. **Bước 2 — Khóa bảng màu (Palette Lock)**:
   - Trích xuất các mã màu chính của `master_base.png` (hoặc tạo 1 file `character.gpl` / chuỗi `--palette "#140c1c,#243a6e,..."`).
   - Từ giờ mọi state đều truyền chung `--palette` này (hoặc cùng `max_colors`) để da, tóc, áo không bao giờ bị đổi tone giữa `Idle` và `Walk`.
3. **Bước 3 — Sinh `Idle` từ Master Sprite (0% lỗi)**:
   - Chạy `create_idle_animation` (hoặc `--idle-from-single`) trực tiếp từ `master_base_raw.png` → ra ngay `char_idle.aseprite` & `char_idle.png` (6 frame, 6 FPS) mượt và đồng bộ tuyệt đối.
4. **Bước 4 — Sinh các state động (`Walk`, `Run`, `Cast`, `Attack`, `Hurt`, `Death`)**:
   - **Luôn truyền `master_base_raw.png` làm ảnh tham chiếu** (`ref_path` / `--ref` hoặc đính kèm cho `image-generator`).
   - Giữ nguyên y hệt phần mô tả ngoại hình của Master Sprite, **chỉ thay phần `Frame 1..4`** theo mẫu ở Mục 7.3.
   - Đảm bảo mọi lệnh `process_image_to_aseprite` dùng chung:
     - `size="48x48"` (bằng đúng size của Idle)
     - `downsample_mode="mode"`
     - `align_mode="centroid"`
     - `despill=True`, `speck_filter=True`
5. **Bước 5 — Kiểm tra & Ghép vào Aseprite / Game Engine**:
   - Mở các file `.aseprite` bằng Aseprite, kiểm tra bằng `Onion Skin` (phím tắt `F3` hoặc bật icon Onion Skin) để chắc chắn **vị trí bàn chân (baseline) giữa `Idle` và `Walk` khớp cùng 1 hàng pixel** → khi nhân vật chuyển từ đứng sang chạy trong Unity/Godot sẽ không bị giật lên xuống.
   - Có thể gộp chung các animation vào 1 file `.aseprite` duy nhất và dùng **Frame Tags** (`Idle`, `Walk`, `Run`, `Cast`, `Hurt`, `Death`) để export thẳng sang engine.

---

## 9. Cách sửa Prompt để đổi Trang phục, Tóc, Vũ khí & Ngoại hình nhân vật

### 9.1. Công thức Modular 6 Slot ("Character DNA")

Đừng viết lại prompt từ đầu mỗi lần tạo nhân vật mới. Hãy chia prompt thành **6 mảnh ghép cố định**, chỉ sửa đúng mảnh bạn muốn đổi:

```text
[Slot 1: Style & Tỷ lệ] + [Slot 2: Tóc & Mặt] + [Slot 3: Mũ/Phụ kiện] + [Slot 4: Trang phục/Áo choàng] + [Slot 5: Vũ khí/Tay] + [Slot 6: Pose/Nền]
```

| Slot | Vai trò | Ví dụ cố định (Khóa phong cách game) |
| :--- | :--- | :--- |
| **Slot 1** | Khóa nét vẽ & tỷ lệ | `single isolated medieval RPG anime character facing forward in top-down 3/4 view, 2.5-head chibi JRPG proportion, crisp 1px dark outline, clean cel-shaded pixel blocks, sized clearly for a 48x48 sprite, 16-bit anime RPG pixel art` |
| **Slot 2** | Tóc, màu mắt, biểu cảm | `silver-white bob hair with clear bangs, distinct expressive blue anime eyes` |
| **Slot 3** | Mũ / Khăn / Phụ kiện đầu | `open dark-navy hood resting on shoulders (face fully visible)` |
| **Slot 4** | Áo giáp / Áo choàng / Giày | `ocean-blue cloak and tunic with gold trim and brown leather belt, dark boots` |
| **Slot 5** | Vũ khí / Vật cầm tay | `holding a wooden staff with glowing cyan crystal in right hand` |
| **Slot 6** | Tư thế & Nền | `feet planted shoulder-width apart, solid uniform flat #D0D0D0 background, no text` |

### 9.2. Ví dụ thực chiến: Biến đổi từ Pháp sư nữ tóc trắng sang các nhân vật khác

Giữ nguyên **Slot 1** và **Slot 6**, chỉ thay **Slot 2–5**:

- **Phiên bản 1 — Pháp sư Băng (Gốc):**
  - *Slot 2:* `silver-white bob hair with bangs, blue anime eyes`
  - *Slot 3:* `open dark-navy hood resting on shoulders`
  - *Slot 4:* `ocean-blue mage robe and cloak with gold trim`
  - *Slot 5:* `wooden staff with glowing cyan crystal`

- **Phiên bản 2 — Đổi thành Pháp sư Mùa Đông / Hoàng gia (Giữ người, đổi đồ & tóc):**
  - *Slot 2:* `long platinum-blonde wavy hair with bangs, emerald green anime eyes`
  - *Slot 3:* `small gold circlet crown on forehead` *(bỏ mũ trùm)*
  - *Slot 4:* `crimson-red velvet dress with white fur collar and gold belt`
  - *Slot 5:* `ornate golden scepter with ruby gem`

- **Phiên bản 3 — Đổi thành Hiệp sĩ / Sát thủ cùng bộ game:**
  - *Slot 2:* `short spiky charcoal-black hair, sharp amber anime eyes`
  - *Slot 3:* `dark-gray scarf wrapped around neck`
  - *Slot 4:* `dark-purple leather stealth tunic, silver shoulder pauldron, strapped boots`
  - *Slot 5:* `holding twin silver daggers angled downward`

### 9.3. Cách đổi ngoại hình cho một bộ Animation đã có sẵn (2 phương pháp)

1. **Phương pháp 1 — Reskin giữ nguyên chuyển động (Reference Pose-Lock):**
   - Đưa ảnh spritesheet `walk` hoặc `idle` cũ làm ảnh tham chiếu (`ref_path`).
   - Dùng mẫu prompt:
     > `Keep the exact same 4-column x 1-row grid layout, character proportions, and exact frame-by-frame pose positions from the reference sprite sheet, ONLY redesign the character's hair and outfit to: [điền Slot 2-5 mới], solid uniform flat #D0D0D0 background, 16-bit anime RPG pixel art for 48x48 frames.`
   - Sau đó chạy lại `process_image_to_aseprite` với cùng `size`, `frames`, `cols`.

2. **Phương pháp 2 — Đổi màu tức thì không cần vẽ lại bằng AI (Palette Swap):**
   - Nếu chỉ muốn đổi phe/hệ nguyên tố (ví dụ: Pháp sư Xanh Biển → Pháp sư Lửa Đỏ Vàng hoặc Pháp sư Hắc Ám Tím Đen), **không cần gọi lại AI tạo ảnh**!
   - Chỉ cần chạy lại `process_image_to_aseprite` (hoặc CLI `--raw-image`) trên ảnh cũ và truyền `--palette` mới (hoặc chỉnh Hue/Palette trực tiếp trong Aseprite bằng `Sprite > Color Mode > Indexed` rồi thay màu trên bảng Palette bên trái).

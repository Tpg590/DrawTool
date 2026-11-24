# DrawTool — CLI tạo Asset Game 2D 8-bit Pixel Art + Aseprite

Công cụ CLI tự động hóa quy trình tạo asset game 2D (**`item`**, **`tile`**, **`anim`**) từ mô tả văn bản, khóa cứng mặc định phong cách **2D 8-bit pixel art, top-down view**, tối ưu token LLM và đóng gói ra file `.aseprite` (có sẵn layer `Base`, `Detail & Shading`, timeline + FPS) kèm `.png` thành phẩm.

## 1. Kiến trúc & Luồng xử lý

- `drawtool.py` / `src/cli.py`: Giao diện dòng lệnh (`argparse`), điều phối toàn bộ pipeline.
- `src/config.py`: Đọc `.env`, biến môi trường và file `config.json` (tùy chọn).
- `src/prompt_optimizer.py`:
  - Khóa cứng từ khóa phong cách mặc định `8bit-topdown`.
  - Chế độ mặc định dùng 0 token LLM (deterministic), hoặc tùy chọn gọi LLM chỉ để dịch/tối ưu mô tả chủ thể ngắn gọn trước khi ghép với bộ quy tắc cứng.
- `src/image_gen.py`:
  - Hỗ trợ `bridge` (`gpt-image-bridge` hoặc HTTP endpoint tùy chỉnh), `openai`, và `gemini`.
  - Lưu cache ảnh gốc theo SHA-256 trong `.cache/` để tái sử dụng không tốn phí API.
- `src/pixel_processor.py`:
  - Xử lý ảnh deterministic bằng `Pillow` + `numpy`: tách nền border flood-fill, crop sát nội dung, resize `NEAREST`, chia/ghép spritesheet `anim` giữ tỉ lệ đồng nhất giữa các frame, xử lý nối cạnh `tile`, và ép màu `--palette` / `--max-colors`.
- `src/aseprite_runner.py` & `lua/build_asset.lua`:
  - Tự dò `aseprite` (hoặc đọc `ASEPRITE_PATH`), sinh Lua script chạy `aseprite -b --script ...` để tạo `.aseprite` và xuất `.png`.

## 2. Cài đặt

Yêu cầu Python 3.10+ và [Aseprite](https://www.aseprite.org/) (để đóng gói `.aseprite`).

```bash
pip install -r requirements.txt
```

## 3. Cấu hình API & Aseprite

Sao chép file mẫu `.env.example` thành `.env` (file `.env` đã nằm trong `.gitignore`, không bị commit lên GitHub):

```powershell
Copy-Item .env.example .env
```

Chỉnh `.env` theo provider bạn dùng:

```dotenv
# Ví dụ chạy Local FREE bằng SD WebUI Forge (bật --api) + SDXL + LoRA Pixel Art:
IMAGE_PROVIDER=sdwebui
IMAGE_API_URL=http://127.0.0.1:7860
IMAGE_MODEL=sd_xl_base_1.0.safetensors
SDWEBUI_LORA=pixel-art-xl
SDWEBUI_LORA_WEIGHT=0.9

# Hoặc dùng local bridge:
# IMAGE_PROVIDER=bridge
# IMAGE_API_URL=http://127.0.0.1:8080/v1/images/generations
# IMAGE_API_KEY=
# IMAGE_MODEL=gpt-image-1

# Ví dụ gọi trực tiếp Gemini API (không cần localhost bridge)
# IMAGE_PROVIDER=gemini
# GEMINI_API_KEY=AIza...
# IMAGE_MODEL=gemini-2.5-flash-image

# Hoặc dùng OpenAI:
# IMAGE_PROVIDER=openai
# IMAGE_API_KEY=sk-...
# IMAGE_MODEL=gpt-image-1

# Tùy chọn LLM dịch/rút gọn prompt (đặt none để tốn 0 token LLM)
LLM_PROVIDER=none

# Đường dẫn Aseprite nếu chưa thêm vào PATH
ASEPRITE_PATH=C:\Program Files\Aseprite\Aseprite.exe
```

## 4. Ví dụ sử dụng

### Tạo Item (`item`)
```bash
python drawtool.py --prompt "bình máu đỏ nắp gỗ" --type item --size 32x32 --output-dir ./output
```

### Tạo Tile địa hình liền mạch (`tile`)
```bash
python drawtool.py --prompt "cỏ xanh có hoa nhỏ" --type tile --size 16x16 --max-colors 16 --output-dir ./output
```

### Tạo Animation Spritesheet (`anim`) kèm ảnh tham chiếu & palette
```bash
python drawtool.py --prompt "nhân vật hiệp sĩ đi bộ" --type anim --size 32x32 --frames 4 --cols 4 --fps 8 --ref ./knight_ref.png --palette "#140c1c,#442434,#30346d,#d04648,#d2aa99,#6daa2c,#deeed6" --output-dir ./output
```

### Chạy lại bước xử lý pixel & đóng gói từ ảnh có sẵn (không gọi API ảnh)
```bash
python drawtool.py --prompt "bình máu đỏ" --type item --size 32x32 --raw-image ./sample_raw.png --skip-aseprite
```

## 5. MCP Server cho Aseprite (Antigravity / Cursor / Claude Desktop)

`DrawTool` đi kèm MCP stdio server thuần Python (`mcp_server.py`, không cần cài thêm thư viện ngoài) với 4 tools:
- `build_pixel_prompt`: Sinh prompt chuẩn 2D 8-bit top-down + kích thước lưới `anim`.
- `process_image_to_aseprite`: Nhận ảnh thô bất kỳ (từ Gemini Web, Antigravity `image-generator`, hoặc file local), tách nền, resize `NEAREST`, ghép spritesheet / nối cạnh tile, ép bảng màu và đóng gói thành `.aseprite` + `.png`.
- `generate_game_asset`: Chạy trọn quy trình từ prompt → gọi `IMAGE_PROVIDER` → xử lý pixel → xuất `.aseprite` + `.png`.
- `check_aseprite_status`: Kiểm tra đường dẫn `aseprite` và trạng thái cấu hình provider.

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

## 6. Kiểm thử

```bash
python -m unittest discover -s tests -v
```

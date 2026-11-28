# Negative Prompt Library (SD WebUI -> Pixel Pipeline)

> Quy ước dùng cho script/tool tự động:
> - Luôn lấy `base` làm nền tảng.
> - Chọn thêm 1 hoặc nhiều khối trong `categories` tùy theo loại asset đang sinh.
> - Lưu ý: Không bật `liquid_raw` nếu đang vẽ `potion_jar` (và ngược lại).

## 1. Base (Dùng chung cho mọi thể loại)
photorealistic, realistic, 3d render, unreal engine, ray tracing, smooth shading, blurry, out of focus, soft focus, anti-aliased, heavy bloom, strong glow, lens flare, aura, noisy, messy pixels, deformed, distorted, disfigured, bad anatomy, extra limbs, missing limbs, ugly, watermark, signature, text, letters, numbers, logo, UI box, frame, border, vignette, cropped, out of frame

## 2. Background & Cutout Clean (Chống bóng đổ, nền nhiễu để dễ cắt hình & pixel hóa)
drop shadow, cast shadow, ground shadow, floating shadow, shadow, reflection, floor, ground, surface, pedestal, stand, scenery, environment, background details, gradient background, textured background, patterned background

## 3. Categories (Ghép thêm theo từng thể loại)

### `item_general` (Vũ khí, trang bị, khoáng sản, đồ ăn, vật phẩm lẻ)
multiple items, duplicates, pile, stack, set, grid, collage, character, person, hands, holding, portrait, bust, full body, animal head, creature

### `monster_material` (Thịt quái thú, da, vảy, răng, xương, nguyên liệu quái vật)
multiple items, duplicates, pile, stack, character, person, hands, holding, head, face, animal head, monster head, creature, monster body, horns, snout, ears, eyes, skulls, portrait, bust, full body

### `liquid_raw` (Mật ong thô, dịch thể, chất lỏng, bột không đựng trong chai/lọ)
puddle spread wide, spilled liquid, potion bottle, glass jar, flask, vial, bottle, cup, bowl, container, multiple items, duplicates

### `potion_jar` (Bình máu, lọ độc, thuốc hồi phục, lọ thủy tinh)
puddle spread wide, spilled liquid, multiple items, duplicates, pile, stack, character, person, hands, holding

### `character_sprite` (Nhân vật game, quái vật hoàn chỉnh, sprite, portrait)
multiple characters, crowd, extra limbs, extra fingers, mutated hands, bad hands, fused fingers, cross-eyed, asymmetric eyes, floating body parts, cut off, cropped head, UI box, frame, border, background clutter, complex background, ground shadow, shadow under feet, painterly, semi-realistic, tall realistic proportions

## 4. Presets (Gợi ý cấu hình ghép sẵn cho script)

| Tên preset | Các khối ghép | Ghi chú |
| :--- | :--- | :--- |
| `item_icon` | `base` + `cutout_clean` + `item_general` | Dùng cho vật phẩm, vũ khí, nguyên liệu thường |
| `beast_meat` | `base` + `cutout_clean` + `monster_material` + `liquid_raw` | Dùng cho thịt/nguyên liệu quái vật không chứa trong bình |
| `potion` | `base` + `cutout_clean` + `potion_jar` | Dùng cho bình thuốc, lọ chứa chất lỏng |
| `character` | `base` + `cutout_clean` + `character_sprite` | Dùng cho nhân vật, quái vật toàn thân hoặc chân dung |
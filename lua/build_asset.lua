-- Parameterized Aseprite CLI template script (mirrors src/aseprite_runner.py)
-- Usage example:
--   aseprite -b --script-param input=processed.png --script-param out_aseprite=out.aseprite \
--     --script-param out_png=out.png --script-param type=anim --script-param width=32 \
--     --script-param height=32 --script-param frames=4 --script-param cols=4 --script-param fps=8 \
--     --script lua/build_asset.lua

local INPUT_PNG = app.params["input"] or error("Missing script-param: input")
local OUT_ASEPRITE = app.params["out_aseprite"] or error("Missing script-param: out_aseprite")
local OUT_PNG = app.params["out_png"] or error("Missing script-param: out_png")
local ASSET_TYPE = app.params["type"] or "item"
local FRAME_W = tonumber(app.params["width"]) or 32
local FRAME_H = tonumber(app.params["height"]) or 32
local FRAMES = tonumber(app.params["frames"]) or 1
local COLS = tonumber(app.params["cols"]) or FRAMES
local FPS = tonumber(app.params["fps"]) or 8

local ROWS = 1
if ASSET_TYPE == "anim" then
  ROWS = math.ceil(FRAMES / COLS)
end
local FRAME_DURATION = 1.0 / FPS

local srcImage = Image{ fromFile = INPUT_PNG }
if not srcImage then
  error("Failed to load input image: " .. INPUT_PNG)
end

local sprite = Sprite(FRAME_W, FRAME_H, ColorMode.RGB)
local baseLayer = sprite.layers[1]
baseLayer.name = "Base"

local detailLayer = sprite:newLayer()
detailLayer.name = "Detail & Shading"

if ASSET_TYPE == "anim" then
  for i = 2, FRAMES do
    sprite:newEmptyFrame()
  end

  for i = 1, FRAMES do
    sprite.frames[i].duration = FRAME_DURATION
    local idx = i - 1
    local col = idx % COLS
    local row = math.floor(idx / COLS)
    local frameImg = Image(FRAME_W, FRAME_H, ColorMode.RGB)
    frameImg:drawImage(srcImage, Point(-col * FRAME_W, -row * FRAME_H))
    sprite:newCel(baseLayer, i, frameImg, Point(0, 0))
  end

  sprite:saveAs(OUT_ASEPRITE)
  app.command.ExportSpriteSheet{
    ui = false,
    askOverwrite = false,
    type = SpriteSheetType.ROWS,
    columns = COLS,
    rows = ROWS,
    textureFilename = OUT_PNG,
    splitLayers = false,
    listLayers = false,
    listTags = false,
    listSlices = false,
    ignoreEmpty = false,
    mergeDuplicates = false,
    trim = false,
    openGenerated = false
  }
else
  sprite.frames[1].duration = FRAME_DURATION
  sprite:newCel(baseLayer, 1, srcImage, Point(0, 0))
  sprite:saveAs(OUT_ASEPRITE)
  sprite:saveCopyAs(OUT_PNG)
end

sprite:close()

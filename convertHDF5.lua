require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'hdf5'
require 'xlua'
require 'src/utils'
require 'loadcaffe'

---------------------------------------------------------
-- Define params
---------------------------------------------------------
local cmd = torch.CmdLine()

cmd:option('-images_path', 'data/contents/JPEGImages')
cmd:option('-save_to', 'data/512_10K_content_image.hdf5')
cmd:option('-resize_to', 512)

cmd:option('-vgg_no_pad', false , 'Whether to use padding in convolutions in descriptor net.')
cmd:option('-gpu', 0)
cmd:option('-backend', 'nn', 'nn|cudnn')

local cmd_params = cmd:parse(arg)


---------------------------------------------------------
-- Parse params
---------------------------------------------------------

if cmd_params.backend == 'cudnn' then
  require 'cudnn'
  cudnn.benchmark = true
  print('cudnn')
end
cutorch.setDevice(cmd_params.gpu+1)

---------------------------------------------------------
-- Define helpers
---------------------------------------------------------

function load_image(image_path, scale_to)
  local img = image.load(image_path, 3)
  img = image.scale(img, scale_to, scale_to, 'bilinear')

  return img
end

-- File to save to
local out_file = hdf5.open(cmd_params.save_to, 'w')

-- Get list of images
local path_generator = paths.files(cmd_params.images_path, '.jpg')
local images = {}
for image_name in path_generator do
  table.insert(images, image_name)
end

-- Go
for i, image_name in ipairs(images) do
  print(image_name)
  local img = load_image(cmd_params.images_path ..'/' .. image_name, cmd_params.resize_to)

  out_file:write(image_name .. '_image', img:float())
  
  if (i%500) == 0 then
    collectgarbage()
  end

  xlua.progress(i,#images)
end
out_file:close()

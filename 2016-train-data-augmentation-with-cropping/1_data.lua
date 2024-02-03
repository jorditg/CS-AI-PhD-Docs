require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'string'

-- random seed to have reproducible results
print("Random seed used: " .. opt.seed)
torch.manualSeed(opt.seed)

local float_size = 4
local image_size_bytes = float_size*channels*width*height

print("Image size: " .. image_size_bytes/1024 .. " kB")
----------------------------------------------------------------------
print '==> loading dataset'
require 'DataSet'
trainData = DataSet(opt.online_augmentation)
testData = DataSet()
-- load all JPEG compressed images into memory
trainData:loadJPEG(opt.train_image_directory, opt.labels_file)
testData:loadJPEG(opt.test_image_directory, opt.labels_file)
----------------------------------------------------------------------
-- training/test size
trsize = trainData:size()
tesize = testData:size()
----------------------------------------------------------------------




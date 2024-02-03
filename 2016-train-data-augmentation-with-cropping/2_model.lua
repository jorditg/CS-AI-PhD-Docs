require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'cudnn'   

----------------------------------------------------------------------
print '==> define parameters'


----------------------------------------------------------------------
print '==> construct model'

model = nn.Sequential()

if opt.actual_model == 'load' then
  model = torch.load(opt.model_file)
else
-- input size 3x218x218
model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- 32x109x109
model:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- 64x54x54
model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))
-- 128x27x27
model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
-- 256x27x27
model:add(nn.SpatialAveragePooling(27,27))
model:add(nn.View(256))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(256, 256))
model:add(nn.ReLU())
model:add(nn.Linear(256, opt.noutputs))

print '==> define loss'
model:add(nn.LogSoftMax())

-- weight initialization
-- Must be done in nn (not implemented for cuDNN, conversion after)
print ('==> weight initialization. Method =>' .. weight_init_method)
model = require('weight-init')(model, weight_init_method)

-- conversion to cuDNN
print('==> Conversion from nn to cuDNN')
--cudnn.convert(model, cudnn)
-- find faster algorithm of cudnn
--cudnn.benchmark = true
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

criterion = nn.ClassNLLCriterion()

print '==> here is the loss function:'
print(criterion)

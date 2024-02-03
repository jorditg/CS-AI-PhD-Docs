require 'optim'

----------------------------------------------------------------------
-- GLOBAL OPTIONS
----------------------------------------------------------------------

local data_path = '../train-256x256/as-is'
opt = {
   seed = 1,
   save = './results/',
   batchSize = 32,
   plot = true,
   threads = 4,
   train_image_directory = data_path .. '/train',
   test_image_directory = data_path .. '/test',
   -- CSV filename containing the label information
   labels_file = data_path .. '/labels.csv',
   -- actual_model defines if the model is loaded (actual_model = 'load') or new defined (actual_model = 'new')
   actual_model = 'new',
   -- only if actual_mode is 'load' loads next file
   model_file = 'model.net',
   batchCalculationType = 'parallel',
   online_augmentation = true,
   noutputs = 5,
   save_training_images = false,
   train_images_hdf5_file = 'train.h5',
}

----------------------------------------------------------------------
-- DATASET IMAGE SIZES
----------------------------------------------------------------------
width = 218
height = 218
channels = 3

----------------------------------------------------------------------
-- MEAN AND STDDEV TO USE FOR NORMALIZATION
----------------------------------------------------------------------

RGBmeans = {
   97.4545 / 255, 
   67.8075 / 255, 
   48.4029 / 255
} 

RGBstddev = {
   74.07 / 255,
   53.71 / 255,
   43.55 / 255
}

----------------------------------------------------------------------
-- WEIGHT INITIALIZATION METHOD
----------------------------------------------------------------------
weight_init_method = 'kaiming'

----------------------------------------------------------------------
-- CLASSES NAMES
----------------------------------------------------------------------
classes = {'1','2','3','4','0'}

----------------------------------------------------------------------
-- OPTIMIZER CONFIGURATION
----------------------------------------------------------------------

-- Optimizer Adam
optimState = {
  learningRate = 0.0001,
  --weightDecay = opt.weightDecay -- only active with Adam with modified version!! (check modification after update!)
}
optimMethod = optim.adam

-- Optimizer RMSProp
--optimState = {
--  learningRate = opt.lr
  --weightDecay = opt.weightDecay
--}
--optimMethod = optim.rmsprop

-- Optimizer SGD
--optimState = {
--   learningRate = opt.lr,
--   weightDecay = opt.weightDecay,
--   momentum = opt.momentum,
--   dampening = 0.0,
--   learningRateDecay = 0.0,
--   nesterov = true
--}
--optimMethod = optim.sgd



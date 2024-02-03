require 'optim'

----------------------------------------------------------------------
-- GLOBAL OPTIONS
----------------------------------------------------------------------

local data_path = '../../3-input'
opt = {
   testRepeats = 5, -- number of different evaluations of the every image in the testing
   seed = 1,
   save = './results/',
   batchSize = 10,
   plot = true,
   threads = 8,
   test_image_directory = data_path .. '/test-724x724/private3',
   -- CSV filename containing the label information
   labels_file = data_path .. '/test-labels.csv',
   -- actual_model defines if the model is loaded (actual_model = 'load') or new defined (actual_model = 'new')
   actual_model = 'load',
   -- only if actual_mode is 'load' loads next file
   model_file = '../7-kappa-criterion/results/012-model-0.70/model-best.net',
   batchCalculationType = 'parallel',
   noutputs=5
}

----------------------------------------------------------------------
-- DATASET IMAGE SIZES
----------------------------------------------------------------------
width = 512
height = 512
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
-- CLASSES NAMES
----------------------------------------------------------------------
classes = {'0','1','2','3','4'}



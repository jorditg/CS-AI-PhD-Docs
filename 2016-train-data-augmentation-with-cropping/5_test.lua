----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'

----------------------------------------------------------------------
print '==> defining test procedure'

testBuffer = torch.FloatTensor(testData:imageSize(1))

-- test function
function test()
   function predict(inp)
   
   end

   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,tesize do
      if t % 100 == 0 then
          collectgarbage()
      end
      -- disp progress
      xlua.progress(t, tesize)

      -- get new sample
      local input = testBuffer
      local target = testData.labels[t]
      testData:get_data_image(t, input)
      normalize_image(input, RGBmeans, RGBstddev)
      
      -- here we receive the image without cropping in the "big" resolution
      -- we have toconvert it to the final nn resolution cropping it
      -- for making the test we pass through the network six cropped versions
      -- and make a voting scheme to decide the final classification result
      
      -- test all the cropped versions and save result in pred list
      local crop_type = {'c', 'tl', 'tr', 'bl','br'}
      local pred_avg = torch.FloatTensor(opt.noutputs,0)
      -- calculate geometric mean
      local N = table.getn(crop_type)
      print("N="..N)
      for i = 1,N do
        local im = image.crop(input, crop_type[i], width, height)
        image.save(i..".jpeg", im)
        --im = im:cuda()
        -- test sample
        local pred = model:forward(im)
        print(pred)
        pred_avg = pred_avg + pred
      end
      pred_avg:div(N)
      print("after div N:") 
      print(pred_avg)
      pred_avg:exp()
      print("after exp:")
      print(pred_avg)
      local denom = pred_avg:sum()
      print("denom:")
      print(denom)
      pred_avg:div(denom)
      print("after div denom:")
      print(pred_avg)
      pred_avg:log()
      print("after log:")
      print(pred_avg)
      os.exit()      
      confusion:add(pred_avg, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / tesize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end

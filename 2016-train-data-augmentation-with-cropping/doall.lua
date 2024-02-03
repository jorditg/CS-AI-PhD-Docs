----------------------------------------------------------------------
-- Jordi de la Torre
----------------------------------------------------------------------
require 'torch'

print '==> loading global parameters'
dofile '0_defs.lua'

-- nb of threads and fixed seed (for repeatable experiments)
print('==> switching to CUDA')
--require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile '1_data.lua'
dofile '2_model.lua'
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print '==> training!'

while true do
   train()
   collectgarbage()
   collectgarbage()
   --test()
   collectgarbage()
   collectgarbage()
end


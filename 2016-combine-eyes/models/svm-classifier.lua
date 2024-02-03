function createModel(nGPU)

   local classifier = nn.Sequential()
   classifier:add(nn.View(opt.features*opt.nimages))
   classifier:add(nn.Linear(opt.features*opt.nimages, nClasses))
   classifier:cuda()

   return classifier
end

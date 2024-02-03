require 'image'
require 'torch'


ParallelDecompressedDataSet = {}
ParallelDecompressedDataSet.__index = ParallelDecompressedDataSet

setmetatable(ParallelDecompressedDataSet, { 
    __call = function (cls, ...)
                return cls.new(...)
             end,
})

function ParallelDecompressedDataSet.new(compressed_dataset, n, c, h, w)
  local THREADS = 16
  
  local self = setmetatable({}, ParallelDecompressedDataSet)
  self.compressed = compressed_dataset
  self.channels = c
  self.batchSize = n
  self.height = h
  self.width = w

  -- compressed size and final size doesn't have to be necessarily equal.
  -- compressed size must be greater in case of cropping
  -- initial decompression buffer must be of the size of compressed images
  self.working_buffer = torch.FloatTensor(n, compressed_dataset:imageSize(1))
  
  self.original_height = self.working_buffer:size()[3]
  self.original_width = self.working_buffer:size()[4]
  
  -- output buffers are of external size
  self.output_buffers = torch.FloatTensor(2, n, c, w, h)
  self.labels_buffers = torch.ByteTensor(2, n)
  self.threads = require 'threads'
  self.active = 1
  self.threads.Threads.serialization('threads.sharedserialize')
  self.pool = self.threads.Threads(
    THREADS,
    function(idx)
      -- first load DataSet o be able to restore metatable in next function call
      require 'DataSet'
    end,
    function()
      compr = self.compressed
      -- trainDataTable looses his metatable inside thread. setmetatable to restore it
      -- and being able to call again its methods, later.
      setmetatable(compr, DataSet)
      dec = self.working_buffer
      out = self.output_buffers
      lab = self.labels_buffers
      original_width = self.original_width
      original_height = self.original_height
      width = self.width
      height = self.height
      N = self.batchSize
    end
  )
  return self
end

function ParallelDecompressedDataSet:
          launchNextDecompressionScalingNormalization(idx, from, to, mean, stddev, max_scaling)
  -- select the active buffer
  if self.active == 1 then
    buffer_idx = 2
  else
    buffer_idx = 1
  end
  
  (self.compressed):get_labels_subset((self.labels_buffers):select(1, buffer_idx), idx, from, to)
  
  for i=from,to do
    self.pool:addjob(
      function(buffer_idx, maxScaling)
        function crop(dest, src)
            -- resizing   
            local randomScaling = 1 + torch.uniform(-1,1)*maxScaling            
              
            local scaled_width = torch.round(randomScaling*original_width)
            local scaled_height = torch.round(randomScaling*original_height)
                 
            local fw = torch.random(1, (scaled_width - width))              
            local fh = torch.random(1, (scaled_height - height))
                        
            local scaled_image = image.scale(src, scaled_width, scaled_height)       
                    
            --local str = "rs=".. randomScaling .. "-fh=" .. fh .. "-fw=" .. fw .. "-sw=" .. scaled_width .. "-sh=" .. scaled_height .. "-w=" .. width .. "-h=" .. height
              --print(str)
            image.crop(dest, scaled_image, fw, fh, fw + width - 1, fh + height - 1)
        end    
        -- decompress image
        compr:get_decompressed_image(dec[i-from+1], idx, i)
        -- crop image
        crop(out:select(1, buffer_idx)[i-from+1], dec[i-from+1])
        -- normalize image
        normalize_image(out:select(1, buffer_idx)[i-from+1], mean, stddev)
      end,
      function() end,
      buffer_idx,
      max_scaling
    )
  end
end

function ParallelDecompressedDataSet:getNextDecompression()
  self.pool:synchronize()
  if self.active == 1 then 
     self.active = 2
  else
     self.active = 1
  end  
  self.data = self.output_buffers:select(1, self.active)
  self.labels = self.labels_buffers:select(1, self.active)
end



require 'image'
require 'torch'


DecompressedDataSet = {}
DecompressedDataSet.__index = DecompressedDataSet

setmetatable(DecompressedDataSet, { 
    __call = function (cls, ...)
                return cls.new(...)
             end,
})

function DecompressedDataSet.new(compressed_dataset, n, c, h, w)
  local self = setmetatable({}, DecompressedDataSet)
  self.compressed = compressed_dataset
  self.channels = c
  self.batchSize = n
  self.height = h
  self.width = w
  self.data = torch.FloatTensor(self.batchSize, self.channels, self.height, self.width) 
  self.labels = torch.ByteTensor(self.batchSize)
  return self
end

function DecompressedDataSet:decompress(idx, from, to)
      self.compressed:get_decompressed_subset(self.data, idx, from, to)
      self.compressed:get_labels_subset(self.labels, idx, from, to)
end





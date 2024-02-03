require 'image'
require 'torch'


DataSet = {}
DataSet.__index = DataSet

setmetatable(DataSet, { 
    __call = function (cls, ...)
                return cls.new(...)
             end,
})

function DataSet.new(augment)
  local self = setmetatable({}, DataSet)
  self.image_extension = '.jpeg'
  self.imagesSet = {}
  self.fileList = {}
  --self.labels = nil
  self.augment = augment or false
  return self
end

function DataSet:size()
    return self.dataset_size
end

function DataSet:filename(i)
  local val = self:map_idx(i)
  return self.fileList[val] 
end

-- return channels, height, width of image i
function DataSet:imageSize(i)
  local val = self:map_idx(i)
  local img_binary = self.imagesSet[val]
  local im = image.decompressJPG(img_binary)
  -- channels, height, width
  return im:size()[1], im:size()[2], im:size()[3] 
end

-- Auxiliary function that maps an augmented index when self.augment == true otherwise return the same index
function DataSet:map_idx(i)
  if self.augment then
    val = self.idx_map[i]
  else
    val = i
  end
  return val
end

-- Loads the CSV file of image name, labels into a lua table
function DataSet:loadLabels(labelsCSVfile)
    local labels = {}
    -- Load as a table the labels of each image
    for line in io.lines(labelsCSVfile) do
        local _, _, name, label = string.find(line, "(.*),(%d*)")
        labels[name .. self.image_extension ] = label
    end
    return labels
end

-- Returns the file list of files inside a directory that have the image extension
-- indicated as a paramater
function DataSet:listFiles(image_directory, image_extension)
  function image_extension_end (filename)
    if string.find(filename, self.image_extension .. "$") then
      return true
    end
    return false
  end
  fl = {}
  i = 1
  for f in paths.files(image_directory, image_extension_end) do
    fl[i] = f
    i = i + 1
  end
  return fl
end

-- Loads into memory all the JPEG compressed files of directory 'dir'
-- The required RAM memory is the same as the disk size
function DataSet:loadJPEG(dir, labelsCSVFile)
    self.fileList = self:listFiles(dir, image_extension)
    local number_files = table.getn(self.fileList)
    self.labels = torch.ByteTensor(number_files)
    local CSVLabels = self:loadLabels(labelsCSVFile)
    self.imagesSet = {}
    for i = 1, number_files do
        local fin = torch.DiskFile(dir .. "/" .. self.fileList[i], 'r')
        fin:binary()
        fin:seekEnd()
        local file_size_bytes = fin:position() - 1
        fin:seek(1)
        self.imagesSet[i] = torch.ByteTensor(file_size_bytes)
        fin:readByte(self.imagesSet[i]:storage())
        fin:close()
        -- Find labels for each file
        self.labels[i] = CSVLabels[self.fileList[i]]
        -- ATENTION!!
        -- Label 0 RETAGGED AS 5 !!!!!!!
        --if self.labels[i] == 0 then
        --  self.labels[i] = 5
        --end
        self.labels[i] = self.labels[i] + 1     
    end
    
    self.dataset_size = table.getn(self.imagesSet)
    if self.augment then
      self:create_idx_map()
    end
end

-- Decompresses a subset of the dataset referenced by the 'idx' vector
function DataSet:get_decompressed_subset(data_v, idx, from, to)
    -- 'from' and 'to' are indexes from vector idx for using a subset of idx
    -- default behaviour: selecting all indexes
    local from = from or 1
    local to = to or idx:size()[1]
    local n = table.getn(self.imagesSet)
    for i = from, to do
        local val = self:map_idx(idx[i])
        local img_binary = self.imagesSet[val]
        local im = image.decompressJPG(img_binary, 3, 'float')
        local j = i - from + 1
        if self.augment then
          generate_augmentation(data_v[{j, {}, {}, {}}], im)
        else
          data_v[{j, {}, {}, {}}] = im
        end
    end
end

-- Decompresses a image of the dataset referenced by the 'idx' vector
function DataSet:get_decompressed_image(buffer, idx, idx_val)
  local val = self:map_idx(idx[idx_val])
  local img_binary = self.imagesSet[val]
  local im = image.decompressJPG(img_binary, 3, 'float')
  if self.augment then
    generate_augmentation(buffer[{{}, {}, {}}], im)
  else
    buffer[{{}, {}, {}}] = im
  end
end

function DataSet:get_labels_subset(target_v, idx, from, to)
    for i = from, to do
        local j = i - from + 1
        local val = self:map_idx(idx[i])
        target_v[j] = self.labels[val]
    end
end

function DataSet:get_data_image(i, buffer)
    local val = self:map_idx(i)
    local img_binary = self.imagesSet[val]
    local im = image.decompressJPG(img_binary)
    if self.augment then
      generate_augmentation(buffer[{{},{},{}}], im)
    else
      buffer[{{},{},{}}] = im
    end        
end

function normalize_image_set(decompressed_image_set, RGBm, RGBs)
    local n = decompressed_image_set:size()[1]
    local channels = decompressed_image_set:size()[2]
    for i = 1, n do
        for c = 1,channels do
            decompressed_image_set[{i, c, {}, {}}]:add(-RGBm[c])
            decompressed_image_set[{i, c, {}, {}}]:div(RGBs[c])
        end
    end
end

function normalize_image(decompressed_image, RGBm, RGBs)
    local channels = decompressed_image:size()[1]
    for c = 1,channels do
        decompressed_image[{c, {}, {}}]:add(-RGBm[c])
        decompressed_image[{c, {}, {}}]:div(RGBs[c])
    end
end

-- Generates a new image from the input with rotation, horizontal flip and
-- vertical flip transformations of defined probaility.
-- wb: working buffer is a channelsxheightxwidth tensor
-- dest is where the final result is saved
-- src is where the initial image is located. Can be modified!! (because is used as a buffer)
function generate_augmentation(dest, src, hflip_prob, vflip_prob, rot_prob)
    local rp = rot_prob or 0.95 -- rotation probability
    local hfp = hflip_prob or 0.5 -- horizontal flip probability
    local vfp = vflip_prob or 0.5 -- vertical flip probability
    local enable_jitter = true -- enable contrast and brigthness augmentation
    local contrast_stddev = 0.1
    local brightness_stddev = 0.1
    
    local trans = 0 -- no transforms done yet
    
    local function get_idx_to_from()
      if math.fmod(trans, 2) == 0 then
        return dest, src
      else
        return src, dest
      end
    end
    
    --print("---")
    -- horizontal flip
    if torch.uniform() <= hfp then
        local to, from = get_idx_to_from()
        --print('hflip')
        image.hflip(to, from)
        trans = trans + 1
    end
    -- vertical flip
    if torch.uniform() <= vfp then
        --print('vflip')
        local to, from = get_idx_to_from()
        image.vflip(to, from)
        trans = trans + 1
    end
    -- rotation if enabled
    if torch.uniform() <= rp then
        local theta = 2.*math.pi*torch.uniform()
        --print('rotate'..theta)
        local to, from = get_idx_to_from()
        image.rotate(to, from, theta)
        trans = trans + 1
    end

    -- if the result is in src (pair number of transforms), copy back to dest
    if math.fmod(trans, 2) == 0 then
      --print('copy')
      dest:copy(src)
    end
    
    if enable_jitter then
      dest:mul(torch.normal(1, contrast_stddev))
      dest:add(torch.normal(0, brightness_stddev))     
    end
end

function DataSet:count_classes()
    local lab = self.labels
    -- Labels must be contiguous from 1 to 5 (0 not allowed)
    local count = torch.IntTensor({0,0,0,0,0})    
    for i = 1,lab:size(1) do
        count[lab[i]] = count[lab[i]] + 1
    end
    return count
end

-- Function for creating the mapping of the index
-- The maximal class is the 0. We will create a mapping with no number
-- augmentation of class 0. The other classes will be augmented to get
-- a equilibrated new augmented dataset with the same number of images
-- for every class. the new size will be 5*size_of_zero_class
function DataSet:create_idx_map()
    local count = self:count_classes()
    local ref = count:max() --count[5]
    
    local ent = torch.FloatTensor({
                                 math.floor(ref/count[1]), 
                                 math.floor(ref/count[2]),
                                 math.floor(ref/count[3]), 
                                 math.floor(ref/count[4]),
                                 math.floor(ref/count[5])
                                })
    local fra = torch.FloatTensor({ 
                                   ref / count[1], 
                                   ref / count[2], 
                                   ref / count[3], 
                                   ref / count[4],
                                   ref / count[5]
                                  })
    fra:add(-ent)

    -- the new augmented data size is
    self.dataset_size = ref * 5
    self.idx_map = torch.IntTensor(self.dataset_size)    
    local j = 1
    for i = 1, self.labels:size(1) do
        for k = 1,ent[self.labels[i]] do
            if j <= self.dataset_size then
              self.idx_map[j] = i
              j = j + 1
            else
              break
            end
        end
        if torch.uniform() < fra[self.labels[i]] then
            if j <= self.dataset_size then
              self.idx_map[j] = i
              j = j + 1
            else
              break
            end
        end    
    end
end



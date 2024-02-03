require 'torch'

dir = './128x128/'
filenames_file = dir .. 'predictions-filenames.th7'
labels_file = dir .. 'predictions-labels.th7'
predictions_file = dir.. 'predictions-tensor.th7'

out_file = dir .. 'output.csv'

files = torch.load(filenames_file)
labels = torch.load(labels_file)
preds = torch.load(predictions_file)

out = assert(io.open(out_file, "w"))

-- let's calculate the geometric mean
print("Remember to enable log first if they are not in log form!!")
--preds:log()
preds = preds:mean(2)[{{},1,{}}]
preds:exp()

-- let's normalize it to obtain probabilities
geom = torch.cdiv(preds,preds:sum(2):repeatTensor(1,5))

out:write('name,label,p1,p2,p3,p4,p5\n')
splitter = ','
for i=1,geom:size(1) do
  out:write(files[i])
  out:write(splitter)
  out:write(labels[i])
  out:write(splitter)
  for j=1,geom:size(2) do
    out:write(preds[i][j])
    if j == geom:size(2) then
      out:write("\n")
    else
      out:write(splitter)
    end
  end
end

out:close()


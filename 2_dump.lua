require 'image'
require 'io'

local num = 10000

local fullset = {}
fullset.size = num
--local data = torch.Tensor(fullset.size, 3, 27, 142)
local inputs = torch.Tensor(fullset.size, 27, 58)
local targets = torch.Tensor(fullset.size, 4)

for i = 1, num do
    local filename = './lines/' .. tostring(i) .. '.jpg'
    inputs[i] = image.load(filename, 1)

    local file = io.open('./lines/' .. tostring(i) .. '.txt', 'r')
    local labelstr = file:read()
    file:close()
    local label = torch.IntTensor(4)
    for j = 1, #labelstr do
       label[j] = tonumber(string.sub(labelstr, j, j)) + 1
    end
    targets[i] = label
end

fullset.inputs = inputs
fullset.targets = targets 

print('saving data...')
torch.save('fullset.dat', fullset)

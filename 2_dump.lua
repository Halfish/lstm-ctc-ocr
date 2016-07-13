require 'image'
require 'io'

local num = tonumber(arg[1]) or 10
print(num)

local fullset = {}
fullset.size = num
local height, width = 32, 255
local strlen = 20
local inputs = torch.Tensor(fullset.size, height, width)
local targets = torch.Tensor(fullset.size, strlen)

for i = 1, num do
    local filename = './lines/' .. tostring(i) .. '.jpg'
    inputs[i] = image.load(filename, 1)

    local file = io.open('./lines/' .. tostring(i) .. '.txt', 'r')
    local labelstr = file:read()
    file:close()
    local label = torch.IntTensor(strlen)
    for j = 1, #labelstr do
       label[j] = tonumber(string.sub(labelstr, j, j))
    end
    targets[i] = label
end

fullset.inputs = inputs
fullset.targets = targets

print('saving data...')
torch.save('fullset.dat', fullset)

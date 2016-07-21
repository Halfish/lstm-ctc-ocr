require 'image'
require 'io'

local num = tonumber(arg[1]) or 10
print(num)

local decoder_util = require 'decoder'
local decoder = decoder_util.create('codec_num.txt')

local fullset = {}
fullset.size = num
local height, width = 32, 255
local strlen = 20
local inputs = torch.Tensor(fullset.size, height, width)
local targets = {}

for i = 1, num do
    local filename = './lines/' .. tostring(i) .. '.jpg'
    inputs[i] = image.load(filename, 1)

    local file = io.open('./lines/' .. tostring(i) .. '.txt', 'r')
    local labelstr = file:read()
    labelstr = string.gsub(labelstr, '\n', '')
    file:close()
    table.insert(targets, decoder:str2target(labelstr))
end

fullset.inputs = inputs
fullset.targets = targets

print('saving data...')
torch.save('fullset.dat', fullset)

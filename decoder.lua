--[[
Here are three formats of information,

1. str = 'Name: Bruce Zhang'
2. target = {index of 'N',..., index of 'g'}
3. output: means the activation of neural network

We offer conversion functions between these three formats here.

]]--

require 'nn';
require 'io'
local path = require 'pl.path'

local decoder_util = {}     -- declare class decoder_util
decoder_util.__index = decoder_util     -- just syntactic sugar

function decoder_util.create(codec_dir, input_dims, max_steps)
    -- constructor for Class decoder_util
    local self = {}
    setmetatable(self, decoder_util)

    self.mapper, self.rev_mapper = decoder_util.get_mapper(codec_dir)
    -- self.vocab_size = #self.mapper -- A bug to figure out, this doesn't work
    self.vocab_size = 1     -- the whole vocabulary plus ' '(the blank symbol)
    for k, v in pairs(self.mapper) do
        self.vocab_size = self.vocab_size + 1
    end

    self.input_dims, self.max_steps = input_dims, max_steps

    return self
end

-- STATIC method, inspired by zhangzibin@github
-- get table with vary length from str, which may include chinese unicode character
function decoder_util.str2vocab(str)
    local vocab = {}
    local len  = #str
    local left = 0
    local arr  = {0, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc}
    local start = 1
    local wordLen = 0
    while len ~= left do
        local tmp = string.byte(str, start)
        local i   = #arr
        while arr[i] do
            if tmp >= arr[i] then break end
            i = i - 1
        end
        wordLen = i + wordLen
        local tmpString = string.sub(str, start, wordLen)
        start, left = start + i, left + i
        vocab[#vocab+1] = tmpString
    end
    return vocab
end

-- STATIC method, get chinese vocabulary mapper and rev_mapper from file,
function decoder_util.get_mapper(filename)
    -- read string from file
    local file = io.open(filename, 'r')
    local str = file:read('*all')
    -- str = string.gsub(str, '\n', '')
    local vocab = decoder_util.str2vocab(str)
    local vocab_set, mapper, rev_mapper = {}, {}, {}
    for i = 1, #vocab do
        vocab_set[vocab[i]] = true
    end
    for k, _ in pairs(vocab_set) do
        table.insert(rev_mapper, k)
        mapper[k] = #rev_mapper
    end
    return mapper, rev_mapper
end

function decoder_util:str2target(str)
    local target = decoder_util.str2vocab(str)
    local result = {}
    for i = 1, #target do
        local index = self.mapper[target[i]]
        if index ~= nil then
            result[#result + 1] = index
        end
    end
    -- assert(#result == #target) may failed when unforeseen chars occurred
    return result
end

function decoder_util:target2str(target)
    local str = ''
    for i = 1, #target do -- so target should be a table
        str = str .. self.rev_mapper[target[i]]
    end
    return str
end

-- parse network's output into standard target
function decoder_util:output2target(output)
    local _, index = output:max(2)
    assert(index:size()[1] == self.max_steps)
    -- remove the repeated elements
    local temp = {}
    for i = 1, self.max_steps do
        if (index[i][1]) ~= temp[#temp] then
            temp[#temp + 1] = index[i][1]
        end
    end
    -- remove blanks
    local target = {}
    for i = 1, #temp do
        if temp[i] ~= 1 then
            target[#target + 1] = temp[i] - 1   -- be careful here
        end
    end
    return target
end

-- if output is batchsize * max_steps * vocab_size
function decoder_util:outputs2targets(outputs)
    --outputs = nn.View(self.max_steps, self.vocab_size):forward(outputs:double())
    local batchsize = outputs:size()[1]
    local targets = {}
    for i = 1, batchsize do
        table.insert(targets, self:output2target(outputs[i]))
    end
    return targets
end

function decoder_util:output2str(output)
    return self.target2str(self.output2target(output))
end

-- only if these two labels are the same will we return true
function decoder_util:compare_target(target1, target2)
    if #target1 ~= #target2 then return false end
    for i = 1, #target1 do
        if target1[i] ~= target2[i] then return false end
    end
    return true
end

function decoder_util:compare_targets(targets1, targets2)
    assert(#targets1 == #targets2)
    local accu = 0
    local equalities = {}
    for i = 1, #targets1 do
        if self:compare_target(targets1[i], targets2[i]) then
            accu = accu + 1
            table.insert(equalities, true)
        else
            table.insert(equalities, false)
        end
    end
    return accu, equalities
end

return decoder_util

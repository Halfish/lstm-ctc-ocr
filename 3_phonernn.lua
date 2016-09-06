require 'rnn'
require 'nnx' -- for CTCCriterion
require 'math'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Option:')
cmd:option('-dataset', 'fullset.dat', 'fullset for training and validation')
cmd:option('-lang', 'number', 'to recognize numbers or Chinese Character')
cmd:option('-model', '', 'load existed model')
cmd:option('-splitrate', 0.8, 'split rate for fullset, trainset and validset')

-- training
cmd:option('-maxepochs', 1000, 'maximum epochs')
cmd:option('-gpuid', -1, 'which GPU to use, start from 0, and -1 means using CPU')
cmd:option('-batchsize', 50, 'batchsize')
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-momentum', 0.9, 'momentum for sgd')
cmd:option('cutoff', 5, 'cutoff for LSTM, solve gradient explosion problem')
cmd:option('dropout', 0.5, 'the probability of dropout activation value for model')
cmd:option('-savefreq', 10, 'save frequency')
cmd:option('-verbose', false, 'to print extra verbose information')
cmd:text()
local opt = cmd:parse(arg or {})

-- loading input
local fullset = torch.load(opt.dataset)
local trainset = {}
local validset = {}
trainset.targets = {}
validset.targets = {}
trainset.size = math.floor(fullset.size * opt.splitrate)
validset.size = fullset.size - trainset.size
trainset.inputs = fullset.inputs[{{1, trainset.size}, {}, {}}]
validset.inputs = fullset.inputs[{{trainset.size + 1, fullset.size}, {}, {}}]
for i = 1, trainset.size do trainset.targets[i] = fullset.targets[i] end
for i = 1, validset.size do validset.targets[i] = fullset.targets[i+trainset.size] end
print(string.format('train size = %d, valid size = %d', trainset.size, validset.size))

local decoder_util = require 'decoder'
local decoder = {}
if opt.lang == 'number' then
    decoder = decoder_util.create('codec_num.txt', 36, 255)
else
    if opt.lang == 'chinese' then
        decoder = decoder_util.create('codec.txt', 36, 2048)
    end
end

local vocab_size = decoder.vocab_size
local height, width = decoder.input_dims, decoder.max_steps

-- building model
local model = nn.Sequential()
model:add(nn.SplitTable(1))

local hiddensize = {height, 128}
local inputsize = hiddensize[1]

nn.FastLSTM.bn = true
nn.FastLSTM.usenngraph = true

for i = 2, #hiddensize do
    local rnn = nn.FastLSTM(inputsize, hiddensize[i])
    model:add(nn.Sequencer(rnn))
    model:add(nn.Sequencer(nn.ReLU()))
    model:add(nn.Sequencer(nn.BatchNormalization(hiddensize[i])))
    model:add(nn.Sequencer(nn.Dropout(opt.dropout)))
    inputsize = hiddensize[i]
end

model:add(nn.Sequencer(nn.Linear(hiddensize[#hiddensize], vocab_size)))
model:add(nn.JoinTable(1))
model:add(nn.View(decoder.max_steps, decoder.vocab_size))

local ctcCriterion = nn.CTCCriterion(true)

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
    local free, total = cutorch.getMemoryUsage(opt.gpuid + 1)
    print(string.format("GPU %d has %dM memory left, with %dM totally", opt.gpuid + 1,
            free/1000000, total/1000000))
    trainset.inputs = trainset.inputs:cuda()
    validset.inputs = validset.inputs:cuda()
    model = model:cuda()
    ctcCriterion = ctcCriterion:cuda()
end

for k, param in ipairs(model:parameters()) do
    param:uniform(-0.08, 0.08)
end

if opt.model ~= '' then
    model = torch.load(opt.model)
end

print(model)

-- training
function train()
    local total_loss = 0
    local total_accu = 0
    local shuffle = torch.randperm(trainset.size)
    local totalsize = math.ceil(trainset.size / opt.batchsize)
    local count = 1
    for t = 1, trainset.size, opt.batchsize do
        xlua.progress(count, totalsize)
        count = count + 1
        local actualsize = math.min(opt.batchsize + t - 1, trainset.size) - t + 1
        local inputs = torch.Tensor(actualsize, width, height):fill(0)
        if opt.gpuid >= 0 then
            inputs = inputs:cuda()
        end
        local targets = {}
        local sizes = {}
        for i = t, t+actualsize-1 do
            inputs[i - t + 1] = trainset.inputs[shuffle[i]]:t()
            table.insert(targets, trainset.targets[shuffle[i]])
            table.insert(sizes, width)
        end
        local outputs = model:forward(inputs)

        -- calc ctc losses
        local loss = ctcCriterion:forward(outputs, targets, torch.Tensor(sizes))
        local gradOutput = ctcCriterion:backward(outputs, targets)
        model:zeroGradParameters()
        model:backward(inputs, gradOutput)
        model:gradParamClip(opt.cutoff)
        model:updateGradParameters(opt.momentum)
        model:updateParameters(opt.lr)

        total_loss = total_loss + loss * actualsize
        local pred_targets = decoder:outputs2targets(outputs)
        local accu, _ = decoder:compare_targets(pred_targets, targets)
        total_accu = total_accu + accu
    end

    return total_loss / trainset.size, total_accu / trainset.size
end

-- evaluating
function eval()
    local total_loss = 0
    local total_accu = 0
    local shuffle = torch.randperm(validset.size)
    for t = 1, validset.size, opt.batchsize do
        local actualsize = math.min(opt.batchsize + t - 1, validset.size) - t + 1
        local inputs = torch.Tensor(actualsize, width, height):fill(0)
        if opt.gpuid >= 0 then
            inputs = inputs:cuda()
        end
        local targets = {}
        local sizes = {}
        for i = t, t+actualsize-1 do
            inputs[i-t+1] = validset.inputs[shuffle[i]]:t()
            table.insert(targets, validset.targets[shuffle[i]])
            table.insert(sizes, width)
        end
        local outputs = model:forward(inputs)
        local loss = ctcCriterion:forward(outputs, targets, torch.Tensor(sizes))
        total_loss = total_loss + loss * actualsize

        local pred_targets = decoder:outputs2targets(outputs)
        local accu, _ = decoder:compare_targets(pred_targets, targets)
        total_accu = total_accu + accu

   end
    return total_loss / validset.size, total_accu / validset.size
end

function showexample(num)
    -- randomly pick 10 pictures from validation set to see how things going
    num = num or 5
    local inputs = torch.Tensor(num, width, height)
    if opt.gpuid >= 0 then
        inputs = inputs:cuda()
    end
    local targets = {}
    for i = 1, num do
        local index = math.random(validset.size)
        inputs[i] = validset.inputs[index]:t()
        targets[#targets + 1] = validset.targets[index]
    end
    local outputs = model:forward(inputs)
    local pred_targets = decoder:outputs2targets(outputs)
    for i = 1, num do
        local pred_str = decoder:target2str(pred_targets[i])
        local str = decoder:target2str(targets[i])
        print(string.format('i = %d,\t%s,\tpred = %s, \t target = %s', i, tostring(pred_str == str), pred_str, str))
    end
end

do
    local stoppinglr = opt.lr * 0.0001
    local stopwatch = 0
    local last_v_loss = 100
    for epoch = 1, opt.maxepochs do
        -- training and validating
        local timer = torch.Timer()
        model:training()
        local loss, accu = train()
        model:evaluate()
        local v_loss, v_accu = eval()
        local format = 'epoch = %d, loss = %.4f, accu = %.4f, v_loss = %.4f, v_accu = %.4f, costed %.3f s'
        print(string.format(format, epoch, loss, accu, v_loss, v_accu, timer:time().real))

        if opt.verbose then showexample() end

        -- early-stopping
        if v_loss > last_v_loss then
            if stopwatch >= 8 then
                if opt.lr < stoppinglr then
                    break   -- minimum learning rate
                else
                    -- decrease the learning rate and recount the stopwatch again
                    opt.lr = opt.lr / 2
                    print('new learning rate is ' .. opt.lr)
                    stopwatch = 0
                end
            else
                stopwatch = stopwatch + 1 -- the valid loss didn't decrease for another time
            end
        end
        last_v_loss = v_loss

        -- dump model
        if epoch % opt.savefreq == 0 then
            local modelname = string.format('model_e%d_a%.2f.t7', epoch, v_accu)
            print('saving model as ' ..  modelname)
            torch.save(modelname, model)
        end
    end
end

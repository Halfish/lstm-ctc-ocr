require 'rnn'
require 'warp_ctc'
require 'math'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Option:')
cmd:option('-maxepochs', 1000, 'maximum epochs')
cmd:option('-dataset', 'fullset.dat', 'fullset for training and validation')
cmd:option('-model', '', 'load existed model')
cmd:option('-splitrate', 0.8, 'split rate for fullset, trainset and validset')
cmd:option('-lr', 0.001, 'learning rate')
cmd:option('-savefreq', 10, 'save frequency')
cmd:option('-gpuid', -1, 'which GPU to use, start from 0, and -1 means using CPU')
cmd:text()
local opt = cmd:parse(arg or {})

-- loading input
local fullset = torch.load(opt.dataset)
local trainset = {}
local validset = {}
trainset.size = math.floor(fullset.size * opt.splitrate)
validset.size = fullset.size - trainset.size
trainset.inputs = fullset.inputs[{{1, trainset.size}, {}, {}}]
trainset.targets = fullset.targets[{{1, trainset.size}, {}}]
validset.inputs = fullset.inputs[{{trainset.size + 1, fullset.size}, {}, {}}]
validset.targets = fullset.targets[{{trainset.size + 1, fullset.size}, {}}]
print(string.format('train size = %d, valid size = %d', trainset.size, validset.size))

-- building model
local vocab_size = 10 + 1   -- for this problem, 0-9 plus ' '(blank)
local height, width, strlen = 32, 255, 20
--local height, width = 31, 58

local model = nn.Sequential()
model:add(nn.SplitTable(1))

local hiddensize = {height, 256}
local inputsize = hiddensize[1]

for i = 2, #hiddensize do
    local rnn = nn.FastLSTM(inputsize, hiddensize[i])
    model:add(nn.Sequencer(rnn))
    model:add(nn.Sequencer(nn.Dropout(0.5)))
    inputsize = hiddensize[i]
end

model:add(nn.Sequencer(nn.Linear(hiddensize[#hiddensize], vocab_size)))
model:add(nn.JoinTable(1))

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
    local free, total = cutorch.getMemoryUsage(opt.gpuid + 1)
    print(string.format("GPU %d has %dM memory left, with %dM totally", opt.gpuid + 1,
            free/1000000, total/1000000))
    trainset.inputs = trainset.inputs:cuda()
    trainset.targets = trainset.targets:cuda()
    validset.inputs = validset.inputs:cuda()
    validset.targets = validset.targets:cuda()
    model = model:cuda()
end

for k, param in ipairs(model:parameters()) do
    param:uniform(-0.08, 0.08)
end

if opt.model ~= '' then
    model = torch.load(opt.model)
end

print(model)

-- decoder function
function maxdecoder(output)
    local _, index = output:max(2)
    index = index:resize(index:size(1))
    local predtarget = {}
    for i = 1, index:size()[1] do
        if index[i] ~= predtarget[#predtarget] then
            predtarget[#predtarget + 1] = index[i]
        end
    end
    local predstr = '#'
    for i, num in ipairs(predtarget) do
        if num == 1 then
            predstr = predstr .. '_'
        else
            predstr = predstr .. tostring(num-2)
        end
    end
    return predstr .. '#'
end

function pred(outputs, targets)
    local count = 0
    local _, index = nn.View(width, 11):forward(outputs:double()):max(3)
    local preds = {}
    for i = 1, #targets do
        local temp = {}
        for j = 1, width do
            if index[i][j][1] ~= temp[#temp] then
                temp[#temp + 1] = index[i][j][1]
            end
        end
        local predtarget = {}
        for j = 1, #temp do
            if temp[j] ~= 1 then
                predtarget[#predtarget + 1] = temp[j]
            end
        end
        if #targets[i] == #predtarget then
            local flag = true
            for j = 1, #predtarget do
                if (targets[i][j] + 2) ~= predtarget[j] then
                    flag = false
                    break
                end
            end
            if flag then count = count + 1 end
        end
    end
    return count
end

-- training
function train()
    local total_loss = 0
    local total_accu = 0
    local shuffle = torch.randperm(trainset.size)
    local batchsize = 50
    local totalsize = math.ceil(trainset.size / batchsize)
    local count = 1
    for t = 1, trainset.size, batchsize do
        xlua.progress(count, totalsize)
        count = count + 1
        local actualsize = math.min(batchsize + t - 1, trainset.size) - t + 1
        local inputs = torch.Tensor(actualsize, width, height):fill(0)
        if opt.gpuid >= 0 then
            inputs = inputs:cuda()
        end
        local targets = {}
        local sizes = {}
        for i = t, t+actualsize-1 do
            inputs[i - t + 1] = trainset.inputs[shuffle[i]]:t()
            local targetstr = trainset.targets[shuffle[i]]
            local target = {}
            for j = 1, targetstr:size(1) do
                table.insert(target, targetstr[j])
            end
            --table.insert(targets, target)
            table.insert(targets, trainset.targets[shuffle[i]]:totable())
            table.insert(sizes, width)
        end
        local outputs = model:forward(inputs)

        -- re-align the activation values for ctc
        local acts = outputs:clone():fill(0)
        for i = 1, actualsize do
            for j = 1, width do
                acts[i + actualsize * (j - 1)] = outputs[j + width * (i - 1)]
            end
        end

        -- calc ctc losses
        local grads = outputs:clone():fill(0)
        local losses = {}
        if opt.gpuid >= 0 then
            acts = acts:cuda()
            grads = grads:cuda()
            losses = gpu_ctc(acts, grads, targets, sizes)
        else
            acts = acts:float()
            grads = grads:float()
            losses = cpu_ctc(acts, grads, targets, sizes)
        end

        -- gradient explosion problem
        grads:clamp(-5, 5)

        -- re-align gradients for back-prop
        local gradients = grads:clone():fill(0)
        for i = 1, actualsize do
            for j = 1, width do
                gradients[j + width * (i - 1)] = grads[i + actualsize * (j - 1)]
            end
        end

        model:zeroGradParameters()
        model:backward(inputs, gradients)
        model:updateGradParameters(0.9)
        model:updateParameters(opt.lr)

        local count = pred(outputs, targets)
        total_accu = total_accu + count

        for i = 1, #losses do
            total_loss = total_loss + losses[i]
        end
    end

    return total_loss / trainset.size, total_accu / trainset.size
end

-- evaluating
function eval()
    local total_loss = 0
    local total_accu = 0
    local shuffle = torch.randperm(validset.size)
    local batchsize = 50
    for t = 1, validset.size, batchsize do
        local actualsize = math.min(batchsize + t - 1, validset.size) - t + 1
        local inputs = torch.Tensor(actualsize, width, height):fill(0)
        if opt.gpuid >= 0 then
            inputs = inputs:cuda()
        end
        local targets = {}
        local sizes = {}
        for i = t, t+actualsize-1 do
            inputs[i-t+1] = validset.inputs[shuffle[i]]:t()
            local targetstr = validset.targets[shuffle[i]]
            local target = {}
            for j = 1, targetstr:size(1) do
                table.insert(target, targetstr[j])
            end
            table.insert(targets, target)
            table.insert(sizes, width)
        end
        local outputs = model:forward(inputs)
        local acts = outputs:clone():fill(0)
        for i = 1, actualsize do
            for j = 1, width do
                acts[i + actualsize * (j - 1)] = outputs[j + width * (i - 1)]
            end
        end

        local grads = torch.Tensor() -- don't need gradients for validation here
        local losses = {}

        if opt.gpuid >= 0 then
            acts = acts:cuda()
            grads = grads:cuda()
            losses = gpu_ctc(acts, grads, targets, sizes)
        else
            acts = acts:float()
            grads = grads:float()
            losses = cpu_ctc(acts, grads, targets, sizes)
        end

        local count = pred(outputs, targets)
        total_accu = total_accu + count

        for i = 1, #losses do
            total_loss = total_loss + losses[i]
        end
    end
    return total_loss / validset.size, total_accu / validset.size
end

function target2str(target)
    str = '#'
    for i = 1, target:size()[1] do
        str = str .. (target[i])
    end
    str = str .. '#'
    return str
end

function showexample()
    -- randomly pick 10 pictures to see how things going
    local inputs = torch.Tensor(1, width, height)
    if opt.gpuid >= 0 then
        inputs = inputs:cuda()
    end
    for i = 1, 99 do
        local index = math.random(validset.size)
        inputs[1] = validset.inputs[index]:t()
        local output = model:forward(inputs)
        print(string.format('i = %d,\t pred = %s,\t target = %s', i, maxdecoder(output), target2str(validset.targets[index])))
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
        local format = 'epoch = %d, loss = %.4f, accu = %.2f, v_loss = %.4f, v_accu = %.2f, costed %.3f s'
        print(string.format(format, epoch, loss, accu, v_loss, v_accu, timer:time().real))

        --showexample()

        -- early-stopping
        if v_loss > last_v_loss then
            if stopwatch >= 3 then
                if opt.lr < stoppinglr then
                    break   -- minimum learning rate
                else
                    -- decrease the learning rate and recount the stopwatch again
                    opt.lr = opt.lr / 2
                    stopwatch = 0
                end
            else
                stopwatch = stopwatch + 1 -- the valid loss didn't decrease for another time
            end
        end

        -- dump model
        if epoch % opt.savefreq == 0 then
            local modelname = string.format('model_e%d_a%.2f.t7', epoch, v_accu)
            print('saving model as ' ..  modelname)
            torch.save(modelname, model)
        end
    end
end

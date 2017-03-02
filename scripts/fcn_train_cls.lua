require 'torch'
require 'optim'
require 'pl'
require 'paths'

local fcn = {}

local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
-- put the labels for each batch in targets
local targets = torch.Tensor(opt.batchSize * opt.labelSize * opt.labelSize)

local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()


-- training function
function fcn.train(inputs_all)
    cutorch.synchronize()
    epoch = epoch or 1
    local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
    local dataBatchSize = opt.batchSize
    local input = inputs_all[1]
    local target = inputs_all[2] 

    -- TODO: implemnet the training function
    local opfunc = function(x)	
	collectgarbage()
        --x = parameters
        --x = x:cuda()
	gradParameters:zero()

        local outputs = model_FCN:forward(input)
        local loss = criterion:forward(outputs, target)
        local dloss_doutputs = criterion:backward(outputs, target)
        model_FCN:backward(input, dloss_doutputs)

        return loss, gradParameters
    end
    local _, loss = optim.sgd(opfunc, parameters, optimState) 
    batchNumber = batchNumber + 1

    cutorch.synchronize(); 
    
    collectgarbage();
    print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f TrainingLoss %.4f'):format(epoch, batchNumber, opt.epochSize, sampleTimer:time().real, dataLoadingTime, loss[1]))
    dataTimer:reset()

end


return fcn


